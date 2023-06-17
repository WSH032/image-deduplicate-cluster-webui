import requests
import os
from tqdm import tqdm
import logging
import hashlib
import zipfile
import shutil
import torch


install_info = "TensorRT 8.6 GA for Windows 10 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7 and 11.8 ZIP Package"
tensorrt_download_url = "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip"
download_dir = ".cache"
filename = os.path.basename(tensorrt_download_url)
file_path = os.path.join(download_dir, filename)
chunk_size = 8192


def main():
    print(f"{'#' * 20}")
    print(f"\n专门为 {install_info} 编写")
    print(f"\n如果不符合所需环境，请参考 https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html 完成安装")
    print(f"\n缓存和下载目录在： { os.path.abspath(download_dir) }，支持断点传输")
    print(f"\n{'#' * 20}")


    os.makedirs(download_dir, exist_ok=True)

    # 建立requests的session
    with requests.Session() as session:
        
        def get_http_size():
            with session.get(tensorrt_download_url, stream=True) as r:
                r.raise_for_status()
                # 显示文件大小,mb为单位
                total_size = int( r.headers['Content-Length'] )
                return total_size
        # 获取本地一下载文件大小
        def get_local_size():
            try:
                file_size = os.path.getsize(file_path)
            except FileNotFoundError:
                file_size = 0
            return file_size
        
        http_size = get_http_size()
        local_size = get_local_size()
        
        print(f"需要下载{(http_size/1024/1024): .2f}MB,本地已存在文件大小{(local_size/1024/1024): .2f}MB")

        def try_download():
            headers = {'Range': f'bytes={local_size}-'}
            with session.get(tensorrt_download_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                # 显示文件大小,mb为单位
                total_size = int( r.headers['Content-Length'] )
                # 写入文件
                with open(file_path, 'ab') as f:
                    with tqdm.wrapattr(f, "write", miniters=1, total=total_size, desc=filename) as fout:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            fout.write(chunk)
                            fout.flush()

        if local_size < http_size:
            try_download()
        elif local_size > http_size:
            logging.warning("本地文件大于服务器文件,删除本地文件")
            os.remove(file_path)
            try_download()
        else:
            pass
        
        # 取本地文件头部和尾部的数据,与服务器文件头部和尾部的数据进行md5校验
        def get_local_head_tail():
            with open(file_path, 'rb') as f:
                head = f.read(1024)
                f.seek(-1024,2)
                tail = f.read(1024)
            return head,tail
        def get_http_head_tail():
            headers = {'Range': f'bytes=0-1023'}  # 注意，这里是1023，不是1024，因为是左闭右闭区间
            with session.get(tensorrt_download_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                head = r.content
            headers = {'Range': f'bytes=-1024'}
            with session.get(tensorrt_download_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                tail = r.content
            return head,tail
        
        local_test_data = get_local_head_tail()
        http_test_data = get_http_head_tail()

        def md5_test():
            local_md5 = hashlib.md5()
            for data in local_test_data:
                local_md5.update(data)

            http_md5 = hashlib.md5()
            for data in http_test_data:
                http_md5.update(data)

            print("本地文件头部和尾部的md5:",local_md5.hexdigest())
            print("服务器文件头部和尾部的md5:",http_md5.hexdigest())
            
            return local_md5.hexdigest() == http_md5.hexdigest()
        
        md5_test_result = md5_test()
        if not md5_test_result:
            raise Exception("警告:md5校验失败,请删除本地文件后重新运行本程序")
        else:
            pass

        # 解压文件
        need_extract = False
        for path in ('lib', 'python'):
            if not os.path.exists( os.path.join(download_dir, 'TensorRT-8.6.1.6', path) ):
                need_extract = True
                break
        if need_extract:
            print("开始解压文件")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir, members=[f for f in zip_ref.namelist()
                                                        if f.startswith( ('TensorRT-8.6.1.6/lib/', 'TensorRT-8.6.1.6/python/') )
                                            ]
                            )
            print("解压文件完成")
        

        torch_path = torch.__path__[0] # type: ignore
        torch_lib_path = os.path.join(torch_path,"lib")
        print("将安装至以下torch_path:",torch_lib_path)

        # 拷贝文件
        print("开始拷贝文件")
        tensorrt_lib_path = os.path.join(download_dir,"TensorRT-8.6.1.6", "lib")

        tensorrt_lib_files_list = os.listdir(tensorrt_lib_path)
        torch_lib_files_list = os.listdir(torch_lib_path)

        dup = set(tensorrt_lib_files_list) & set(torch_lib_files_list)
        if dup:
            print(f"警告:{torch_lib_path}中以下文件已存在")
            for file in dup:
                print(file)
            print("似乎你已经安装过tensorrt")
            print(f"如果你仍然想重新安装，请关闭python后,自行将{os.path.abspath(tensorrt_lib_path)}内的文件拷贝至{os.path.abspath(torch_lib_path)}")
        else:
            print("开始拷贝文件")
            shutil.copytree(tensorrt_lib_path, torch_lib_path, dirs_exist_ok=True)
            print("拷贝文件完成")
            print(f"你可以自行删除{os.path.abspath(download_dir)}内的文件")

if __name__ == "__main__":
    main()
