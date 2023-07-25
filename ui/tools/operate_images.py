import os
import PIL.Image as Image
from tqdm import tqdm
import logging
from typing import List, Literal, Any
from datetime import datetime
import shutil


CLUSTER_DIR_PREFIX = "cluster"
BIT = 6  # 假设聚类数不超过999999
DELIMITER = "-"  # 用于分隔聚类序号和图片名字


def change_ext_with_old_name(path: str, new_ext: str) -> str:
    """ 将path的扩展名改成new_ext """
    path_without_ext, ext = os.path.splitext(path)
    new_path = path_without_ext + new_ext
    return new_path

def change_name_with_old_ext(path: str, new_name: str) -> str:
    """ 保留path扩展名，更改其文件名为new_name """
    path_without_ext, ext = os.path.splitext(path)
    # 取出除最后一个部分的路径，将其与新名字join一起
    new_path = os.path.join( os.path.dirname(path_without_ext), new_name + ext )
    return new_path


# TODO: 最好加上修改时间
# TODO: 有可能因为缓存失败照成图片无法显示，考虑返回缓存后的图片地址
# TODO: 或许能直接返回PIL.Image对象，而不是把图片保存在磁盘上
def cache_images_file(cache_images_list: List[str], cache_dir: str, resolution: int=512 ) -> bool:
    """
    调用pillow，将重复的图片缓存到同路径下的一个cache文件夹中，分辨率最大为resolution,与前面图片名字一一对应
    如果存在同名文件就不缓存了

    Args:
        cache_images_list (List[str]): 需要缓存的图片路径列表，建议为绝对路径
        cache_dir (str): 缓存的文件夹路径
        resolution (int, optional): 缓存的分辨率. Defaults to 512.

    Returns:
        bool: 发生缓存错误则为True，否则为False
    """
    
    # 建一个文件夹
    os.makedirs(cache_dir, exist_ok=True)

    exist_error = False  # TODO: 先暂时用这个告知发生缓存错误，通知画廊用原图；最好是把返回缓存后的图片地址返回去，失败的用源地址
    
    print("缓存缩略图中，caching...")
    for image_path in tqdm(cache_images_list):
        image_name = os.path.basename(image_path)
        # 已经存在同名缓存文件，就不缓存了
        if not os.path.exists( os.path.join(cache_dir, image_name) ):
            try:
                with Image.open( image_path  ) as im:
                    im.thumbnail( (resolution, resolution) )
                    im.save( os.path.join(cache_dir, image_name) )
            except Exception as e:
                exist_error = True
                logging.error(f"缓存 {image_path} 失败, error: {e}")
    print(f"缓存完成: {cache_dir}\nDone!")

    return exist_error

# TODO: 返回操作后图片的新绝对路径
def operate_images_file(
        images_dir: str,
        clustered_images_list: List[List[str]],
        extra_file_ext_list: List[str] = [],
        copy_to_subfolder_file_list: List[str] = [],
        operation: Literal["copy", "move", "rename", "remove"] = "copy",
    ):
    """ 依据clustered_images_list中聚类情况，对images_dir下图片以及同名的扩展名为extra_file_ext_list中的文件进行操作

    Args:
        images_dir (str): 需要重命名的图片所在文件夹
        clustered_images_list (List[List[str]]): 聚类后的图片列表，每个子列表为一个聚类，每个子列表元素为图片名（不带路径）
        extra_file_ext_list (List[str], optional): 需要一起处理的文件的扩展名字符串，如".txt"。 Defaults to [].
            注意这里的扩展名是相对于图片名字而言的扩展名
            如果图片名为"abc.jpg.pn"，extra_file_ext_list = [".wd14.txt"]
            则匹配的文件名为"abc.jpg.wd14.txt"
            只有在operation为copy或move时才有效
        copy_to_subfolder_file_list (List[str], optional): 会被复制到每一个子文件夹的文件的全名字符串，如"abc.txt"。 Defaults to [].
            只有在operation为copy或move时才有效
        operation (Literal["copy", "move", "rename", "remove"], optional): 操作类型，copy为复制，move为移动，rename为重命名，remove为删除. Defaults to "copy".
    """

    # 获取当前时间
    time_now = datetime.now().strftime('%Y%m%d%H%M%S')

    def my_os_remove(x: str, _: Any):
        os.remove(x)

    if operation == "copy":
        operate_func = shutil.copy2
        operate_desc_str = "复制原图中，copying..."
    elif operation == "move":
        operate_func = shutil.move
        operate_desc_str = "移动原图中，moving..."
    elif operation == "rename":
        operate_func = os.rename
        operate_desc_str = "重命名原图中，renaming..."
    elif operation == "remove":
        operate_func = my_os_remove
        operate_desc_str = "删除原图中，removing..."
    else:
        raise ValueError(f"operation参数错误，应为copy, move, rename中的一个，而不是{operation}")
    
    total_len = sum( [len(cluster) for cluster in clustered_images_list] )  # 总共需要操作的图片数
    p_bar = tqdm(total=total_len, desc=operate_desc_str)

    for cluster_index, cluster in enumerate(clustered_images_list):

        operate_prefix = f"{CLUSTER_DIR_PREFIX}{DELIMITER}{cluster_index:0{BIT}d}"  # 带有6位聚类序号

        # 需要加时间戳来避免重名
        cluster_son_dir = os.path.join(images_dir, f"{CLUSTER_DIR_PREFIX}{DELIMITER}{time_now}", operate_prefix)
        
        def create_subfolder_and_copy():
            """ 创建子文件夹并复制copy_to_subfolder_file_list中指定的文件到子文件夹中 """
            os.makedirs(cluster_son_dir, exist_ok=True)
            # 如果给出了extra_file_name_list，就复制一份到每个子文件夹
            for need_copy_file_name in copy_to_subfolder_file_list:
                need_copy_file_old_path = os.path.join(images_dir, need_copy_file_name)
                need_copy_file_new_path = os.path.join(cluster_son_dir, need_copy_file_name)
                if not os.path.exists(need_copy_file_old_path):
                    # 在类似查重，或者依靠tag文本聚类时候，用户不一定需要附带文件，所以不存在是正常的，这里只报个info记录下表示正常运行
                    logging.info(f"文件{need_copy_file_old_path}不存在，跳过")
                    continue
                try:
                    shutil.copy2(need_copy_file_old_path, need_copy_file_new_path)
                except Exception as e:
                    logging.error(f"复制文件{need_copy_file_old_path}到{cluster_son_dir}文件夹时失败, error: {e}")
            
        # 只有复制或移动才需要建立子文件夹
        if operation in ["copy", "move"]:
            create_subfolder_and_copy()

        for image_index, image_name in enumerate(cluster):

            # 图片的路径
            image_name = image_name
            image_path = os.path.join(images_dir, image_name)

            # 对应的扩展
            extra_file_name_list = []
            for ext in extra_file_ext_list:
                # 同名的附带文件路径，和名字（带扩展名）
                extra_file_name_list.append( change_ext_with_old_name(image_name, ext) )

            def get_new_path(old_name: str) -> str:
                """
                根据操作类型，获取新的路径
                
                输入的old_name可以是图片名，也可以是同名的附带文件名；但注意是名字而不是路径

                Returns:
                    str: 新的路径，如果是删除操作则返回""
                """
                if operation in ["rename"]:
                # 重命名图片路径并保留扩展名
                    # 必须保证新名字不和任何一个原来的旧图片名字重复，否则重命名会发生错误
                    new_path = os.path.join(images_dir, f"{operate_prefix}_{old_name}")  # 在原有基础上加上聚类序号
                elif operation in ["copy", "move"]:
                    # 复制或移动图片路径，保存原名
                    new_path = os.path.join(cluster_son_dir, old_name)  # 不改名
                elif operation in ["remove"]:
                    new_path = "" # 删除操作不需要新路径
                else:
                    raise ValueError(f"operation参数错误，应为 copy, move, rename, remove 中的一个，而不是{operation}")
                
                return new_path
            

            # 如果连图片都不在，剩下都别操作了
            if not os.path.exists(image_path):
                logging.warning(f"图片 {image_path} 不存在，将不会对其进行任何操作")
                p_bar.update(1)
                continue

            # 操作图片
            try:
                new_image_path = get_new_path(image_name)
                operate_func(image_path, new_image_path)
            except Exception as e:
                logging.error(f"操作 {image_name} 失败, error: {e}")
                # 图片操作失败，剩下的就别操作了
                continue
            
            # 操作附带文件
            for extra_file_name in extra_file_name_list:
                extra_file_path = os.path.join(images_dir, extra_file_name) 
                if os.path.exists(extra_file_path):
                    try:
                        new_extra_file_path = get_new_path(extra_file_name)
                        operate_func(extra_file_path, new_extra_file_path)
                    except Exception as e:
                        logging.error(f"操作 {extra_file_name} 失败, error: {e}")
                else:
                    # 在类似查重，或者依靠tag文本聚类时候，用户不一定需要附带文件，所以不存在是正常的，这里只报个info记录下表示正常运行
                    logging.info(f"文件 {extra_file_name} 不存在，跳过")

            p_bar.update(1)
    

    if operation in ["copy", "move"]:
        print(f"操作完成: {images_dir}\nDone!")  # 拷贝和移动创建了新的文件夹，所以要显示给用户
    else:
        print("操作完成  Done!")
