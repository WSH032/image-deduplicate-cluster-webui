# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:38:31 2023

@author: WSH
"""



import gradio as gr
from imagededup.methods import PHash
import toml
import bisect
import os
from tqdm import tqdm
from PIL import Image
import logging
from typing import Callable, Dict, List, Union, Tuple

# import pandas as pd
# encodings = {}  # 用于记录图片编码  #已经弃用，不再需要编码

choose_image_index = "0:0"  # 用于记录当前点击了画廊哪个图片
cluster_list = []  # list[ list[str] ] 用于记录重复图片的聚类结果
confirmed_images_dir = ""  # 用于记录查重结果对应的文件夹路径，防止使用者更改导致错误
images_info_list = []  # list[dict] 用于记录重复图片的属性信息
duplicates = {}  # Dict[str, List] 记录最原始的查重结果，将用于自动选择的启发式算法



def find_duplicates_images(images_dir: str, use_cache:bool):
    
    global confirmed_images_dir, cluster_list, images_info_list
    
    # 全局变量
    confirmed_images_dir = images_dir
    
    def cluster_duplicates(images_dir: str) -> list:
        """ 返回该目录下重复图像聚类，为一个集合列表，每个集合内为重复图像名字（不包含路径） """
        
        # 载入模型
        phasher = PHash()

        # 不再需要编码了，直接从路径找即可
        # global encodings
        # encodings = phasher.encode_images(image_dir=images_dir)
        
        global duplicates
        # 查找重复
        duplicates = phasher.find_duplicates(image_dir=images_dir) # type: ignore
        # 只保留确实有重复的图片,并弄成集合列表
        indeed_duplicates_set = [set(v).union({k}) for k, v in duplicates.items() if v]
        # 将重复的图片聚类
        cluster_list = []
        for s in indeed_duplicates_set:
            for m in cluster_list:
                if s & m:
                    m.update(s)
                    break
            else:
                cluster_list.append(s)
        # 把内部的集合改为列表，让其有序
        cluster_list = [ list(s) for s in cluster_list] 
        return cluster_list
    
    # 全局变量
    cluster_list = cluster_duplicates(images_dir)
    

    def cache_image( input_dir: str, cluster_list: list, resolution: int=512 ):
        """ 如果使用缓存，就调用pillow，将重复的图片缓存到同路径下的一个cache文件夹中，分辨率最大为512,与前面图片名字一一对应 """
        
        # 建一个文件夹
        cache_dir = os.path.join(input_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        print("缓存缩略图中，caching...")
        for cluster in tqdm(cluster_list):
            for image_name in cluster:
                with Image.open( os.path.join( input_dir, image_name ) ) as im:
                    im.thumbnail( (resolution, resolution) )
                    im.save( os.path.join(cache_dir, image_name) )
        print(f"缓存完成: {cache_dir}\nDone!")
                    
    
    def cluster_to_gallery(images_dir: str, cluster_list: list) -> list:
    # 将图片的绝对路径和索引合成一个元组，最后把全部元组放入一个列表，向gradio.gallery传递
    
        # 合成元组列表
        images_tuple_list = []
        parent_index = 0
        for parent in cluster_list:
            son_index = 0
            for son in parent:
                images_tuple_list.append( (os.path.join(images_dir, son), f"{parent_index}:{son_index}") )
                son_index += 1
            parent_index += 1
        
        return images_tuple_list
    
    # 如果使用缓存就缓存图像
    if use_cache:
        cache_image(images_dir, cluster_list, resolution=512)
    
    # 如果用缓存就展示缓存图
    gallery_images_dir = images_dir if not use_cache else os.path.join( images_dir, "cache" )
    images_tuple_list = cluster_to_gallery( gallery_images_dir, cluster_list)
    
    def get_images_info(images_dir: str, cluster_list: list) -> list[dict]:
        # 获取图片的信息成一个字典，放入列表中
        
        images_info_list = []
        for cluster in cluster_list:
            for image_name in cluster:
                image_path = os.path.join(images_dir, image_name)
                with Image.open( image_path ) as im:
                    size_MB = round( os.path.getsize(image_path)/1024/1024, 2 )
                    image_info_dict = {"resolution(l,w)":im.size,
                                       "size":f"{size_MB} MB",
                                       "format":im.format,
                                       "filename":im.filename                   # type: ignore
                    }
                    images_info_list.append(image_info_dict)

        return images_info_list
    
    # 全局变量
    images_info_list = get_images_info(images_dir, cluster_list)
    
    # 根据聚类数,生成如下字典
    """
    "1":[]
    "2":[]
    ...
    """
    # delet_images_dict = {f"{i}": [] for i in range( len(cluster_list) )}
    # delet_images_str = toml.dumps(delet_images_dict)
    # return images_tuple_list, delet_images_str
    
    return images_tuple_list, ""
    


def confirm(delet_images_str: str) -> Tuple[str, dict, dict]:
    
    # 尝试将字符串载入成字典
    try:
        delet_images_dict = toml.loads(delet_images_str)
    except Exception as e:
        toml_error_str = f"{delet_images_str}\n待删除列表toml格式错误，请修正\nerror: {e}"
        return toml_error_str, gr.update( variant="secondary" ), gr.update( variant="secondary" )
    
    # 把删除标志如 "0:1" 分成 "0" 和 "1"
    [parent_index_str, son_index_str] = choose_image_index.split(":")[0:2]

    # 如果字典里没有这个键，就给他赋值一个列表
    if delet_images_dict.get(parent_index_str) is None:
        delet_images_dict[parent_index_str] = [int(son_index_str)]
    # 如果有这个键，并且该键对应的列表值中不含有这个子索引，就按二分法把子索引插入到该键对应的列表值中
    else:
        if int(son_index_str) not in delet_images_dict[parent_index_str]:
            bisect.insort(
                delet_images_dict[parent_index_str], int(son_index_str))

    # 按键名排序
    delet_images_dict = dict(
        sorted(delet_images_dict.items(), key=lambda x: int(x[0])))
    
    
    return toml.dumps(delet_images_dict), gr.update( variant="secondary" ), gr.update( variant='primary' )


def cancel(delet_images_str: str) -> Tuple[str, dict, dict]:
    
    # 尝试将字符串载入成字典
    try:
        delet_images_dict = toml.loads(delet_images_str)
    except Exception as e:
        toml_error_str = f"{delet_images_str}\n待删除列表toml格式错误，请修正\nerror: {e}"
        return toml_error_str, gr.update( variant="secondary" ), gr.update( variant="secondary" )
    
    # 把删除标志如 "0:1" 分成 "0" 和 "1"
    [parent_index_str, son_index_str] = choose_image_index.split(":")[0:2]
    # 如果有这个键，就执行操作
    if delet_images_dict.get(parent_index_str) is not None:
        # 如果列标中有这个子索引，就删掉这个子索引
        if int(son_index_str) in delet_images_dict[parent_index_str]:
            delet_images_dict[parent_index_str].remove(int(son_index_str))
            # 如果删去后列表为空，则把相应的键一起删了
            if not delet_images_dict[parent_index_str]:
                delet_images_dict.pop(parent_index_str, None)
    
    # 按钮变色，按谁谁白，另一个红
    gr_confirm_button = gr.update( variant='primary' )
    gr_cancel_button = gr.update( variant="secondary" )
    
    return toml.dumps(delet_images_dict), gr_confirm_button, gr_cancel_button

# https://github.com/WSH032/image-deduplicate-cluster-webui/issues/1
# gradio似乎无法正常识别函数注解，暂时注释掉
def delet(duplicates_images_gallery,  # Tuple[str, str]
          delet_images_str,  # str
):
    
    """
    output=[duplicates_images_gallery, delet_images_str]
    第一个输出为画廊组件，第二个输出将delet_images_str文本清空
    """
    
    global confirmed_images_dir, cluster_list
    
    #如果还没查找过重复图片就什么都不做
    if (confirmed_images_dir is None or not confirmed_images_dir) or (cluster_list is None or not cluster_list):
        return [], ""
    
    # 尝试将字符串载入成字典
    try:
        delet_images_dict = toml.loads(delet_images_str)
    except Exception as e:
        toml_error_str = f"{delet_images_str}\n待删除列表toml格式错误，请修正\nerror: {e}"
        return duplicates_images_gallery, toml_error_str
    
    
    #获取待删除图片名字
    need_delet_images_name_list = []
    for parent_index, son_index_list in delet_images_dict.items():
        need_delet_images_name_list.extend( [ cluster_list[int(parent_index)][i] for i in son_index_list ] )
    
    print("删除图片中deleting...")
    for f in tqdm(need_delet_images_name_list):
        try:
            os.remove( os.path.join(confirmed_images_dir,f) )
        except Exception as e:
            logging.error( f"删除 {f} 时遭遇错误： {e}" )
    print("删除完成 Done!")
    
    #重置状态，阻止使用自动选择，除非再扫描一次
    cluster_list = None
    return [], ""

def auto_select() -> str:
    """
    由自带的启发式算法找出应该删去的图片
    https://idealo.github.io/imagededup/user_guide/finding_duplicates/
    """
    
    """
    弃用算法
    
    注意，这里我们又用了一次find_duplicates
    虽然该算法可以启发式找出去重所需最少的图片，但这导致多了一次计算
    或许我们可以用pandas自己写一个启发式算法
    每个图片对应一个索引；建立两个列，一个用于表示重复类标签，一个用于进行重复数投票
    删除被投票大于2的图片即可

    global encodings, cluster_list
    
    if (encodings is None or not encodings) or (cluster_list is None or not cluster_list):
        return "请先扫描"
    
    # 载入模型
    phasher = PHash()
    # 启发式算法获得待删除的列表，里面包含了图片名字字符串
    auto_duplicates_list = phasher.find_duplicates_to_remove(encoding_map=encodings)
    """
    
    global duplicates, cluster_list
    
    if (duplicates is None or not duplicates) or (cluster_list is None or not cluster_list):
        return "请先扫描"
    
    
    def get_files_to_remove(duplicates: Dict[str, List]) -> List:
        """
        来自: https://github.com/idealo/imagededup/blob/4e0b15f4cd82bcfa321eb280b843e57ebc5ff154/imagededup/utils/general_utils.py#L13
        Get a list of files to remove.
    
        Args:
            duplicates: A dictionary with file name as key and a list of duplicate file names as value.
    
        Returns:
            A list of files that should be removed.
        """
        # iterate over dict_ret keys, get value for the key and delete the dict keys that are in the value list
        files_to_remove = set()
    
        for k, v in duplicates.items():
            tmp = [
                i[0] if isinstance(i, tuple) else i for i in v
            ]  # handle tuples (image_id, score)
    
            if k not in files_to_remove:
                files_to_remove.update(tmp)
    
        return list(files_to_remove)
    
    auto_duplicates_list = get_files_to_remove(duplicates)
    
    
    # 定义一个空字典来存储映射关系
    mapping = {}
    # 遍历cluster_list中的每个集合
    for i, s in enumerate(cluster_list):
        # 遍历集合中的每个元素
        for j, x in enumerate(s):
            # 把元素作为键，把它所在的集合和位置作为值，存入字典
            mapping[x] = (i, j)

    # 定义输出的字典
    result = { f"{i}":[] for i in range( len( cluster_list ) ) }
    # 遍历lst1中的每个元素
    for x in auto_duplicates_list:
        # 获取它所在的集合和位置
        i, j = mapping[x]
        # 把第二个索引值加入到列表中
        bisect.insort(result[f"{i}"], j)

    return toml.dumps(result)

def all_select() -> str:
    """ 根据cluster_list的重复图片分类及数量选择全部图片 """
    
    global cluster_list
    
    if cluster_list is None or not cluster_list:
        return "请先扫描"
    
    # 生成类似
    # {"1":[0,1,2],
    #  "2":[0,1],
    #  ...
    #  }
    result = { f"{parent_index}":list( range(len(cluster)) )  for parent_index, cluster in enumerate( cluster_list ) }
    
    return toml.dumps(result)


def get_choose_image_index(evt: gr.SelectData, delet_images_str: str):
    # evt.value 为标签 ；evt.index 为图片序号； evt.target 为调用组件名
    global choose_image_index, images_info_list 
    
    # 获取图片属性信息
    choose_image_index = evt.value
    if  not isinstance(evt.index, int):
        raise TypeError(f"get_choose_image_index函数中evt.index 应为整数, 但是现在是{type(evt.index)}")
    image_info_json = images_info_list[evt.index]
    
    # 尝试判断所浏览的图片是否已经在待删除列表中
    # 如果是则为选择按钮标红，如果不是则取消按钮标红
    flag = (0,0)
    parent_index, son_index = evt.value.split(":")
    try:
        delet_images_dict = toml.loads(delet_images_str)

        if int(son_index) in delet_images_dict.get( str(parent_index), []):
            flag = (0,1)
        else:
            flag = (1,0)
    except Exception as e:
        logging.error(f"待删除列表无法以toml格式读取  error: {e}")
   
    def variant(flag: int):
        if flag==1:
            return 'primary'
        else:
            return "secondary" 

    gr_confirm_button = gr.update( value=f"选择 {evt.value}", variant=variant(flag[0]) )
    gr_cancel_button = gr.update( value=f"取消 {evt.value}", variant=variant(flag[1]) )
    
    return gr_confirm_button, gr_cancel_button, image_info_json


with gr.Blocks(css="#delet_button {color: red}") as demo:
    with gr.Row():
        with gr.Column(scale=10):
            images_dir = gr.Textbox(label="图片目录")
        with gr.Column(scale=1):
            use_cache = gr.Checkbox(label="使用缓存")
            find_duplicates_images_button = gr.Button("扫描重复图片")
    with gr.Row():
        with gr.Column(scale=10):
            with gr.Row():
                duplicates_images_gallery = gr.Gallery(label="重复图片", value=[]).style(columns=6, height="auto", preview=False)
            with gr.Row():
                confirm_button = gr.Button("选择图片")
                cancel_button = gr.Button("取消图片")
            with gr.Row():
                image_info_json = gr.JSON()
        with gr.Column(scale=1):
            delet_button = gr.Button("删除（不可逆）", elem_id="delet_button")
            auto_select_button = gr.Button("自动选择")
            all_select_button = gr.Button("全部选择")
            delet_images_str = gr.Textbox(label="待删除列表（手动编辑时请保证toml格式的正确）")

    # 按下后，在指定的目录搜索重复图像，并返回带标签的重复图像路径
    find_duplicates_images_button.click(fn=find_duplicates_images,
                                        inputs=[images_dir, use_cache],
                                        outputs=[duplicates_images_gallery, delet_images_str]
                                        )

    # 点击一个图片后，记录该图片标签于全局变量choose_image_index，并且把按钮更名为该标签;同时显示该图片分辨率等信息
    duplicates_images_gallery.select(fn=get_choose_image_index,
                                     inputs=[delet_images_str],
                                     outputs=[confirm_button, cancel_button, image_info_json]
                                     )

    # 按下后，将全局变量choose_image_index中的值加到列表中,同时将取消按钮变红
    confirm_button.click(fn=confirm,
                         inputs=[delet_images_str],
                         outputs=[delet_images_str, confirm_button, cancel_button]
                         )

    # 按下后，将全局变量choose_image_index中的值从列表中删除,同时将选择按钮变红
    cancel_button.click(fn=cancel,
                        inputs=[delet_images_str],
                        outputs=[delet_images_str, confirm_button, cancel_button]
                        )
    
    # 按下后，删除指定的图像，并更新画廊
    delet_button.click(fn=delet,
                       inputs=[duplicates_images_gallery, delet_images_str],
                       outputs=[duplicates_images_gallery, delet_images_str]
                       )
    
    # 按下后，用启发式算法自动找出建议删除的重复项
    auto_select_button.click(fn=auto_select,
                             inputs=[],
                             outputs=[delet_images_str]
                            )
    # 按下后，选择全部图片
    all_select_button.click(fn=all_select,
                            inputs=[],
                            outputs=[delet_images_str]
                            )
    
if __name__ == "__main__":
    demo.launch(debug=True, inbrowser=True)
