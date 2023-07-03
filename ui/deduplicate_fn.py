import gradio as gr
from imagededup.methods import PHash
import toml
import bisect
import os
from PIL import Image
import logging
from typing import Callable, Dict, List, Tuple

from ui.tools.operate_images import (
    cache_images_file,
    operate_images_file,
)


##############################  常量  ##############################

CACHE_RESOLUTION = 192  # 缓存图片时最大分辨率

tag_file_ext = ".txt"  # 存放特征tag的文件后缀名
wd4_file_ext = ".wd14.npz"  # 存放特征向量的文件后缀名
extra_file_ext_list = [tag_file_ext, wd4_file_ext]  # 在确认聚类中随图片一起被操作的文件后缀名列表

cache_folder_name = "cache"  # 缓存文件夹名


############################## 全局变量 ##############################

choose_image_index: str = "0:0"  # 用于记录当前点击了画廊哪个图片
cluster_list: list[ list[str] ] = []  # 用于记录重复图片的聚类结果
confirmed_images_dir: str = ""  # 用于记录查重结果对应的文件夹路径，防止使用者更改导致错误
images_info_dict: Dict[str, dict] = {}  # 用于记录重复图片的属性信息
duplicates: Dict[str, List] = {}  # 记录最原始的查重结果，将用于自动选择的启发式算法


##############################  运行查重  ##############################

# TODO: 添加更多查重方式和查重参数
def cluster_duplicates(images_dir: str) -> List[List[str]]:
    """ 返回该目录下重复图像聚类列表，元素为列表，每个子列表内为重复图像名字（不包含路径） """
    
    # 载入模型
    phasher = PHash()
    
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


def cluster_to_gallery(images_dir: str, cluster_list: List[List[str]]) -> List[Tuple[str, str]]:
    """
    将图片的绝对路径和索引合成一个元组，最后把全部元组放入一个列表，向gradio.Gallery传递
    列表中的元素顺序： 先排完第一个重复类，在排第二个重复类，以此类推
    """

    # 合成元组列表
    images_tuple_list = []
    for parent_index, parent in enumerate(cluster_list):
        for son_index, son in enumerate(parent):
            images_tuple_list.append( (os.path.join(images_dir, son), f"{parent_index}:{son_index}") )
            son_index += 1
    
    return images_tuple_list


def get_images_info(images_tuple_list: List[Tuple[str, str]]) -> Dict[str, dict]:
    """读取图片的信息，返回一个字典，键为图片的标签字符串，值为图片的信息字典

    Args:
        images_tuple_list (List[Tuple[str, str]]): 列表元素为二元组，元组第一个元素为图片的绝对路径，第二个元素为图片的标签字符串

    Returns:
        Dict[str, dict]: 键为图片的标签字符串，值为图片的信息字典
    """
    images_info_dict = {}
    for image_path, image_label in images_tuple_list:
        try:
            with Image.open(image_path) as im:
                size_MB = round( os.path.getsize(image_path)/1024/1024, 2 )
                image_info_dict = {
                    "resolution(l,w)":im.size,
                    "size":f"{size_MB} MB",
                    "format":im.format,
                    "filename":im.filename  # type: ignore
                }
                images_info_dict[image_label] = image_info_dict
        except Exception as e:
            logging.exception(f"{image_path}\n获取图片信息时发生了错误: {e}")
            images_info_dict[image_label] = {}

    return images_info_dict


def find_duplicates_images_error_wrapper(func: Callable) -> Callable:
    """ 用于处理find_duplicates_images的异常 """
    def wrapper(*args):
        try:
            return func(*args)
        except Exception as e:
            logging.exception(f"{func.__name__}函数出现了异常: {e}")
            return None, f"发生了错误: {e}"
    return wrapper

@find_duplicates_images_error_wrapper
def find_duplicates_images(images_dir: str,
                           use_cache: bool,
    ):
    """
    outputs=[duplicates_images_gallery, delet_images_str]
    """
    
    global confirmed_images_dir, cluster_list, images_info_dict
    
    # 全局变量
    confirmed_images_dir = images_dir
    
    # 全局变量
    cluster_list = cluster_duplicates(images_dir)  # 获取查重的聚类结果
    
    # 缓存缩略图
    cache_dir = os.path.join(images_dir, cache_folder_name)
    if use_cache:
        cache_images_list = []
        for cluster in cluster_list:
            for name in cluster:
                image_path = os.path.join(images_dir, name)
                cache_images_list.append(image_path)
        if cache_images_list:
            cache_images_file(cache_images_list, cache_dir, resolution=CACHE_RESOLUTION)
    
    # 如果用缓存就展示缓存图
    gallery_images_dir = images_dir if not use_cache else cache_dir
    # 转成gradio.Gallery需要的格式
    images_tuple_list = cluster_to_gallery(gallery_images_dir, cluster_list)
    
    # 全局变量
    images_info_dict = get_images_info(images_tuple_list)  # 获取图片的信息
    
    if not images_tuple_list:
        no_duplicates_str = "没有重复图像！！！"
        print(no_duplicates_str)
        return images_tuple_list, no_duplicates_str
    else:
        print(f"共有{len(images_tuple_list)}张重复图像")
        return (
            images_tuple_list,
            gr.update( value="", lines=len(cluster_list)+1 ),  # 让gr.Textbox的行数比聚类数多1，方便用户编辑
        )
    

##############################  确定选择某个图片  ##############################

# TODO: 用包装器处理出错的情况
def confirm(delet_images_str: str) -> Tuple[str, dict, dict]:
    
    # 尝试将字符串载入成字典
    try:
        delet_images_dict = toml.loads(delet_images_str)
    except Exception as e:
        toml_error_str = f"{delet_images_str}\n待删除列表toml格式错误，请修正\nerror: {e}"
        # 确认和取消按钮都变红
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
                delet_images_dict[parent_index_str],
                int(son_index_str)
            )

    # 按键名排序
    delet_images_dict = dict(
        sorted(delet_images_dict.items(), key=lambda x: int(x[0])))
    
    # 确定按钮变灰，取消按钮变亮
    return toml.dumps(delet_images_dict), gr.update( variant="secondary" ), gr.update( variant='primary' )


##############################  取消选择某个图片  ##############################

# TODO: 用包装器处理出错的情况
def cancel(delet_images_str: str) -> Tuple[str, dict, dict]:
    
    # 尝试将字符串载入成字典
    try:
        delet_images_dict = toml.loads(delet_images_str)
    except Exception as e:
        toml_error_str = f"{delet_images_str}\n待删除列表toml格式错误，请修正\nerror: {e}"
        # 确认和取消按钮都变红
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


##############################  确认删除图片  ##############################

# TODO: 用包装器处理出错的情况
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
    if (not confirmed_images_dir) or (not cluster_list):
        print("还没查找过重复图片，无法删除")
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
        # 某一个重复类的图片名字列表
        sub_cluster_list = cluster_list[ int(parent_index) ]
        # 该重复类中需要删除的图片名字
        need_delet_images_name_list.append( [ sub_cluster_list[i] for i in son_index_list ] )
    
    operate_images_file(
        images_dir=confirmed_images_dir,
        clustered_images_list=need_delet_images_name_list,
        extra_file_ext_list=extra_file_ext_list,
        operation="remove",
    )
    
    #重置状态，阻止使用自动选择，除非再扫描一次
    cluster_list = []
    confirmed_images_dir = ""
    return [], ""


##############################  自动选择  ##############################

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


def auto_select() -> str:
    """
    由自带的启发式算法找出应该删去的图片
    https://idealo.github.io/imagededup/user_guide/finding_duplicates/
    """
    
    global duplicates, cluster_list
    
    if (not duplicates) or (not cluster_list):
        return "请先扫描"

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

    # 生成类似
    # {"1":[0,1,2],
    #  "2":[0,1],
    #  ...
    #  }
    return toml.dumps(result)


##############################  全部选择  ##############################

def all_select() -> str:
    """ 根据cluster_list的重复图片分类及数量选择全部图片 """
    
    global cluster_list
    
    if not cluster_list:
        return "请先扫描"
    
    # 生成类似
    # {"1":[0,1,2],
    #  "2":[0,1],
    #  ...
    #  }
    result = { f"{parent_index}":list( range(len(cluster)) )  for parent_index, cluster in enumerate( cluster_list ) }
    
    return toml.dumps(result)


##############################  根据当前浏览的图片，更改按钮颜色和显示图片信息  ##############################

def get_choose_image_index(evt: gr.SelectData, delet_images_str: str):
    # evt.value 为标签 ；evt.index 为图片序号； evt.target 为调用组件名
    global choose_image_index, images_info_dict
    
    # 获取图片属性信息
    choose_image_index = evt.value  # 画廊中图片标签，例如 "0:1"
    try:
        image_info_json = images_info_dict[choose_image_index]
    except Exception as e:
        logging.error(f"获取{choose_image_index}图片属性信息失败\nerror: {e}")
        image_info_json = {}
    
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