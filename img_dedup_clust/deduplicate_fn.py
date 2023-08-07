import logging
from typing import Callable, Dict, List, Tuple, Optional
import bisect
import os
import gc
import time

import torch  # 如无明确理由，请把torch放前面调用它的cuda环境，避免某些意外
import gradio as gr
from imagededup.methods import (
    PHash,
    AHash,
    WHash,
    DHash,
    CNN,
)
import toml
from PIL import Image

from img_dedup_clust.tools.operate_images import (
    cache_images_file,
    operate_images_file,
    CLUSTER_DIR_PREFIX,
)


##############################  常量  ##############################

# 注意常量列表的字符串元素会被assert检查，修改时请修改相应的assert

CACHE_RESOLUTION = 192  # 缓存图片时最大分辨率

CACHE_FOLDER_NAME = "cache"  # 缓存文件夹名

PROCESS_CLUSTERS_METHOD_CHOICES = [
    "更名选中图片（推荐全部选择）",
    "移动选中图片（推荐此方式）",
    "删除选中图片（推荐自动选择）",
]

IMAGEDEDUP_MODE_CHOICES_LIST = [
    "Hash - recommend on CPU",
    "CNN - recommend on GPU",
]

HASH_METHODS_CHOICES_LIST = [
    "PHash",
    "AHash",
    "DHash",
    "WHash",
]

CNN_METHODS_CHOICES_LIST = [
    "CNN",
]

# 请使用js标准key，在deduplicate_ui中被使用
CONFIRM_KEYBORAD_KEY = "Enter"  # 确定选择按钮的键盘按键
CANCEL_KEYBORAD_KEY = "Backspace"  # 取消选择按钮的键盘按键


############################## 全局变量 ##############################

# TODO: 换成gr.State，这样可以在界面刷新后失效，和避免多用户间干扰
choose_image_index: str = "0:0"  # 用于记录当前点击了画廊哪个图片
cluster_list: list[ list[str] ] = []  # 用于记录重复图片的聚类结果
confirmed_images_dir: str = ""  # 用于记录查重结果对应的文件夹路径，防止使用者更改导致错误
images_info_dict: Dict[str, dict] = {}  # 用于记录重复图片的属性信息
duplicates: Dict[str, List] = {}  # 记录最原始的查重结果，将用于自动选择的启发式算法


##############################  查重模式选择  ##############################

def imagededup_mode_choose_trigger(imagededup_mode_index: int):
    gr_Box_update_list = [ gr.update(visible=False) for i in IMAGEDEDUP_MODE_CHOICES_LIST ]
    gr_Box_update_list[imagededup_mode_index] = gr.update(visible=True)
    return gr_Box_update_list


##############################  释放Torch内存  ##############################

# TODO: 将deduplicator以全局变量形式存在，就像wd14聚类那样
# 这样可以使用deduplicator=None更加深度的释放内存，但是这样需要多加一层模型是否改变判断
# 不过目前torch.cuda.empty_cache()已经释放的较彻底了
# 至于CPU内存的占用，似乎deduplicator是以函数方式运行，结束后自己就释放了
def release_torch_memory():
    # 最好先回收垃圾再释放显存
    gc.collect()
    time.sleep(0.1)
    torch.cuda.empty_cache()
    print("释放完毕")


##############################  运行查重  ##############################

# TODO: 如果有继承需要
# 可以使用from img_dedup_clust.tools.partialmethod_tools import make_cls_partialmethod
# 将方法修改为偏方法
class MyHasher:
    def __init__(
        self,
        hash_methods_choose: int,
        max_distance_threshold: int,
    ):
        # 输出信息的同时可以判断输入值是否非法
        hash_methods_str = HASH_METHODS_CHOICES_LIST[hash_methods_choose]
        logging.info(f"选择了：{hash_methods_str}")

        if hash_methods_choose == 0:
            assert hash_methods_str == "PHash"
            self.hasher = PHash()
        elif hash_methods_choose == 1:
            assert hash_methods_str == "AHash"
            self.hasher = AHash()
        elif hash_methods_choose == 2:
            assert hash_methods_str == "DHash"
            self.hasher = DHash()
        elif hash_methods_choose == 3:
            assert hash_methods_str == "WHash"
            self.hasher = WHash()
        else:
            raise ValueError(
                (
                    f"hash_methods_str = {hash_methods_str}非法, "
                    f"目前只支持{[i for i in range(len(HASH_METHODS_CHOICES_LIST))]}"
                )
            )

        self.max_distance_threshold = max_distance_threshold

    def find_duplicates(self, **kwargs):
        # 使用实例化时候的阈值
        kwargs["max_distance_threshold"] = self.max_distance_threshold
        return self.hasher.find_duplicates(**kwargs)


class MyCNN:
    def __init__(
        self,
        cnn_methods_choose: int,
        min_similarity_threshold: float,
    ):
        # 输出信息的同时可以判断输入值是否非法
        cnn_methods_str = CNN_METHODS_CHOICES_LIST[cnn_methods_choose]
        logging.info(f"选择了：{cnn_methods_str}")

        if cnn_methods_choose == 0:
            assert cnn_methods_str == "CNN"
            self.cnn = CNN()
        else:
            raise ValueError(
                (
                    f"cnn_methods_str = {cnn_methods_str}非法, "
                    f"目前只支持{[i for i in range(len(CNN_METHODS_CHOICES_LIST))]}"
                )
            )

        self.min_similarity_threshold = min_similarity_threshold

    def find_duplicates(self, **kwargs):
        # 使用实例化时候的阈值
        kwargs["min_similarity_threshold"] = self.min_similarity_threshold
        return self.cnn.find_duplicates(**kwargs)


def cluster_duplicates(
    images_dir: str,
    imagededup_mode_choose_index: int,
    hash_methods_choose: int,
    max_distance_threshold: int,
    cnn_methods_choose: int,
    min_similarity_threshold: float,
) -> List[List[str]]:
    """ 返回该目录下重复图像聚类列表，元素为列表，每个子列表内为重复图像名字（不包含路径） """
    
    # 载入模型

    # 输出信息的同时可以判断输入值是否非法
    imagededup_mode_str = IMAGEDEDUP_MODE_CHOICES_LIST[imagededup_mode_choose_index]
    logging.info(f"选择了：{imagededup_mode_str}")

    if imagededup_mode_choose_index == 0:
        assert imagededup_mode_str == "Hash - recommend on CPU"
        deduplicator = MyHasher(
            hash_methods_choose = hash_methods_choose,
            max_distance_threshold = max_distance_threshold,
        )
    elif imagededup_mode_choose_index == 1:
        assert imagededup_mode_str == "CNN - recommend on GPU"
        deduplicator = MyCNN(
            cnn_methods_choose = cnn_methods_choose,
            min_similarity_threshold = min_similarity_threshold,
        )
    else:
        raise ValueError(
            (
                f"imagededup_mode_str = {imagededup_mode_str}非法, "
                f"目前只支持{[i for i in range(len(IMAGEDEDUP_MODE_CHOICES_LIST))]}"
            )
        )
    
    global duplicates
    # 查找重复
    duplicates = deduplicator.find_duplicates(image_dir=images_dir) # type: ignore
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
            logging.error(f"{image_path}\n获取图片信息时发生了错误: {e}")
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
def find_duplicates_images(
    images_dir: str,
    use_cache: bool,
    imagededup_mode_choose_index: int,
    hash_methods_choose: int,
    max_distance_threshold: int,
    cnn_methods_choose: int,
    min_similarity_threshold: float,
):
    """
    outputs=[duplicates_images_gallery, delet_images_str]
    """
    
    global confirmed_images_dir, cluster_list, images_info_dict
    
    # 全局变量
    confirmed_images_dir = images_dir
    
    # 全局变量
    cluster_list = cluster_duplicates(
        images_dir,
        imagededup_mode_choose_index,
        hash_methods_choose,
        max_distance_threshold,
        cnn_methods_choose,
        min_similarity_threshold,
    )  # 获取查重的聚类结果
    
    # 缓存缩略图
    exist_cache_error = False
    cache_dir = os.path.join(images_dir, CACHE_FOLDER_NAME)
    if use_cache:
        cache_images_list = []
        for cluster in cluster_list:
            for name in cluster:
                image_path = os.path.join(images_dir, name)
                cache_images_list.append(image_path)
        if cache_images_list:
            exist_cache_error = cache_images_file(cache_images_list, cache_dir, resolution=CACHE_RESOLUTION)
    
    # TODO: 缓存失败的个别图片用原图
    # 如果用缓存就展示缓存图
    gallery_images_dir = images_dir if (not use_cache or exist_cache_error) else cache_dir
    if exist_cache_error:
        logging.warning("某个图片出现缓存失败，缓存功能失效，将统一使用原图")

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
            gr.update( value="", lines=len(cluster_list) ),  # 让gr.Textbox的行数比聚类数多1，方便用户编辑
        )
    

##############################  确定选择某个图片  ##############################

def confirm_exception_wrapper(func) -> Callable:
    """
    用于处理confirm函数的异常
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"{func.__name__}函数出现了异常: {e}")
            return (
                f"警告，发生了未知错误: {e}\n详细请查看控制台，解决后请重新运行查重",
                gr.update( variant="secondary" ),  # 确认按钮变灰
                gr.update( variant="secondary" ),  # 取消按钮变灰
            )
    return wrapper

@confirm_exception_wrapper
def confirm(delet_images_str: str) -> Tuple[str, dict, dict]:
    
    # 尝试将字符串载入成字典
    try:
        delet_images_dict = toml.loads(delet_images_str)
    except Exception as e:
        toml_error_str = f"{delet_images_str}\n待删除列表toml格式错误，请修正\nerror: {e}"
        # 确认和取消按钮都变灰
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

def cancel_exception_wrapper(func) -> Callable:
    """
    用于处理cancel函数的异常
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"{func.__name__}函数出现了异常: {e}")
            return (
                f"警告，发生了未知错误: {e}\n详细请查看控制台，解决后请重新运行查重",
                gr.update( variant="secondary" ),  # 确认按钮变灰
                gr.update( variant="secondary" ),  # 取消按钮变灰
            )
    return wrapper

@cancel_exception_wrapper
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


##############################  确认处理图片  ##############################

def split_str_by_comma(str_with_comma: Optional[str] =  None) -> List[str]:
    """ 输入的undesired_tags使用逗号分割 """
    undesired_tags_list = []
    if isinstance(str_with_comma, str) and str_with_comma:
        undesired_tags_list = str_with_comma.split(",")
        undesired_tags_list = [tag.strip() for tag in undesired_tags_list]
        undesired_tags_list = list( set(undesired_tags_list) )
    return undesired_tags_list


def confirm_cluster_exception_wrapper(func) -> Callable:
    """
    用于处理confirm_cluster函数的异常
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"{func.__name__}函数出现了异常: {e}")
            return (
                [],  # 清空画廊
                f"警告，发生了未知错误: {e}\n详细请查看控制台，解决后请重新运行查重",
            )
    return wrapper

# https://github.com/WSH032/image-deduplicate-cluster-webui/issues/1
# gradio似乎无法正常识别函数注解，暂时注释掉
@confirm_cluster_exception_wrapper
def confirm_cluster(
    selected_images_str,  # type: str
    process_clusters_method,  # type: int
    need_operated_extra_file_extension_Textbox,  # type: str
    need_operated_extra_file_name_Textbox,  # type: str
) -> Tuple[
        dict,  # 画廊组件，gr.update的返回值
        str,
    ]:
    
    """
    outputs=[duplicates_images_gallery, selected_images_str],
    第一个输出为画廊组件，第二个输出将selected_images_str文本清空
    """
    
    global confirmed_images_dir, cluster_list
    
    #如果还没查找过重复图片就什么都不做
    if (not confirmed_images_dir) or (not cluster_list):
        no_duplicates_str = "还没查找过重复图片，无法操作"
        print(no_duplicates_str)
        return gr.update(value=[]), no_duplicates_str
    
    # 尝试将字符串载入成字典
    try:
        selected_images_dict = toml.loads(selected_images_str)
    except Exception as e:
        toml_error_str = f"{selected_images_str}\n待操作列表toml格式错误，请修正\nerror: {e}"
        return gr.update(), toml_error_str
    
    #获取待操作图片名字
    clustered_images_list = []
    for parent_index, son_index_list in selected_images_dict.items():
        # 某一个重复类的图片名字列表
        sub_cluster_list = cluster_list[ int(parent_index) ]
        # 该重复类中需要操作的图片名字
        clustered_images_list.append( [ sub_cluster_list[i] for i in son_index_list ] )

    # 输出信息
    def display_info(clustered_images_list: List[List[str]]):
        if clustered_images_list:
            for index, sub_cluster_list in enumerate(clustered_images_list):
                print(f"第{index}类 待操作图片列表: {sub_cluster_list}")
        else:
            logging.warning("待操作图片列表为空，你似乎没有选择任何图片")
    display_info(clustered_images_list)

    # 输出信息的同时可以判断输入值是否非法
    process_clusters_method_choose_str = PROCESS_CLUSTERS_METHOD_CHOICES[process_clusters_method]
    logging.info(f"选择了：{process_clusters_method_choose_str}")

    # 带时间戳的重命名原图片和附带文件
    if process_clusters_method == 0:
        assert process_clusters_method_choose_str == "更名选中图片（推荐全部选择）"
        operation = "rename"
        operate_result_str = f"{confirmed_images_dir}\n内被选中的图片均加上了{CLUSTER_DIR_PREFIX}前缀"
    # 移动原图至Cluster文件夹
    elif process_clusters_method == 1:
        assert process_clusters_method_choose_str == "移动选中图片（推荐此方式）"
        operation = "move"
        operate_result_str = f"{confirmed_images_dir}\n内被选中的的图片均被移动至该目录的{CLUSTER_DIR_PREFIX}子文件夹"
    # 删除原图
    elif process_clusters_method == 2:
        assert process_clusters_method_choose_str == "删除选中图片（推荐自动选择）"
        operation = "remove"
        operate_result_str = f"{confirmed_images_dir}\n内被选中的的图片均被删除"
    else:
        raise ValueError(
            (
                f"process_clusters_method = {process_clusters_method}非法, "
                f"目前只支持{[i for i in range(len(PROCESS_CLUSTERS_METHOD_CHOICES))]}"
            )
        )

    operate_result_str = f"成功{operation}了被选中的{len(clustered_images_list)}个重复类\n" + operate_result_str

    # 需要一起处理的文件的扩展名字符串，如".txt, .wd14.npz"
    extra_file_ext_list = split_str_by_comma(need_operated_extra_file_extension_Textbox)
    # 会被复制到每一个子文件夹的文件的全名字符串，如"abc.txt"
    copy_to_subfolder_file_list = split_str_by_comma(need_operated_extra_file_name_Textbox)

    operate_images_file(
        images_dir=confirmed_images_dir,
        clustered_images_list=clustered_images_list,
        extra_file_ext_list=extra_file_ext_list,
        copy_to_subfolder_file_list=copy_to_subfolder_file_list,
        operation=operation,
    )


    #重置状态，阻止使用自动选择，除非再扫描一次
    cluster_list = []
    confirmed_images_dir = ""
    return gr.update(value=[]), operate_result_str


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


def auto_select_exception_wrapper(func) -> Callable:
    """
    用于处理auto_selectr函数的异常
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"{func.__name__}函数出现了异常: {e}")
            return f"警告，发生了未知错误: {e}\n详细请查看控制台，解决后请重新运行查重"
    return wrapper

@auto_select_exception_wrapper
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

def all_select_exception_wrapper(func) -> Callable:
    """
    用于处理all_select函数的异常
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"{func.__name__}函数出现了异常: {e}")
            return f"警告，发生了未知错误: {e}\n详细请查看控制台，解决后请重新运行查重"
    return wrapper

@all_select_exception_wrapper
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

def get_choose_image_index_exception_wrapper(func) -> Callable:
    """
    用于处理get_choose_image_index函数的异常
    """
    # 注意，这里的第一个参数evt: gr.SelectData不要动
    def wrapper(
            evt: gr.SelectData,
            *args,
            **kwargs,
        ):
        try:
            return func(evt, *args, **kwargs)
        except Exception as e:
            logging.exception(f"{func.__name__}函数出现了异常: {e}")
            return (
                gr.update( value="出错", variant="secondary" ),
                gr.update( value="出错", variant="secondary" ),
                {"Warning":f"警告，发生了未知错误: {e}\n详细请查看控制台，解决后请重新运行查重"},
            )
    return wrapper

@get_choose_image_index_exception_wrapper
def get_choose_image_index(
    evt: gr.SelectData,
    delet_images_str: str
) -> Tuple[dict, dict, dict]:

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

    gr_confirm_button = gr.update( value=f"选择 {evt.value} [{CONFIRM_KEYBORAD_KEY}]", variant=variant(flag[0]) )
    gr_cancel_button = gr.update( value=f"取消 {evt.value} [{CANCEL_KEYBORAD_KEY}]", variant=variant(flag[1]) )
    
    return gr_confirm_button, gr_cancel_button, image_info_json
