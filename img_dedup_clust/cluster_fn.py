import os
from typing import List, Tuple, Callable, Union, Literal
import logging
import math
import textwrap


import sklearn.cluster as skc
import sklearn.feature_extraction.text as skt
from sklearn.metrics import silhouette_score, davies_bouldin_score
# 评分器： chi2, mutual_info_regression, f_classif
# 选择器： SelectPercentile, SelectKBest
from sklearn.feature_selection import (
    chi2,
    # mutual_info_regression,
    # mutual_info_classif,
    # f_classif,
    # SelectPercentile,
    SelectKBest,
)
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np
import numpy.typing as npt
import gradio as gr
from tqdm import tqdm


from img_dedup_clust.tools.operate_images import (
    cache_images_file,
    operate_images_file,
    CLUSTER_DIR_PREFIX,
)
from img_dedup_clust.tools.SearchImagesTags import SearchImagesTags
from tag_images_by_wd14_tagger import (
    DEFAULT_TAGGER_CAPTION_EXTENSION,  # 默认打标文件的扩展名
    WD14_NPZ_EXTENSION,  # 用于保存推理所得特征向量的文件扩展名 # .wd14用来区分kohya的潜变量cache
    WD14_TAGS_TOML_FILE,  # 存储各列向量对应的tag的文件的名字
    read_wd14_tags_toml,  # 读取存储各列向量对应的tag的文件的内容以获取tags列表
    WD14_NPZ_ARRAY_PREFIX,  # 所保存的npz文件中，特征向量的前缀名字，后面会加上层数，如layer0, layer1, layer2, layer3
    WD14_TAGS_CATEGORY_LIST,  # noqa 401 # wd14标签的种类 # 别删，cluster_ui.py中有用到
)


##############################  常量  ##############################

# 注意常量列表的字符串元素会被assert检查，修改时请修改相应的assert

MAX_GALLERY_NUMBER = 100  # 画廊里展示的最大聚类数量为100
CACHE_RESOLUTION = 192  # 缓存图片时最大分辨率

CACHE_FOLDER_NAME = "cache"  # 缓存文件夹名

FEATURE_EXTRACTION_METHOD_LIST = [
    "Text tags文本特征向量",
    "Image wd14图片特征向量",
]

WD14_FEATURE_LAYER_CHOICE_LIST = [
    "predictions_sigmoid 全向量层",
    "predictions_norm 压缩层",
]

TEXT_VECTORIZATION_METHOD_LIST = [
    "TfidfVectorizer",
    "CountVectorizer",
]

PROCESS_CLUSTERS_METHOD_CHOICES = [
    "重命名原图片(不推荐)",
    f"在{CLUSTER_DIR_PREFIX}文件夹下生成聚类副本(推荐)",
    f"移动原图至{CLUSTER_DIR_PREFIX}文件夹(大数据集推荐)",
]

CLUSTER_MODEL_LIST = [
    "K-Means聚类",
    "Spectral谱聚类",
    "Agglomerative层次聚类",
    "OPTICS聚类"
]

KMEANS_N_INIT = 8  # KMeans聚类时的n_init参数

# 特征重要性分析时候用于划分prompt和negetive的阈值
CLUSTER_IMPORTANT_FEATURE_PROMPT_TAGS_THRESHOLD = 0.5  # 闭区间 # 某一类中的特征向量中，某个tag的比例大于这个值就认为是prompt
CLUSTER_IMPORTANT_FEATURE_NEGETIVE_TAGS_THRESHOLD = 0.2  # 开区间 # 某一类中的特征向量中，某个tag的比例小于这个值就认为是negetive

DEFAULT_SVD_N_COMPONENTS_PERCENTAGE = 0.85  # 预处理后，SVD降维数滑条所处的默认值与最大值比值


##############################  私有常量  ##############################

_X_OUTSIDE_LAYER_INDEX: Literal[0, 1] = 0  # 外层特征向量所在的层数索引，0对应sigmoid输出后，1对应sigmoid输出前
"""sigmoid的输出能把 [-2, 2] 非线性映射到 [0.2, 0.8]， 似乎更有助于聚类"""

def inv_sigmoid(y):
    """计算 sigmoid 函数的反函数值"""
    y = np.clip(y, 1e-15, 1 - 1e-15)  # 避免除以 0 和取对数负无穷大
    return np.log(y / (1 - y))

_WD14_OUTSIDE_BOOL_ARRAY_THRESHOLD = 0.35  # 外层特征向量二值化的阈值，大于此阈值视为某样本存在相应的tag特征
# 如果选择了1，则将阈值转为sigmoid的反函数值
if _X_OUTSIDE_LAYER_INDEX == 1:
    _WD14_OUTSIDE_BOOL_ARRAY_THRESHOLD = inv_sigmoid(_WD14_OUTSIDE_BOOL_ARRAY_THRESHOLD)


##############################  类型别名  ##############################

# 所支持的聚类模型，用于类型检查
ClusterModelAlias = Union[
    skc.KMeans,
    skc.SpectralClustering,
    skc.AgglomerativeClustering,
    skc.OPTICS
]

# 请保证这个类型为Tuple，避免某个组件隐式修改了它
vectorize_X_and_label_State_Alias = Tuple[
    # 如果是tag聚类，则两个矩阵相同
    # 如果是wd14聚类，则第一个矩阵为入口层，第二个矩阵为出口层
    # 第一个矩阵将用于聚类； 第二矩阵的列和用于特征重要性分析的tag长度对应，将用于特征重要性分析
    Tuple[np.ndarray, npt.NDArray[np.bool_]],
    # 向量每列对应的tag
    List[str],
    # 每行对应的图片名
    List[str]
]


##############################  全局变量  ##############################

# TODO: 换成gr.State，这样可以在界面刷新后失效，和避免多用户间干扰
tag_file_ext = DEFAULT_TAGGER_CAPTION_EXTENSION  # 存放特征tag的文件后缀名
wd4_file_ext = WD14_NPZ_EXTENSION  # 存放特征向量的文件后缀名


##############################  工具函数  ##############################

def aoto_set_sklearn_model_n_value(
    model: ClusterModelAlias,  # 聚类模型
    n_value: int,
) -> ClusterModelAlias:
    """为聚类模型自动设置相应的n值"""

    # 除OPTICS聚类外其他
    if "n_clusters" in model.get_params().keys():
        model.set_params(n_clusters=n_value)
    # 对应OPTICS聚类
    elif "min_samples" in model.get_params().keys():
        model.set_params(min_samples=n_value)
    else:
        raise ValueError(f"选择模型{model}，n参数指定出现问题")

    return model


def check_is_able_to_decomposition(min_n: int, decomposition_arr: np.ndarray):
    """检查是否能降维，如果不能则抛出异常

    Args:
        min_n (int): 最小能接受的降维数，需要降维矩阵的样本数和特征数要严格小于这个值
        decomposition_arr (np.ndarray): 需要降维的矩阵，行为样本数，列为特征数

    Raises:
        ValueError: 无法降维则抛出该错误
    """    
    # 如果样本数或特征数为1，则无法降维
    error_value_name = []
    if decomposition_arr.shape[0] <= min_n:
        error_value_name += ["样本数"]
    if decomposition_arr.shape[1]  <= min_n:
        error_value_name += ["特征数"]
    
    if error_value_name:
        raise ValueError(f"{', '.join(error_value_name)}小于{min_n}，无法降维")


# 二维可视化
def visualization_2D(visualization_arr: np.ndarray) -> np.ndarray:
    """
    用PCA算法，将输入的矩阵列数降至二

    Args:
        visualization_arr (np.ndarray): 需要降维可视化的矩阵
    """
    # 避免隐式修改了原矩阵
    visualization_arr = np.array(visualization_arr, copy=True)

    row_n, col_n  = visualization_arr.shape
    # 如果只有一个特征，则复制一列，假装为两个特征
    if col_n == 1:
        visualization_arr = np.hstack( (visualization_arr, visualization_arr) )
        col_n = 2
    
    # 如果只有一个样本，就取前两列当做降维
    if row_n == 1:
        return visualization_arr[:, 0:2]

    # 降维管道
    # TODO: 目前使用速度最快的pca降维，但是效果可能不是很好； 可以考虑使用t-SNE，但是后者速度较慢，要考虑更具不同情况优化参数
    # https://www.scikit-yb.org/en/latest/api/features/manifold.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE
    pipeline = make_pipeline(
        PCA(n_components=2),  # 设置成二维以在散点图上展示
    )
    return pipeline.fit_transform(visualization_arr)


##############################  聚类特征选择Box切换  ##############################

def feature_extraction_method_change_trigger(feature_extraction_method_index: int):
    gr_Box_update_list = [ gr.update(visible=False) for i in FEATURE_EXTRACTION_METHOD_LIST ]
    gr_Box_update_list[feature_extraction_method_index] = gr.update(visible=True)
    return gr_Box_update_list


##############################  聚类特征选择Box切换  ##############################

def wd14_feature_layer_choice_change_trigger(wd14_feature_layer_choice: int):
    """ outputs=[tags_category_choices_CheckboxGroup] """

    wd14_feature_layer_choose_str = WD14_FEATURE_LAYER_CHOICE_LIST[wd14_feature_layer_choice]
    if wd14_feature_layer_choice == 0:
        assert wd14_feature_layer_choose_str == "predictions_sigmoid 全向量层"
        return gr.update(visible=True)  # 全向量层运行使用部分特征
    elif wd14_feature_layer_choice == 1:
        assert wd14_feature_layer_choose_str == "predictions_norm 压缩层"
        return gr.update(visible=False)  # 前端隐藏这个功能，不过参数还是会被传递给后端
    else:
        raise ValueError(
            (
                f"wd14_feature_layer_choice = {wd14_feature_layer_choice}非法, "
                f"目前只支持{[i for i in range(len(WD14_FEATURE_LAYER_CHOICE_LIST))]}"
            )
        )


##############################  特征获取  ##############################
# 逗号分词器
def comma_tokenizer(text: str) -> List[str]:
    """
    定义一个以逗号为分隔符的分词器函数，并对每个标签进行去空格操作
    如输入"xixi, haha"
    返回["xixi", "haha"]
    """
    return [tag.strip() for tag in text.split(',')]

# tag聚类
def text_vectorizer_func(
    images_dir: str,
    text_vectorizer_method: Literal[0, 1],
    use_comma_tokenizer: bool,
    use_binary_tokenizer: bool,
) -> Tuple[Tuple[np.ndarray, npt.NDArray[np.bool_]], List[str], List[str]]:

    # 搜索存放tag的txt文本
    searcher = SearchImagesTags(images_dir, tag_file_ext=tag_file_ext)

    # 文本提取器参数
    vectorizer_args_dict = {
        "binary": use_binary_tokenizer,
        "max_df": 0.99,
    }
    if use_comma_tokenizer:
        vectorizer_args_dict["tokenizer"] = comma_tokenizer


    # 输出信息的同时可以判断输入值是否非法
    text_vectorizer_method_str = TEXT_VECTORIZATION_METHOD_LIST[text_vectorizer_method]
    logging.info(f"选择了：{text_vectorizer_method_str}")

    # 选择特征提取器
    if text_vectorizer_method == 0 :
        assert text_vectorizer_method_str == "TfidfVectorizer"
        tfvec = skt.TfidfVectorizer(**vectorizer_args_dict)
    elif text_vectorizer_method == 1 :
        assert text_vectorizer_method_str == "CountVectorizer"
        tfvec = skt.CountVectorizer(**vectorizer_args_dict)
    else:
        raise ValueError(
            (
                f"text_vectorizer_method = {text_vectorizer_method}非法, "
                f"目前只支持{[i for i in range(len(TEXT_VECTORIZATION_METHOD_LIST))]}"
            )
        )
    
    # tag内容, 用于文本提取
    tag_content_list = searcher.tag_content(error_then_tag_is="_no_tag")
    
    # tags转为向量特征
    X_inside = tfvec.fit_transform(tag_content_list).toarray()  # type: ignore # 向量特征 # 原来是稀疏矩阵，但是转为易于操作的np矩阵
    X_outside: npt.NDArray[np.bool_] = (X_inside > 0)  # 在文本聚类中，大于0代表相应的tag在文本中出现过

    tf_tags_list = tfvec.get_feature_names_out().tolist()  # 向量每列对应的tag
    # stop_tags = tfvec.stop_words_  # 被过滤的tag

    # 特征矩阵每行对应的文件名
    image_files_list = searcher.image_files_list
    assert image_files_list is not None, "image_files_list is None"  # 正常来说不会为None，make pylance happy

    return ( (X_inside, X_outside), tf_tags_list, image_files_list )


def _check_error(need_check_list: List[Union[np.ndarray, None]], error_indies: List[int]):
    """ 将错误读取的矩阵元素全部置一，会直接修改传入的need_check_list """

    # 没有错误就不检查了
    if not error_indies:
        return
    for temp in need_check_list:
        # 找到一个不为空的正确元素，记录其中矩阵的维度，生成一个同样大小的一矩阵
        # 赋值给需要检查列表中错误的元素
        if temp is not None:
            # repalce_arr = np.zeros_like(temp)
            repalce_arr = np.ones_like(temp)
            for index in error_indies:
                need_check_list[index] = repalce_arr
            break

def read_wd14_npz_files(
    dir: str,
    npz_files_list: List[str],
    layer_index: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    读取储存在.wd14.npz文件中的向量特征
    如果某个npz文件读取错误，其对应的行向量将其置一
    返回一个三元组
        第一个元素为内层特征向量矩阵，由layer_index指定内层向量所位于的层数
        第二个元素始终为最外层外层特征向量矩阵
        第三个元素为出错的行向量索引
    每个矩阵的行按照npz_files_list的顺序排列

    Args:
        dir (str): npz文件所在的文件夹
        npz_files_list (List[str]): npz文件名字列表
        layer_index (int): _description_

    Raises:
        ValueError: 全部npz都读取错误时候抛出

    Returns:
        Tuple[np.ndarray, np.ndarray, List[int]]: 第一个元素为内层特征向量矩阵，第二个元素为外层特征向量矩阵，第三个元素为出错的行向量索引
    """

    # 用来暂存向量
    X_inside = []  # 内层，取决于layer_index
    X_outside = []  # 出口层，固定取最外层
    error_indies = []  # 读取错误的文件索引
    for index, npz_file in enumerate(npz_files_list):
        try:
            with np.load( os.path.join(dir, npz_file) ) as npz:
                X_inside.append(npz[f"{WD14_NPZ_ARRAY_PREFIX}{layer_index}"])
                X_outside.append(npz[f"{WD14_NPZ_ARRAY_PREFIX}{_X_OUTSIDE_LAYER_INDEX}"])  # 0对应sigmoid输出后，1对应sigmoid输出前
        except Exception as e:
            logging.error(f"读取 {npz_file} 向量特征时出错，其将会被置一：\n{e}")
            # 读错了就跳过
            X_inside.append(None)
            X_outside.append(None)
            error_indies.append(index)  # 记录出错位置
            continue

    if len(error_indies) == len(npz_files_list):
        raise ValueError("所有向量特征读取错误，无法继续")
    
    # 注意，请先检查数据，因为如果读取出错了列表里面会有None元素，无法转为矩阵
    _check_error(X_inside, error_indies)
    _check_error(X_outside, error_indies)

    return ( np.array(X_inside), np.array(X_outside), error_indies )


def np_split_copy(*args, **kwargs):
    """在np.split的基础返回copy而不是视图"""
    splited_X_list = np.split(*args, **kwargs)
    return [splited_X.copy() for splited_X in splited_X_list]


def _bipolarize(arr, min=0, max=1):
    """二维矩阵，每一行最大的元素为max，其余为min"""
    max_indices = np.argmax(arr, axis=1)  # 找到每一行最大元素的索引    
    output_array = np.ones_like(arr, dtype=int) * min  # 将其余全部置为min
    output_array[np.arange(len(max_indices)), max_indices] = max  # 原本最大值对应的位置置为max
    return output_array


def _choose_and_concatenate(arr_list: List[np.ndarray], choose: List[int]):
    """横着把choose中指定的arr_list中的二维矩阵连起来"""
    arr_list = [arr_list[i] for i in choose]
    return np.concatenate(arr_list, axis=1)


# wd14特征向量聚类
def get_wd14_feature_func(
    images_dir: str,
    wd14_feature_layer_choice: Literal[0, 1],
    tags_category_choices: List[Literal[0, 1, 2]],
) -> Tuple[Tuple[np.ndarray, npt.NDArray[np.bool_]], List[str], List[str]]:
    
    # 输出信息的同时可以判断输入值是否非法
    wd14_feature_layer_choose_str = WD14_FEATURE_LAYER_CHOICE_LIST[wd14_feature_layer_choice]
    logging.info(f"选择了：{wd14_feature_layer_choose_str}")

    if wd14_feature_layer_choice == 0:
        assert wd14_feature_layer_choose_str == "predictions_sigmoid 全向量层"
        # TODO: 让它暂时和特征重要性分析所用的层一致，到时候再增加自定义的0，1，2，3层
        layer_index = _X_OUTSIDE_LAYER_INDEX
    elif wd14_feature_layer_choice == 1:
        assert wd14_feature_layer_choose_str == "predictions_norm 压缩层"
        layer_index = 2
    else:
        raise ValueError(
            (
                f"wd14_feature_layer_choice = {wd14_feature_layer_choice}非法, "
                f"目前只支持{[i for i in range(len(WD14_FEATURE_LAYER_CHOICE_LIST))]}"
            )
        )
    
    # 是否可以使用tags_category_choices启动部分特征聚类
    if layer_index in (0, 1):
        # 只有全向量层才可以，因为全向量层的列特征和tags是一一对应的
        able_to_use_tags_category_choices = True
    elif layer_index in (2, 3):
        # 压缩层的列数和tags不是一一对应的，不支持部分特征功能
        able_to_use_tags_category_choices = False
    else:
        assert False, "layer_index非法"

    # 读取储存在.wd14.npz文件中的向量特征
    searcher = SearchImagesTags(images_dir, tag_file_ext=wd4_file_ext)
    npz_files_list = searcher.tag_files()  # 储存特征向量的文件名列表

    # 读取特征向量
    (X_inside, X_outside, error_indies) = read_wd14_npz_files(
        dir = images_dir,
        npz_files_list = npz_files_list,
        layer_index = layer_index,
    )
    
    # 将外层特征向量布尔化用于特征重要性分析
    X_outside: npt.NDArray[np.bool_] = (X_outside > _WD14_OUTSIDE_BOOL_ARRAY_THRESHOLD)

    X_inside_dol_len = X_inside.shape[1]  # 内层特征向量列长度
    X_outside_col_len = X_outside.shape[1]  # 外层特征向量列长度
    try:
        # 读取对应的tag文件
        wd14_tags_toml_path = os.path.join(images_dir, WD14_TAGS_TOML_FILE)

        rating_tags, general_tags, characters_tags = read_wd14_tags_toml( wd14_tags_toml_path )
        tf_tags_list = rating_tags + general_tags + characters_tags  # 向量每列对应的tag

        tf_tags_list_len = len(tf_tags_list)  # 所读取到的总tag数量
        if tf_tags_list_len != X_outside_col_len:
            raise ValueError(f"读取的wd14_tags文件{wd14_tags_toml_path}中所指定的tags数量{tf_tags_list_len}与实际特征数量{X_outside_col_len}不一致")

    except Exception as e:
        logging.error(f"读取 {WD14_TAGS_TOML_FILE} 文件时出错，无法进行特征重要性分析来显示聚类标签，也无法使用部分特征功能：\n{e}")
        # 出错了就生成与outside层特征维度数(列数)一样数量的标签"error"
        tf_tags_list = [ "error" for i in range(X_outside_col_len) ]

    # 只有正确读取了tags文件才能启用部分特征聚类
    else:
        if able_to_use_tags_category_choices:

            tags_len_arr = np.array( [len(rating_tags), len(general_tags), len(characters_tags)] )  # 如 [4, 4000, 4000]
            cumsum = np.cumsum(tags_len_arr)  # 累加和  # 如 [4, 4004, 8004]
            total_len = cumsum[-1]  # 总长

            # 总长度tf_tags_list_len已经再try中判断过和外层特征数量对应
            # 其实able_to_use_tags_category_choices为真，那么内外两层特征数量应该一致
            # 这里再做一次保险
            assert_msg = f"内层{X_inside_dol_len}，外层{X_outside_col_len}，tags数{total_len}并不互相相等"
            assert (total_len == X_inside_dol_len) and (total_len == X_outside_col_len), assert_msg
            
            # 切割矩阵
            def partial_np_split_copy(arr):
                return np_split_copy(arr, indices_or_sections=cumsum[:-1], axis=1)  # 注意cumsum[:-1]不要取到最后一个点

            # 切成各个部分
            splited_X_outside = partial_np_split_copy(X_outside)
            splited_X_inside = partial_np_split_copy(X_inside)
            splited_tf_tags_list = partial_np_split_copy( np.array(tf_tags_list).reshape(1,-1) )  # 变为1行n列
            
            # 将rating对应的部分两极化，便于聚类
            # 注意这里可以直接将0索引视为rating_tags对应的部分，这已经在read_wd14_tags_toml()内部读取时完成了判断
            splited_X_outside[0] = _bipolarize(splited_X_outside[0])
            splited_X_inside[0] = _bipolarize(splited_X_inside[0])
            
            # 挑选所需矩阵并横向合并
            def partial_choose_and_concatenate(arr_list):
                return _choose_and_concatenate(arr_list, choose=list(tags_category_choices))  # list make pylance happy
            
            # 挑选所需矩阵并横向合并
            X_outside = partial_choose_and_concatenate(splited_X_outside)
            X_inside = partial_choose_and_concatenate(splited_X_inside)
            tf_tags_list = partial_choose_and_concatenate(splited_tf_tags_list).tolist()[0]  # 这个原返回1行n列矩阵，要特别处理下


    # 特征矩阵每行对应的文件名
    image_files_list = searcher.image_files_list
    assert image_files_list is not None, "image_files_list is None"  # 正常来说不会为None，make pylance happy

    return ( ( X_inside, X_outside ), tf_tags_list, image_files_list )


def vectorizer_exception_wrapper(func) -> Callable:
    """
    用于处理vectorizer函数的异常
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"{func.__name__}函数出现了异常: {e}")
            return (
                None,
                None,
                f"预处理出错: {e}",
                gr.update(value="预处理出错",interactive=False),
                gr.update(value="预处理出错",interactive=False),
                gr.update(interactive=False),
                gr.update(value="预处理出错",interactive=False),
            )
    return wrapper

@vectorizer_exception_wrapper
def vectorizer(
    images_dir: str,
    feature_extraction_method_Radio: int,
    # 文本聚类
    text_vectorizer_method: Literal[0, 1],
    use_comma_tokenizer: bool,
    use_binary_tokenizer: bool,
    text_feature_file_extension_name: str,
    # wd14特征聚类
    wd14_feature_layer_choice: Literal[0, 1],
    wd14_feature_file_extension_name: str,
    tags_category_choices_CheckboxGroup: List[Literal[0, 1, 2]],
    # 聚类方法
    cluster_model: int,
) ->    Tuple[
            Tuple[
                # 如果是tag聚类，则两个矩阵相同
                # 如果是wd14聚类，则第一个矩阵为入口层，第二个矩阵为出口层
                # 第一个矩阵将用于聚类； 第二矩阵的和用于特征重要性分析的tag长度对应，将用于特征重要性分析
                Tuple[np.ndarray, np.ndarray],
                # 向量每列对应的tag
                List[str],
                # 每行对应的图片名
                List[str]
            ],
            ClusterModelAlias,  # 聚类模型
            str,  # 处理完毕提示
            dict,  # 聚类分析按钮
            dict,  # 确定聚类按钮
            dict,  # SVD降维数
            dict,  # 确定SVD降维按钮
        ]:

    """读取images_dir下的tags文本或者wd14_npz文件，用其生成特征向量

    Args:
        images_dir (str): 图片目录
        feature_extraction_method_Radio (int): tag文本特征还是wd14特征
            请对照 FEATURE_EXTRACTION_METHOD_LIST
        text_vectorizer_method (int): tag文本特征提取方法
            请对照 TEXT_VECTORIZATION_METHOD_LIST
        use_comma_tokenizer (bool): 是否强制逗号分词
        use_binary_tokenizer (bool): 是否tag频率二值化
        text_feature_file_extension_name (str): tag文本文件扩展名
        wd14_feature_layer_choice (int): wd14特征层选择
            请对照 WD14_FEATURE_LAYER_CHOICE_LIST
        wd14_feature_file_extension_name (str): wd14特征向量文件扩展名
        cluster_model (int): 聚类模型选择
            请对照 CLUSTER_MODEL_LIST
    """


    global tag_file_ext, wd4_file_ext
    # 先把指定tag和wd14向量的扩展名全局变量改了
    tag_file_ext = text_feature_file_extension_name
    wd4_file_ext = wd14_feature_file_extension_name
    
    # 注意，两个特征提取方法，X[1] 要求输出npt.NDArray[np.bool_]
    # 其中真值代表相应的样本具有该tag特征，将用于特征重要性分析

    if feature_extraction_method_Radio == 0:
        X, tf_tags_list, image_files_list = text_vectorizer_func(
            images_dir = images_dir,
            text_vectorizer_method = text_vectorizer_method,
            use_comma_tokenizer = use_comma_tokenizer,
            use_binary_tokenizer = use_binary_tokenizer,
        )
    elif feature_extraction_method_Radio == 1:
        X, tf_tags_list, image_files_list = get_wd14_feature_func(
            images_dir = images_dir,
            wd14_feature_layer_choice = wd14_feature_layer_choice,
            tags_category_choices = tags_category_choices_CheckboxGroup,
        )
    else:
        raise ValueError(
            (
                f"feature_extraction_method_Radio = {feature_extraction_method_Radio}非法，"
                f"目前只支持{[i for i in range(len(FEATURE_EXTRACTION_METHOD_LIST))]}"
            )
        )


    # TODO: 在WebUI中开放参数选择    
    # 聚类模型选择
    def _get_model(model_index) -> ClusterModelAlias:
        cluser_model_choose_str = CLUSTER_MODEL_LIST[model_index]
        logging.info(f"选择了：{cluser_model_choose_str}")

        if model_index == 0 :
            assert cluser_model_choose_str == "K-Means聚类"
            cluster_model_State = skc.KMeans(n_init=KMEANS_N_INIT)
        elif model_index == 1 :
            assert cluser_model_choose_str == "Spectral谱聚类"
            cluster_model_State = skc.SpectralClustering( affinity='cosine' )
        elif model_index == 2 :
            assert cluser_model_choose_str == "Agglomerative层次聚类"
            cluster_model_State = skc.AgglomerativeClustering( metric='cosine', linkage='average' )
        elif model_index == 3 :
            assert cluser_model_choose_str == "OPTICS聚类"
            cluster_model_State = skc.OPTICS( metric="cosine")
        else:
            raise ValueError(
            (
                f"cluser_model_choose_str = {cluser_model_choose_str}非法, "
                f"目前只支持{[i for i in range(len(CLUSTER_MODEL_LIST))]}"
            )
        )

        return cluster_model_State
    cluster_model_State = _get_model(cluster_model)
    
    # 两个特征矩阵的形状
    X_inside, X_outside = X
    X_inside_shape = X_inside.shape
    X_outside_shape = X_outside.shape

    # 显示读取的特征维度给用户
    title_str  = textwrap.dedent(f"""\
        预处理完成\\
        特征维度: {[X_inside_shape, X_outside_shape]}\\
        tag数量: {len(tf_tags_list)}""")

    # SVD降维数滑条所允许选择的最大值，应该比样本数、特征数最小值小1，不然报错
    SVD_n_components_slider_maximum = min(X_inside_shape)-1

    return (
        (X, tf_tags_list, image_files_list),
        cluster_model_State,
        title_str,
        gr.update(value = "开始分析", interactive = True),
        gr.update(value = "开始聚类并展示结果", interactive = True),
        gr.update(
            minimum = 1,
            maximum = SVD_n_components_slider_maximum,
            value = max(1, round( DEFAULT_SVD_N_COMPONENTS_PERCENTAGE * SVD_n_components_slider_maximum )),
            interactive = True,
        ),
        gr.update(value = "开始降维", interactive = True),
    )


##############################  确定SVD降维  ##############################

def confirm_SVD_exception_wrapper(func) -> Callable:
    """
    用于处理confirm_SVD函数的异常
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"{func.__name__}函数出现了异常: {e}")
            return (
                f"SVD降维出错，请重新预处理: {e}",
                gr.update(),  # 不修改特征矩阵
                gr.update(value="SVD降维出错，请重新预处理",interactive=False),
            )
    return wrapper

@confirm_SVD_exception_wrapper
def confirm_SVD(
    SVD_n_components: int,
    vectorize_X_and_label_State: vectorize_X_and_label_State_Alias,
) -> Tuple[str, vectorize_X_and_label_State_Alias, dict]:

    # 需要降维的矩阵
    X_inside, X_outside = vectorize_X_and_label_State[0]
    
    # 样本数或者特征数为1时无法降维，抛出错误
    check_is_able_to_decomposition(min_n=1, decomposition_arr=X_inside)

    # 降维管道
    lsa = make_pipeline(
        TruncatedSVD(n_components=SVD_n_components), Normalizer(copy=False)
    )

    X_lsa = lsa.fit_transform(X_inside)  # 降维后所得矩阵
    explained_variance = lsa[0].explained_variance_ratio_.sum()  # 降维后矩阵可解释性占原来成分百分比

    # 赋值回去, make pylance happy
    def _change_tuple(vectorize_X_and_label_State):
        new_vectorize_X_and_label_State = list(vectorize_X_and_label_State)
        new_vectorize_X_and_label_State[0] = (X_lsa, X_outside)  # 把X_inside改了然后还原回tuple
        return tuple(new_vectorize_X_and_label_State)
    vectorize_X_and_label_State = _change_tuple(vectorize_X_and_label_State)

    # 降维后的矩阵
    X_inside, X_outside = vectorize_X_and_label_State[0]
    # tags列表
    tf_tags_list = vectorize_X_and_label_State[1]

    confirm_SVD_Markdown = textwrap.dedent(f"""\
        SVD降维完成\\
        降维后特征维度: {[X_inside.shape, X_outside.shape]}\\
        tag数量: {len(tf_tags_list)}\\
        对原矩阵解释性Explained variance of the SVD step: {explained_variance * 100:.1f}%""")

    return (
        confirm_SVD_Markdown,
        vectorize_X_and_label_State,
        gr.update(value = "再次预处理才可降维", interactive = False)
    )


##############################  聚类  ##############################

def cluster_images_exception_wrapper(func) -> Callable:
    """
    用于处理cluster_image函数的异常
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"{func.__name__}函数出现了异常: {e}")
            # 前一半是accordion的，后一半是gallery的
            gr_Accordion_list = [gr.update(visible=False) for i in range(MAX_GALLERY_NUMBER)]
            # 对于画廊，尽量不要把visible设为False
            gr_Gallery_list = [gr.update(value=None) for i in range(MAX_GALLERY_NUMBER)]  

            return gr_Accordion_list + gr_Gallery_list + [gr.update(visible=False)] + [gr.update(value={})]
    return wrapper

# TODO: cluster_feature_tags_list, clustered_images_list, pred_df 数据结构实在过于糟糕，需要重构
# TODO: 特征矩阵被命名为 X 实在过于糟糕，而且还在不同时期分别为出口层和内层，需要修改
@cluster_images_exception_wrapper
def cluster_images(
    images_dir: str,
    confirmed_cluster_number: int,
    use_cache: bool,
    global_dict_State: dict,
    vectorize_X_and_label_State: vectorize_X_and_label_State_Alias,
    cluster_model_State,
) -> List[dict]:
    """
    对指定目录下的图片进行聚类，将会使用该目录下与图片同名的txt中的tag做为特征
    confirmed_cluster_number为指定的聚类数
    如果use_cache为真，则会把图片拷贝到指定目录下的cache文件夹，并显示缩略图

    outputs=gr_Accordion_and_Gallery_list + [confirm_cluster_Row] + [global_dict_State]
    """
    

    assert len(vectorize_X_and_label_State) == 3, "vectorize_X_and_label_State应既包含特征向量组又包含对应的特征tag，还有图片名字"
    images_files_list = vectorize_X_and_label_State[2]  # 每行对应的图片名
    
    # 注意此时有两个特征向量，第一个为入口层，第二个为出口层
    # 第一个特征为入口层降维向量，特征数量可能与tags数不一样
    # 第二个特征是出口的向量，特征数量与tags数一直
    # 所以此处的X在预测完后，要在特征重要性分析之前换为出口向量
    assert len(vectorize_X_and_label_State[0]) == 2, "vectorize_X_and_label_State[0]应既包含入口层特征向量组又包含出口层特征向量组"
    X = vectorize_X_and_label_State[0][0]

    
    tf_tags_list = vectorize_X_and_label_State[1]  # 向量每列对应的tag
    assert len(tf_tags_list) == vectorize_X_and_label_State[0][1].shape[1], "向量每列对应的tag数量应与向量特征数量一致"
    
    # 聚类，最大聚类数不能超过样本数
    # 如果X是个二维矩阵，X.shape[0]应该能获取行数，即样本数
    n_clusters = min( confirmed_cluster_number, X.shape[0] )
    cluster_model = cluster_model_State
    
    # 自动根据不同模型设置n值
    cluster_model = aoto_set_sklearn_model_n_value(cluster_model, n_clusters)
    
    print("聚类算法", type(cluster_model).__name__)
    if isinstance(X, np.ndarray):
        print(f"预测特征维度 : {X.shape}")
    
    y_pred = cluster_model.fit_predict(X) # 训练模型并得到聚类结果
    logging.info(f"预测结果：\n{y_pred}")


    # 第一个特征为入口层降维向量，特征数量可能与tags数不一样
    # 所以此处的X在预测完后，要在特征重要性分析之前换为出口向量
    X = vectorize_X_and_label_State[0][1]  # 出口层向量，用于特征重要性分析
    X: npt.NDArray[np.bool_] = X.astype(bool, copy=True)  # 其原本就应该是bool类型，但是为了安全还是转换一下
    
    # 分类的ID
    clusters_ID = np.unique(y_pred)
    # 每个元素一个一个类的列表，按照clusters_ID的顺序排列
    clustered_images_list = [np.compress(np.equal(y_pred, i), images_files_list).tolist() for i in clusters_ID]
    
    # #################################
    # pandas处理数据
    
    # 注意！！！！不要改变{"images_file":images_files_list, "y_pred":y_pred}的位置，前两列必须是这两个
    def _create_pred_df(X, tf_tags_list, images_files_list, y_pred):
        """
        创建包含了聚类结果的panda.DataFrame
        X为特征向量， tf_tags_list为特征向量每列对应的tag， images_files_list为每行对应的图片名字， y_pred为每行预测结果
        """
        vec_df = pd.DataFrame(X,columns=tf_tags_list)  # 向量
        cluster_df = pd.DataFrame( {"images_file":images_files_list, "y_pred":y_pred} )  #聚类结果
        pred_df = pd.concat([cluster_df, vec_df ], axis=1)
        return pred_df

    pred_df = _create_pred_df(X, tf_tags_list, images_files_list, y_pred)  # 包含了聚类结果，图片名字，和各tag维度的向量值

    # 确保标签部分矩阵为bool类型
    # 其实前面如果X正确转换为bool类型，这里是显然成立的，因为相应的部分就是X
    assert (pred_df.iloc[:,2:].dtypes == bool).all(), "pred_df.iloc[:,2:]不是bool类型"


    def find_duplicate_tags(tags_df):
        """ 找出一个dataframe内每一行都不为0的列，返回一个pandas.Index对象 """
        tags_columns_index = tags_df.columns  # 取出列标签
        duplicate_tags_index = tags_columns_index[ tags_df.all(axis=0) ] # 找出每一行都为真,即不为0的列名，即共有的tags
        return duplicate_tags_index  # 输出pandas.Index对象
    
    # 其中 pred_df.iloc[:,2:] 为与tags有关部分
    common_duplicate_tags_set = set( find_duplicate_tags( pred_df.iloc[:,2:] ) )
    

    def _cluster_feature_select(clusters_ID, X: npt.NDArray[np.bool_], y_pred, pred_df: pd.DataFrame):
        """
        根据聚类结果，评估是哪个特征导致了这个类的生成
        clusters_ID是聚类后会有的标签列表如[-1,0,1...]  只有DBSCAN会包含-1,代表噪声
        X是特征向量，应为bool类型矩阵，真假应该对应该tag的有无
            用于chi2算法找出对应的列
        y_pred是聚类结果
        pred_df是聚类信息dataframe，要求对应的部分为bool类型矩阵，真假应该对应该tag的有无
            用找出的列，经过占比判断得出最终的 tag: List[str]

        Returns:
            cluster_feature_tags_list: 一个列表，子元素为字典，字典中包含了每一个聚类的prompt和negetive tags。按照clusters_ID的顺序排列
        """
        
        cluster_feature_tags_list = []

        # TODO: 显示的重要性tag特征似乎有明显问题，需要尽快改进重要性tag的判断方法
        for i in clusters_ID:
            # 将第i类和其余的类二分类
            temp_pred = y_pred.copy()
            temp_pred[ temp_pred != i ] = i+1  # 将第i类以外的类都置为i+1，这样就可以二分类了

            k = min(10, X.shape[1])  # 最多只选10个特征，不够10个就按特征数来

            # 特征选择器
            # Selector = SelectPercentile(chi2, percentile=30)  # 按百分比选择
            Selector = SelectKBest(chi2, k=k)  # 按个数选择

            # 开始选择
            X_selected = Selector.fit_transform(X, temp_pred)  # 被选择的特征向量矩阵 用于调试
            X_selected_index = Selector.get_support(indices=True)  # 被选中的特征向量所在列的索引
            tags_selected = np.array(tf_tags_list)[X_selected_index]  # 对应的被选择的tags列表 用于调试

            # 将pred_df中第i类的部分拿出来分析
            cluster_df: pd.DataFrame = pred_df[pred_df["y_pred"] == i]  # 取出所处理的第i聚类df部分
            cluster_tags_df = cluster_df.iloc[:,2:]  # 取出第i聚类df中与tags有关部分  # 前两列是图片名字和聚类预测结果y_pred，不要
            cluster_selected_tags_df = cluster_tags_df.iloc[:,X_selected_index]  # 第i聚类df中被认为是照成了此次聚类结果tags
            
            # 这里认为传入的特征矩阵，如果某一个样本有该tag属性，则其相应的列值应为1，否则为0
            # 只要这个聚类有一个样本有，就算做prompt； 如果都没有，则算做negetive
            # 这里的columns已经是带有tags名了
            prompt_tags_list = cluster_selected_tags_df.columns[ cluster_selected_tags_df.mean(axis=0) >= CLUSTER_IMPORTANT_FEATURE_PROMPT_TAGS_THRESHOLD ].tolist()
            negetive_tags_list = cluster_selected_tags_df.columns[ cluster_selected_tags_df.mean(axis=0) < CLUSTER_IMPORTANT_FEATURE_NEGETIVE_TAGS_THRESHOLD ].tolist()

            # 最终的特征重要性分析结果，为一个列表，子元素为字典，字典中包含了每一个聚类的prompt和negetive tags
            # 不要改 "prompt" 和 "negetive" 的名字，下面会用到
            # TODO: 将 "prompt" 和 "negetive" 改为常量
            cluster_feature_tags_list.append( {"prompt":prompt_tags_list, "negetive":negetive_tags_list} )

        return cluster_feature_tags_list

    if isinstance(X, np.ndarray):
        print(f"重要性分析特征维度 : {X.shape}")
    cluster_feature_tags_list = _cluster_feature_select(clusters_ID, X, y_pred, pred_df)


    # 赋值到全局组件中，将会传递至confirm_cluster_button.click
    global_dict_State["common_duplicate_tags_set"] = common_duplicate_tags_set  # 好像弃用了？
    global_dict_State["cluster_feature_tags_list"] = cluster_feature_tags_list
    global_dict_State["clustered_images_list"] = clustered_images_list
    global_dict_State["images_dir"] = images_dir

    # 缓存缩略图
    exist_cache_error = False
    cache_dir = os.path.join(images_dir, CACHE_FOLDER_NAME)
    if use_cache:
        cache_images_list = []
        for cluster in clustered_images_list:
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

    # TODO: 重构生成画廊为函数

    # 注意，返回的列表长度为显示gallery组件数的两倍
    # 因为列表里偶数位为Accordion组件，奇数位为Gallery组件
    visible_gr_gallery_list: List[dict] = []
    visible_gr_Accordion_list: List[dict] = []
    # 最多只能处理MAX_GALLERY_NUMBER个画廊
    clustered_images_list_len = len(clustered_images_list)
    if clustered_images_list_len > MAX_GALLERY_NUMBER:
        logging.warning(f"聚类数达到了{clustered_images_list_len}个，只显示前{MAX_GALLERY_NUMBER}个")

    # 显示有图的画廊
    # cluster_feature_tags_list和clustered_images_list都是按照聚类结果的ID顺序排列的
    for i in range( min( len(clustered_images_list), MAX_GALLERY_NUMBER ) ):
        gallery_images_tuple_list = [
            (os.path.join(gallery_images_dir, name), name)
            for name in clustered_images_list[i]
        ]
        # 每类的tag标题
        prompt = cluster_feature_tags_list[i].get("prompt", [])
        negetive = cluster_feature_tags_list[i].get("negetive", [])

        visible_gr_Accordion_list.append(
            gr.update( visible=True, label=f"聚类{i} : prompt: {prompt} ### negetive: {negetive}" )
        )
        visible_gr_gallery_list.append(
            gr.update( value=gallery_images_tuple_list, visible=True)
        )
    # 隐藏无图的画廊
    unvisible_gr_gallery_list: List[dict] = [ gr.update( visible=False ) for i in range( MAX_GALLERY_NUMBER-len(visible_gr_gallery_list) ) ]
    unvisible_gr_Accordion_list: List[dict] = [ gr.update( visible=False ) for i in range( MAX_GALLERY_NUMBER-len(visible_gr_Accordion_list) ) ]

    # 汇总有图和无图
    all_gr_gallery_list = visible_gr_gallery_list + unvisible_gr_gallery_list
    all_gr_Accordion_list = visible_gr_Accordion_list + unvisible_gr_Accordion_list

    return all_gr_Accordion_list + all_gr_gallery_list + [gr.update(visible=True)] + [global_dict_State]


##############################  聚类最佳n值分析  ##############################

def cluster_analyse_exception_wrapper(func) -> Callable:
    """
    用于处理cluster_analyse函数的异常
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"{func.__name__}函数出现了异常: {e}")
            return [None, None, None, None]
    return wrapper

@cluster_analyse_exception_wrapper
def cluster_analyse(
    max_cluster_number: int,
    vectorize_X_and_label_State: vectorize_X_and_label_State_Alias,
    cluster_model_State
) -> Tuple[
        dict,  # 轮廓系数图
        dict,  # Davies系数图
        dict,  # 最佳聚类数
        dict,  # 样本散点图
    ]:
    """
    将评估从聚类数从 2~max_cluster_number 的效果
    返回matplotlib类型的肘部曲线和轮廓系数

    outputs=[Silhouette_gr_Plot, Davies_gr_Plot, bset_cluster_number_DataFrame, samples_ScatterPlot]
    """

    
    # 提取特征标签
    # 注意此时有两个特征向量，第一个为入口层，第二个为出口层
    # 第一个特征为入口层降维向量，特征数量可能与tags数不一样
    # 第二个特征是出口的向量，特征数量与tags数一致
    # 这里不进行特征重要性分析，也不涉及到特征与tag一一对应，所以可以不变为出口向量
    assert len(vectorize_X_and_label_State) == 3, "vectorize_X_and_label_State应既包含特征向量组又包含对应的特征tag，还有图片名字"
    assert len(vectorize_X_and_label_State[0]) == 2, "vectorize_X_and_label_State[0]应既包含入口层特征向量组又包含出口层特征向量组"
    X = vectorize_X_and_label_State[0][0]

    if isinstance(X, np.ndarray):
        print(f"n值分析特征维度 : {X.shape}")
    
    # 使用肘部法则和轮廓系数确定最优的聚类数
    davies_bouldin_scores_list = [] # 存储每个聚类数对应的davies系数
    silhouette_scores_list = []  # 用于存储不同k值对应的轮廓系数
    
    # 最大聚类数不能超过样本,最多只能样本数-1
    # 如果X是个二维矩阵，len(X)应该能获取行数，即样本数
    print(f"最大样本数 {X.shape[0]}")
    if X.shape[0] < 3:
        raise ValueError("样本数过少，无法进行聚类分析")
    k_range = range( 2, min(max_cluster_number+1, len(X) ) )  # 聚类数的范围(左闭右开)

    final_clusters_number = 0  # 最终的聚类的次数

    cluster_model = cluster_model_State # 获取模型
    print("聚类分析开始")
    print("分析算法", type(cluster_model_State).__name__)
    for k in tqdm( k_range ):

        # 自动根据不同模型设置n值
        cluster_model = aoto_set_sklearn_model_n_value(cluster_model, k)

        y_pred = cluster_model.fit_predict(X) # 训练模型
        
        # 如果出现只有一个聚类的情况，或者每一个图片都自成一类，就不用继续分析了
        if len( np.unique(y_pred) ) == 1 or len( np.unique(y_pred) ) == len(X):
            break
        
        sil_score = silhouette_score(X, y_pred)  # 计算轮廓系数,聚类数为1时无法计算
        dav_score= davies_bouldin_score(X, y_pred)  # 计算davies系数

        silhouette_scores_list.append(sil_score) # 储存轮廓系数
        davies_bouldin_scores_list.append(dav_score) # 储存
        final_clusters_number += 1  # 记录最终的聚类数

    """注意，下面的pf中类似 "file", "x", "y"的键名不要改，除非修改相应的gr.update """

    # 可视化
    print("可视化开始")
    samples_visualization_np = visualization_2D(X)
    samples_visualization_df = pd.DataFrame(
        {
            "file": vectorize_X_and_label_State[2],
            "x": samples_visualization_np[:,0],
            "y": samples_visualization_np[:,1],
        }
    )
    
    Silhouette_df = pd.DataFrame( {"x":k_range[0:len(silhouette_scores_list)], "y":silhouette_scores_list} )
    # 注意，这里Davies_df的y值乘了 -1，因为Davies系数原本是越小越好，这里让它变得和轮廓系数一样越大越好
    Davies_df = pd.DataFrame( {"x":k_range[0:len(davies_bouldin_scores_list)], "y":( -1 * np.array(davies_bouldin_scores_list) ) } )
    
    _df = pd.concat( [ Silhouette_df.loc[:,"y"], Davies_df.loc[:,"y"] ], axis=1, keys=["Silhouette",'Davies'])
    logging.info(_df)  # 打印分数

    # 由实际聚类次数决定展示的聚类次数
    head_number = max( 1, min( 10, round( math.log2(final_clusters_number) ) ) )  # 展示log2(实际聚类数)个，最少要展示1个，最多展示10个
    # 对轮廓系数从大到小排序，展示前head_number个
    _bset_cluster_number_DataFrame = Silhouette_df.sort_values(by='y', ascending=False).head(head_number)

    Silhouette_LinePlot = gr.update(
        value=Silhouette_df,
        label="轮廓系数",
        x="x",
        y="y",
        tooltip=["x", "y"],
        x_title="Number of clusters",
        y_title="Silhouette Score",
        title="Silhouette Method",
        overlay_point=True,
        width=400,
        visible=True,
    )
    Davies_LinePlot = gr.update(
        value=Davies_df,
        label="Davies-Bouldin指数",
        x="x",
        y="y",
        tooltip=["x", "y"],
        x_title="Number of clusters",
        y_title="Davies_bouldin Scores",
        title="Davies-Bouldin Method",
        overlay_point=True,
        width=400,
        visible=True,
    )
    bset_cluster_number_DataFrame = gr.update(
        value=_bset_cluster_number_DataFrame,
        label="根据轮廓曲线推荐的聚类数（y越大越好）",
        visible=True,
    )
    # TODO: 聚类结果出来后加上颜色，考虑用一个gr.state来传递df
    samples_ScatterPlot = gr.update(
        value=samples_visualization_df,
        label="样本分布图",
        x="x",
        y="y",
        tooltip=["file", "x", "y"],
        title="Sample distribution",
        visible=True,
    )

    print("聚类分析结束")
    
    # 绘制肘部曲线
    return Silhouette_LinePlot, Davies_LinePlot, bset_cluster_number_DataFrame, samples_ScatterPlot


##############################  确定聚类  ##############################

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
                gr.update(visible=False),  # 确定聚类结果操作的那一行全部隐藏
                gr.update(value="出错，解决后请重新预处理", interactive=False),  # 分析按钮
                gr.update(value="出错，解决后请重新预处理", interactive=False)  # 确定聚类按钮
            )
    return wrapper

@confirm_cluster_exception_wrapper
def confirm_cluster(
    process_clusters_method:int,
    global_dict_State: dict,
) -> Tuple[dict, dict, dict]:
    """
    根据选择的图片处理方式，对global_dict_State中聚类后的图片列表，以及路径进行相关操作
    
    # 以下是输入参数的例子：
    
    process_clusters_method = gr.Radio(label="图片处理方式",
                                       choices=["重命名原图片","在Cluster文件夹下生成聚类副本","移动原图至Cluster文件夹"],
                                       type="index",
    )
    
    global_dict_State["common_duplicate_tags_set"] = common_duplicate_tags_set
    global_dict_State["cluster_feature_tags_list"] = cluster_feature_tags_list
    global_dict_State["clustered_images_list"] = clustered_images_list
    global_dict_State["images_dir"] = images_dir
    
    """
    
    images_dir = global_dict_State.get("images_dir", "")
    clustered_images_list = global_dict_State.get("clustered_images_list", [] )

    # 输出信息的同时可以判断输入值是否非法
    process_clusters_method_choose_str = PROCESS_CLUSTERS_METHOD_CHOICES[process_clusters_method]
    logging.info(f"选择了：{process_clusters_method_choose_str}")
    
    # 带时间戳的重命名原图片和附带文件
    if process_clusters_method == 0:
        assert process_clusters_method_choose_str == "重命名原图片(不推荐)"
        operation = "rename"
    # 在Cluster文件夹下生成聚类副本
    elif process_clusters_method == 1:
        assert process_clusters_method_choose_str == f"在{CLUSTER_DIR_PREFIX}文件夹下生成聚类副本(推荐)"
        operation = "copy"
    # 移动原图至Cluster文件夹
    elif process_clusters_method == 2:
        assert process_clusters_method_choose_str == f"移动原图至{CLUSTER_DIR_PREFIX}文件夹(大数据集推荐)"
        operation = "move"
    else:
        raise ValueError(
            (
                f"process_clusters_method = {process_clusters_method}非法, "
                f"目前只支持{[i for i in range(len(PROCESS_CLUSTERS_METHOD_CHOICES))]}"
            )
        )

    operate_images_file(
        images_dir,
        clustered_images_list,
        extra_file_ext_list = [tag_file_ext, wd4_file_ext],
        copy_to_subfolder_file_list = [WD14_TAGS_TOML_FILE],  # 把特征tag文件也复制到每一个子目录下
        operation = operation,
    )
    
    # 如果是重命名或者移动，会改变原有图片路径，要求用户重新预处理
    if operation in ["rename", "move"]:
        cluster_analyse_button_update = gr.update(value="请先预处理再分析", interactive=False)
        cluster_images_button = gr.update(value="请先预处理再开始聚类", interactive=False)
    else:
        cluster_analyse_button_update = gr.update()
        cluster_images_button = gr.update()
    
    return (
        gr.update(visible=False),  # 确定聚类结果操作的那一行全部隐藏
        cluster_analyse_button_update,
        cluster_images_button,
    )
