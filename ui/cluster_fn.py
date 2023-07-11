import os
from typing import List, Tuple, Callable, Union, Literal
import logging
import math


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
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import gradio as gr
from tqdm import tqdm


from ui.tools.operate_images import (
    cache_images_file,
    operate_images_file,
    cluster_dir_prefix,
)
from ui.tools.SearchImagesTags import SearchImagesTags
from tag_images_by_wd14_tagger import (
    DEFAULT_TAGGER_CAPTION_EXTENSION,  # 默认打标文件的扩展名
    WD14_NPZ_EXTENSION,  # 用于保存推理所得特征向量的文件扩展名 # .wd14用来区分kohya的潜变量cache
    WD14_TAGS_TOML_FILE,  # 存储各列向量对应的tag的文件的名字
    read_wd14_tags_toml,  # 读取存储各列向量对应的tag的文件的内容以获取tags列表
    WD14_NPZ_ARRAY_PREFIX,
)


##############################  常量  ##############################

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
    f"在{cluster_dir_prefix}文件夹下生成聚类副本(推荐)",
    f"移动原图至{cluster_dir_prefix}文件夹(大数据集推荐)",
]

CLUSTER_MODEL_LIST = [
    "K-Means聚类",
    "Spectral谱聚类",
    "Agglomerative层次聚类",
    "OPTICS聚类"
]

KMEANS_N_INIT = 8  # KMeans聚类时的n_init参数


##############################  全局变量  ##############################

# TODO: 换成gr.State，这样可以在界面刷新后失效，和避免多用户间干扰
tag_file_ext = DEFAULT_TAGGER_CAPTION_EXTENSION  # 存放特征tag的文件后缀名
wd4_file_ext = WD14_NPZ_EXTENSION  # 存放特征向量的文件后缀名


##############################  聚类特征选择Box切换  ##############################
def feature_extraction_method_change_trigger(feature_extraction_method_index: int):
    gr_Box_update_list = [ gr.update(visible=False) for i in FEATURE_EXTRACTION_METHOD_LIST ]
    gr_Box_update_list[feature_extraction_method_index] = gr.update(visible=True)
    return gr_Box_update_list


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
) -> Tuple[Tuple[np.ndarray, np.ndarray], List[str], List[str]]:

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
    print(f"选择了{text_vectorizer_method_str}")

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
    X = tfvec.fit_transform(tag_content_list).toarray()  # type: ignore # 向量特征
    X = (X, X)  # 为了与WD14的输出保持一致，这里将X变为二元组

    tf_tags_list = tfvec.get_feature_names_out().tolist()  # 向量每列对应的tag
    # stop_tags = tfvec.stop_words_  # 被过滤的tag

    # 特征矩阵每行对应的文件名
    image_files_list = searcher.image_files_list
    assert image_files_list is not None, "image_files_list is None"  # 正常来说不会为None，这里为了让pylance开心

    return (X, tf_tags_list, image_files_list)


# wd14特征向量聚类
def get_wd14_feature_func(
    images_dir: str,
    wd14_feature_layer_choice: Literal[0, 1]
) -> Tuple[Tuple[np.ndarray, np.ndarray], List[str], List[str]]:

    # 读取储存在.wd14.npz文件中的向量特征
    searcher = SearchImagesTags(images_dir, tag_file_ext=wd4_file_ext)
    npz_files_list = searcher.tag_files()  # 储存特征向量的文件名列表

    # 输出信息的同时可以判断输入值是否非法
    wd14_feature_layer_choose_str = WD14_FEATURE_LAYER_CHOICE_LIST[wd14_feature_layer_choice]
    print(f"选择了{wd14_feature_layer_choose_str}")

    if wd14_feature_layer_choice == 0:
        assert wd14_feature_layer_choose_str == "predictions_sigmoid 全向量层"
        layer_index = 0
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

    # 用来暂存向量
    X_inside = []  # 内层，取决于layer_index
    X_outside = []  # 出口层，固定取最外层
    error_indies = []  # 读取错误的文件索引
    for index, npz_file in enumerate(npz_files_list):
        try:
            with np.load( os.path.join(images_dir, npz_file) ) as npz:
                X_inside.append(npz[f"{WD14_NPZ_ARRAY_PREFIX}{layer_index}"])
                X_outside.append(npz[f"{WD14_NPZ_ARRAY_PREFIX}0"])  # 0强制指定为最外层
        except Exception as e:
            logging.error(f"读取 {npz_file} 向量特征时出错，其将会被置一：\n{e}")
            # 读错了就跳过
            X_inside.append(None)
            X_outside.append(None)
            error_indies.append(index)  # 记录出错位置
            continue
    
    if len(error_indies) == len(npz_files_list):
        raise ValueError("所有向量特征读取错误，无法继续")
    
    def check_error(need_check_list: List[np.ndarray], error_indies: List[int]):
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
    # 注意，请先检查数据，因为如果读取出错了列表里面会有None元素，无法转为矩阵
    check_error(X_inside, error_indies)
    check_error(X_outside, error_indies)
    
    # list[np.ndarray]  # np.ndarray.shape = (n_samples, n_features)
    X = ( np.array(X_inside), np.array(X_outside) )

    X_outside_col_len = X[1].shape[1]  # 外层特征向量列长度
    try:
        # 读取对应的tag文件
        wd14_tags_toml_path = os.path.join(images_dir, WD14_TAGS_TOML_FILE)

        rating_tags, general_tags, characters_tags = read_wd14_tags_toml( wd14_tags_toml_path )
        tf_tags_list = rating_tags + general_tags + characters_tags  # 向量每列对应的tag

        tf_tags_list_len = len(tf_tags_list)  # 所读取到的总tag数量
        if tf_tags_list_len != X_outside_col_len:
            raise ValueError(f"读取的wd14_tags文件{wd14_tags_toml_path}中所指定的tags数量{tf_tags_list_len}与实际特征数量{X_outside_col_len}不一致")
    except Exception as e:
        logging.error(f"读取 {WD14_TAGS_TOML_FILE} 文件时出错，无法进行特征重要性分析，后续不会显示聚类标签：\n{e}")
        # 出错了就生成与outside层特征维度数(列数)一样数量的标签"error"
        tf_tags_list = [ "error" for i in range(X_outside_col_len) ]

    # 特征矩阵每行对应的文件名
    image_files_list = searcher.image_files_list
    assert image_files_list is not None, "image_files_list is None"  # 正常来说不会为None，make pylance happy

    return (X, tf_tags_list, image_files_list)


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
        Union[skc.KMeans, skc.SpectralClustering, skc.AgglomerativeClustering, skc.OPTICS],  # 聚类模型
        str,  # 处理完毕提示
        dict,  # 聚类分析按钮
        dict,  # 确定聚类按钮
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

    Returns:
        _type_: outputs=[vectorize_X_and_label_State, cluster_model_State, preprocess_Markdown]
    """


    global tag_file_ext, wd4_file_ext
    # 先把指定tag和wd14向量的扩展名全局变量改了
    tag_file_ext = text_feature_file_extension_name
    wd4_file_ext = wd14_feature_file_extension_name
    

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
    def _get_model(model_index) -> Union[skc.KMeans, skc.SpectralClustering, skc.AgglomerativeClustering, skc.OPTICS]:
        cluser_model_choose_str = CLUSTER_MODEL_LIST[model_index]
        print(f"选择了{cluser_model_choose_str}")

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
    
    # 显示读取的特征维度给用户
    title_str = (" " * 3) + \
                f"特征维度: {  X.shape if isinstance(X, np.ndarray) else [i.shape for i in X] }" + \
                (" " * 3) + \
                f"tag数量: {len(tf_tags_list)}"
    
    return (
        (X, tf_tags_list, image_files_list),
        cluster_model_State,
        "预处理完成" + title_str,
        gr.update(value = "开始分析", interactive = True),
        gr.update(value = "开始聚类并展示结果", interactive = True),
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

@cluster_images_exception_wrapper
def cluster_images(
    images_dir: str,
    confirmed_cluster_number: int,
    use_cache: bool,
    global_dict_State: dict,
    vectorize_X_and_label_State: Tuple,
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

    
    # TODO: SVD降维
    """
    SVD = TruncatedSVD( n_components = X.shape[1] - 1 )
    X_SVD = SVD.fit_transform(X)
    X_SVD = normalize(X_SVD)
    """
    
    # 聚类，最大聚类数不能超过样本数
    # 如果X是个二维矩阵，X.shape[0]应该能获取行数，即样本数
    n_clusters = min( confirmed_cluster_number, X.shape[0] )
    cluster_model = cluster_model_State
    
    # 根据模型的不同设置不同的参数n
    # 目前是除了OPTICS，其他都是n_clusters
    # TODO: 把设置n值部分提取出来，和聚类分析那一块重构为公共函数
    if "n_clusters" in cluster_model.get_params().keys():
        cluster_model.set_params(n_clusters=n_clusters)
    # 对应OPTICS聚类
    elif "min_samples" in cluster_model.get_params().keys():
        cluster_model.set_params(min_samples=n_clusters)
    else:
        assert False, "选择的模型中，n参数指定出现问题"
    
    print("聚类算法", type(cluster_model).__name__)
    if isinstance(X, np.ndarray):
        print(f"预测特征维度 : {X.shape}")
    
    y_pred = cluster_model.fit_predict(X) # 训练模型并得到聚类结果
    print(y_pred)

    
    # 第一个特征为入口层降维向量，特征数量可能与tags数不一样
    # 所以此处的X在预测完后，要在特征重要性分析之前换为出口向量
    X = vectorize_X_and_label_State[0][1]
    
    # 分类的ID
    clusters_ID = np.unique(y_pred)
    # 每个元素一个一个类的列表
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
    # 确保标签部分矩阵为数值类型
    try:
        pred_df.iloc[:,2:].astype(float)
    except Exception as e:
        logging.error(f"pred_df的标签部分包含了非法值，可能后续会引发意外错误\nerror: {e}")


    def find_duplicate_tags(tags_df):
        """ 找出一个dataframe内每一行都不为0的列，返回一个pandas.Index对象 """
        tags_columns_index = tags_df.columns  # 取出列标签
        duplicate_tags_index = tags_columns_index[ tags_df.all(axis=0) ] # 找出每一行都为真,即不为0的列名，即共有的tags
        return duplicate_tags_index  # 输出pandas.Index对象
    
    # 其中 pred_df.iloc[:,2:] 为与tags有关部分
    common_duplicate_tags_set = set( find_duplicate_tags( pred_df.iloc[:,2:] ) )
    

    def _cluster_feature_select(clusters_ID, X, y_pred, pred_df):
        """
        根据聚类结果，评估是哪个特征导致了这个类的生成
        clusters_ID是聚类后会有的标签列表如[-1,0,1...]  只有DBSCAN会包含-1,代表噪声
        X是特征向量
        y_pred是聚类结果
        pred_df是聚类信息dataframe
        """
        
        cluster_feature_tags_list = []

        # TODO: 显示的重要性tag特征似乎有明显问题，需要尽快改进重要性tag的判断方法
        for i in clusters_ID:
            # 将第i类和其余的类二分类
            temp_pred = y_pred.copy()
            temp_pred[ temp_pred != i ] = i+1
            
            k = min(10, X.shape[1])  # 最多只选10个特征，不够10个就按特征数来
            
            # 特征选择器
            # Selector = SelectPercentile(chi2, percentile=30)  # 按百分比选择
            Selector = SelectKBest(chi2, k=k)  # 按个数选择
            
            # 开始选择
            X_selected = Selector.fit_transform(X, temp_pred)  # 被选择的特征向量矩阵 用于调试
            X_selected_index = Selector.get_support(indices=True)  # 被选中的特征向量所在列的索引
            tags_selected = np.array(tf_tags_list)[X_selected_index]  # 对应的被选择的tags列表 用于调试
            
            # 将pred_df中第i类的部分拿出来分析
            cluster_df = pred_df[pred_df["y_pred"] == i]  # 取出所处理的第i聚类df部分
            cluster_tags_df = cluster_df.iloc[:,2:]  # 取出第i聚类df中与tags有关部分
            cluster_selected_tags_df = cluster_tags_df.iloc[:,X_selected_index]  # 第i聚类df中被选择的tags
            
            # 这些被选择的特征，只要这个聚类有一个样本有，就算做prompt； 如果都没有，则算做negetive
            prompt_tags_list = cluster_selected_tags_df.columns[ cluster_selected_tags_df.any(axis=0) ].tolist()
            negetive_tags_list = cluster_selected_tags_df.columns[ ~cluster_selected_tags_df.any(axis=0) ].tolist()
            
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
    cache_dir = os.path.join(images_dir, CACHE_FOLDER_NAME)
    if use_cache:
        cache_images_list = []
        for cluster in clustered_images_list:
            for name in cluster:
                image_path = os.path.join(images_dir, name)
                cache_images_list.append(image_path)
        if cache_images_list:
            cache_images_file(cache_images_list, cache_dir, resolution=CACHE_RESOLUTION)
     
    # 如果用缓存就展示缓存图
    gallery_images_dir = images_dir if not use_cache else cache_dir

    # TODO: 有可能因为缓存失败照成图片无法显示

    # 注意，返回的列表长度为显示gallery组件数的两倍
    # 因为列表里偶数位为Accordion组件，奇数位为Gallery组件
    visible_gr_gallery_list: List[dict] = []
    visible_gr_Accordion_list: List[dict] = []
    # 最多只能处理MAX_GALLERY_NUMBER个画廊
    clustered_images_list_len = len(clustered_images_list)
    if clustered_images_list_len > MAX_GALLERY_NUMBER:
        logging.warning(f"聚类数达到了{clustered_images_list_len}个，只显示前{MAX_GALLERY_NUMBER}个")

    for i in range( min( len(clustered_images_list), MAX_GALLERY_NUMBER ) ):
        gallery_images_tuple_list = [
            (os.path.join(gallery_images_dir, name), name)
            for name in clustered_images_list[i]
        ]
        prompt = cluster_feature_tags_list[i].get("prompt", [])
        negetive = cluster_feature_tags_list[i].get("negetive", [])

        visible_gr_Accordion_list.append(
            gr.update( visible=True, label=f"聚类{i} :\nprompt: {prompt}\nnegetive: {negetive}" )
        )
        visible_gr_gallery_list.append(
            gr.update( value=gallery_images_tuple_list, visible=True)
        )
    unvisible_gr_gallery_list: List[dict] = [ gr.update( visible=False ) for i in range( MAX_GALLERY_NUMBER-len(visible_gr_gallery_list) ) ]
    unvisible_gr_Accordion_list: List[dict] = [ gr.update( visible=False ) for i in range( MAX_GALLERY_NUMBER-len(visible_gr_Accordion_list) ) ]

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
            return [None, None, None]
    return wrapper

@cluster_analyse_exception_wrapper
def cluster_analyse(
    max_cluster_number: int,
    vectorize_X_and_label_State:list,
    cluster_model_State
) -> Tuple[dict, dict, dict]:
    """
    将评估从聚类数从 2~max_cluster_number 的效果
    返回matplotlib类型的肘部曲线和轮廓系数

    outputs=[Silhouette_gr_Plot, Davies_gr_Plot, bset_cluster_number_DataFrame]
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
    print(f"最大样本数 {len(X)}")
    if len(X) < 3:
        raise ValueError("样本数过少，无法进行聚类分析")
    k_range = range( 2, min(max_cluster_number+1, len(X) ) )  # 聚类数的范围(左闭右开)

    final_clusters_number = 0  # 最终的聚类的次数

    print("聚类分析开始")
    for k in tqdm( k_range ):
        cluster_model = cluster_model_State # 获取模型
        
        # 根据模型的不同设置不同的参数n
        if "n_clusters" in cluster_model.get_params().keys():
            cluster_model.set_params(n_clusters=k)
        elif "min_samples" in cluster_model.get_params().keys():
            cluster_model.set_params(min_samples=k)
        else:
            assert False, "选择的模型中，n参数指定出现问题"
        
        y_pred = cluster_model.fit_predict(X) # 训练模型
        
        # 如果出现只有一个聚类的情况，或者每一个图片都自成一类，就不用继续分析了
        if len( np.unique(y_pred) ) == 1 or len( np.unique(y_pred) ) == len(X):
            break
          
        sil_score = silhouette_score(X, y_pred)  # 计算轮廓系数,聚类数为1时无法计算
        silhouette_scores_list.append(sil_score) # 储存轮廓系数
        
        dav_score= davies_bouldin_score(X,y_pred)  # 计算davies系数
        davies_bouldin_scores_list.append(dav_score) # 储存

        final_clusters_number += 1  # 记录最终的聚类数
    
    print("分析算法", type(cluster_model_State).__name__)
    
    Silhouette_df = pd.DataFrame( {"x":k_range[0:len(silhouette_scores_list)], "y":silhouette_scores_list} )
    # 注意，这里Davies_df的y值乘了 -1，因为Davies系数原本是越小越好，这里让它变得和轮廓系数一样越大越好
    Davies_df = pd.DataFrame( {"x":k_range[0:len(davies_bouldin_scores_list)], "y":( -1 * np.array(davies_bouldin_scores_list) ) } )
    
    _df = pd.concat( [ Silhouette_df.loc[:,"y"], Davies_df.loc[:,"y"] ], axis=1, keys=["Silhouette",'Davies'])
    print(_df)  # 打印分数

    Silhouette_LinePlot = gr.update(value=Silhouette_df,
                                    label="轮廓系数",
                                    x="x",
                                    y="y",
                                    tooltip=["x", "y"],
                                    x_title="Number of clusters",
                                    y_title="Silhouette score",
                                    title="Silhouette Method",
                                    overlay_point=True,
                                    width=400,
    )
    Davies_LinePlot = gr.update(value=Davies_df,
                                    label="Davies-Bouldin指数",
                                    x="x",
                                    y="y",
                                    tooltip=["x", "y"],
                                    x_title="Number of clusters",
                                    y_title="-1 * np.array(davies_bouldin_scores_list)",
                                    title="Davies-Bouldin Method",
                                    overlay_point=True,
                                    width=400,
    )
    
    # 由实际聚类次数决定展示的聚类次数
    head_number = max( 1, min( 10, round( math.log2(final_clusters_number) ) ) )  # 展示log2(实际聚类数)个，最少要展示1个，最多展示10个
    
    # 对轮廓系数从大到小排序，展示前head_number个
    bset_cluster_number_DataFrame = Silhouette_df.sort_values(by='y', ascending=False).head(head_number)
    
    """
    from kneed import KneeLocator
    自动找拐点，在聚类数大了后效果不好，不再使用
    kl = KneeLocator(k_range, davies_bouldin_scores_list, curve="convex", direction="decreasing")
    kl.plot_knee()
    print( round(kl.elbow, 3) )
    """
    
    print("聚类分析结束")
    
    # 绘制肘部曲线
    return Silhouette_LinePlot, Davies_LinePlot, gr.update(value=bset_cluster_number_DataFrame,visible=True)


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

def confirm_cluster(
        process_clusters_method:int,
        global_dict_State: dict
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
    print(f"选择了{process_clusters_method_choose_str}")
    
    # 带时间戳的重命名原图片和附带文件
    if process_clusters_method == 0:
        assert process_clusters_method_choose_str == "重命名原图片(不推荐)"
        operation = "rename"
    # 在Cluster文件夹下生成聚类副本
    elif process_clusters_method == 1:
        assert process_clusters_method_choose_str == f"在{cluster_dir_prefix}文件夹下生成聚类副本(推荐)"
        operation = "copy"
    # 移动原图至Cluster文件夹
    elif process_clusters_method == 2:
        assert process_clusters_method_choose_str == f"移动原图至{cluster_dir_prefix}文件夹(大数据集推荐)"
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
