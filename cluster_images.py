# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:38:31 2023

@author: WSH
"""

# 导入所需的库
import sklearn.cluster as skc
import sklearn.feature_extraction.text as skt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif, f_classif, SelectPercentile, SelectKBest
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import os 
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from scipy.cluster.hierarchy import linkage, dendrogram
from SearchImagesTags import SearchImagesTags
from tqdm import tqdm
from PIL import Image
from typing import Callable, List
import pandas as pd
from datetime import datetime
import shutil
import math
import logging



MAX_GALLERY_NUMBER = 100  # 画廊里展示的最大聚类数量为100
CACHE_RESOLUTION = 256  # 缓存图片时最大分辨率


# 逗号分词器
def comma_tokenizer(text: str) -> List[str]:
    """
    定义一个以逗号为分隔符的分词器函数，并对每个标签进行去空格操作
    如输入"xixi, haha"
    返回["xixi", "haha"]
    """
    return [tag.strip() for tag in text.split(',')]


def vectorizer(images_dir: str,
               vectorizer_method: int,
               use_comma_tokenizer: bool,
               use_binary_tokenizer: bool,
               cluster_model: int
):
    """
    读取images_dir下的图片或tags文本，用其生成特征向量
    
    images_dir为图片路径
    vectorizer_method为特征提取方法
        0 使用tfid
        1 使用Count
        2 使用WD14
    use_comma_tokenizer对于前两种方法使用逗号分词器comma_tokenizer
    use_binary_tokenizer对于前两种方法使用binary参数
    cluster_model为聚类模型
        0，1，2...等分别对应不同的聚类模型
    
    return vectorize_X_and_label_State=[X, tf_tags_list]
    """
    
    # 选择特征提取器
    vectorizer_args_dict = dict(tokenizer=comma_tokenizer if use_comma_tokenizer else None,
                                binary=use_binary_tokenizer,
                                max_df=0.99)
    if vectorizer_method == 0 :
        tfvec = skt.TfidfVectorizer(**vectorizer_args_dict)
    elif vectorizer_method == 1 :
        tfvec = skt.CountVectorizer(**vectorizer_args_dict)
    
    
    # 实例化搜索器
    searcher = SearchImagesTags(images_dir, tag_file_ext=".txt")
    # 读取图片名字， 用WD14
    image_files_list = searcher.image_files()
    # tag内容, 用于文本提取
    tag_content_list = searcher.tag_content(error_then_tag_is="_no_tag")
    
    
    # tags转为向量特征
    X = tfvec.fit_transform(tag_content_list).toarray()  # 向量特征
    tf_tags_list = tfvec.get_feature_names_out()  # 向量每列对应的tag
    # 被过滤的tag
    # stop_tags = tfvec.stop_words_
    

    """
    已经弃用尝试继承类
    parent_class_list = [skc.KMeans, skc.SpectralClustering, skc.AgglomerativeClustering, skc.OPTICS]
    
    class ClusterModel(object):
        
        def __init__(self, model_index: int=0):
            # cluster_model_list = ["K-Means聚类", "Spectral谱聚类", "Agglomerative层次聚类", "OPTICS聚类"]
            def _get_modl(model_index):
                if model_index == 0 :
                    cluster_model_State = skc.KMeans()
                elif model_index == 1 :
                    cluster_model_State = skc.SpectralClustering( affinity='cosine' )
                elif model_index == 2 :
                    cluster_model_State = skc.AgglomerativeClustering( affinity='cosine', linkage='average' )
                elif model_index == 3 :
                    cluster_model_State = skc.OPTICS( metric="cosine")
                return cluster_model_State
            
            self.cluster_model_State = _get_modl(model_index)  # 存放模型
            self.model_index = model_index  # 存放模型序号，用于确认存放的是哪一种模型，方便设置n值
        
        def set_n(self, n: int=2):
            # 如果不为3，即不为OPTICS，则其他模型都可以直接用n_clusters参数
            if not self.model_index == 3 :
                self.cluster_model_State.set_params(n_clusters=n)
            # 如果为3，即OPTICS，要使用min_samples参数
            else:
                self.cluster_model_State.set_params(min_samples=n)
            return self
    """
    
    
    def _get_modl(model_index):
        if model_index == 0 :
            cluster_model_State = skc.KMeans()
        elif model_index == 1 :
            cluster_model_State = skc.SpectralClustering( affinity='cosine' )
        elif model_index == 2 :
            cluster_model_State = skc.AgglomerativeClustering( affinity='cosine', linkage='average' )
        elif model_index == 3 :
            cluster_model_State = skc.OPTICS( metric="cosine")
        return cluster_model_State
    cluster_model_State = _get_modl(cluster_model)
    
    return [X, tf_tags_list], cluster_model_State, "预处理完成"
    


def cluster_images(images_dir: str,
                   confirmed_cluster_number: int,
                   use_cache: bool,
                   global_dict_State: dict,
                   vectorize_X_and_label_State: list,
                   cluster_model_State,
) -> list:
    """
    对指定目录下的图片进行聚类，将会使用该目录下与图片同名的txt中的tag做为特征
    confirmed_cluster_number为指定的聚类数
    如果use_cache为真，则会把图片拷贝到指定目录下的cache文件夹，并显示缩略图
    """
    
    # 实例化搜索器
    searcher = SearchImagesTags(images_dir, tag_file_ext=".txt")
    # 读取图片名字，tag内容
    image_files_list = searcher.image_files()
    tag_content_list = searcher.tag_content(error_then_tag_is="_no_tag")
    images_and_tags_tuple = tuple( zip( image_files_list, tag_content_list ) )
    
    
    # 获取文件名列表和tags列表
    images_files_list = [image for image, _ in images_and_tags_tuple]
    tags_list = [tags for _, tags in images_and_tags_tuple]
    
    # tags转为向量特征
    X = vectorize_X_and_label_State[0]  # 向量特征
    tf_tags_list = vectorize_X_and_label_State[1]  # 向量每列对应的tag

    
    """ todo SVD降维
    SVD = TruncatedSVD( n_components = X.shape[1] - 1 )
    X_SVD = SVD.fit_transform(X)
    X_SVD = normalize(X_SVD)
    """
    
    # 聚类，最大聚类数不能超过样本数
    
    n_clusters = min( confirmed_cluster_number, len(tags_list) )
    cluster_model = cluster_model_State
    
    # 根据模型的不同设置不同的参数n
    if "n_clusters" in cluster_model.get_params().keys():
        cluster_model.set_params(n_clusters=n_clusters)
    elif "min_samples" in cluster_model.get_params().keys():
        cluster_model.set_params(min_samples=n_clusters)
    else:
        logging.error("选择的模型中，n参数指定出现问题")
    
    print("聚类算法", type(cluster_model).__name__)
    
    y_pred = cluster_model.fit_predict(X) # 训练模型并得到聚类结果
    print(y_pred)
    # centers = kmeans_model.cluster_centers_  #kmeans模型的聚类中心，用于调试和pandas计算，暂不启用
    
    
    """
    特征重要性分析，计算量太大，不启用
    t1 = datetime.now()
    permutation_importance(cluster_model, X, y_pred, n_jobs=-1)
    t2 = datetime.now()
    print(t2-t1)
    """
    
    # 分类的ID
    clusters_ID = np.unique(y_pred)
    clustered_images_list = [np.compress(np.equal(y_pred, i), images_files_list).tolist() for i in clusters_ID]
    
    # #################################
    # pandas处理数据
    
    """
    弃用，原本打算基于kmeans的距离分析，现在更换为chi2算法
    all_center = pd.Series( np.mean(X, axis=0), tf_tags_list )  # 全部样本的中心
    """

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
        logging.error(f"pred_df的标签部分包含了非法值 error: {e}")


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
        
        # 评分器： chi2, mutual_info_regression, f_classif
        # 选择器： SelectPercentile, SelectKBest
        
        cluster_feature_tags_list = []

        for i in clusters_ID:
            # 将第i类和其余的类二分类
            temp_pred = y_pred.copy()
            temp_pred[ temp_pred != i ] = i+1
            
            """
            # 依据特征tags的数量分段决定提取特征的数量
            # 第一段斜率0.5，第二段0.3，第三段为log2(x-50)
            def cul_k_by_tags_len(tags_len):
                if tags_len < 10:
                    return max( 1, 0.5*tags_len )
                if tags_len <60:
                    return 5 + 0.3*(tags_len-10)
                if True:
                    return 20 - math.log2(60-50) + math.log2(tags_len-50)
            k = round( cul_k_by_tags_len(X.shape[1]) )
            """
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
            cluster_feature_tags_list.append( {"prompt":prompt_tags_list, "negetive":negetive_tags_list} )
            
        return cluster_feature_tags_list

    cluster_feature_tags_list = _cluster_feature_select(clusters_ID, X, y_pred, pred_df)
    

    # 赋值到全局组件中，将会传递至confirm_cluster_button.click
    global_dict_State["common_duplicate_tags_set"] = common_duplicate_tags_set
    global_dict_State["cluster_feature_tags_list"] = cluster_feature_tags_list
    global_dict_State["clustered_images_list"] = clustered_images_list
    global_dict_State["images_dir"] = images_dir
    
    def cache_image(input_dir: str, cluster_list: list, resolution: int=512 ):
        """ 如果使用缓存，就调用pillow，将重复的图片缓存到同路径下的一个cache文件夹中，分辨率最大为resolution,与前面图片名字一一对应 """
        
        # 建一个文件夹
        cache_dir = os.path.join(input_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        print("缓存缩略图中，caching...")
        for cluster in tqdm(cluster_list):
            for image_name in cluster:
                # 已经存在同名缓存文件，就不缓存了
                if not os.path.exists( os.path.join(cache_dir, image_name) ):
                    try:
                        with Image.open( os.path.join( input_dir, image_name ) ) as im:
                            im.thumbnail( (resolution, resolution) )
                            im.save( os.path.join(cache_dir, image_name) )
                    except Exception as e:
                        logging.error(f"缓存 {image_name} 失败, error: {e}")
        print(f"缓存完成: {cache_dir}\nDone!")
        
    if use_cache:
        cache_image(images_dir, clustered_images_list, resolution=CACHE_RESOLUTION)
     
    # 如果用缓存就展示缓存图
    gallery_images_dir = images_dir if not use_cache else os.path.join( images_dir, "cache" )
    
    # 注意，返回的列表长度为显示gallery组件数的两倍
    # 因为列表里偶数位为Accordion组件，奇数位为Gallery组件
    visible_gr_gallery_list = []
    # 最多只能处理MAX_GALLERY_NUMBER个画廊
    for i in range( min( len(clustered_images_list), MAX_GALLERY_NUMBER ) ):
        gallery_images_tuple_list = [ (os.path.join(gallery_images_dir,name), name) for name in clustered_images_list[i] ]
        prompt = cluster_feature_tags_list[i].get("prompt", [])
        negetive = cluster_feature_tags_list[i].get("negetive", [])
        visible_gr_gallery_list.extend( [gr.update( visible=True, label=f"聚类{i} :\nprompt: {prompt}\nnegetive: {negetive}" ),
                                         gr.update( value=gallery_images_tuple_list, visible=True)
                                        ] 
        )
        
    unvisible_gr_gallery_list = [ gr.update( visible=False ) for i in range( 2*( MAX_GALLERY_NUMBER-len(clustered_images_list) ) ) ]
    
    return visible_gr_gallery_list + unvisible_gr_gallery_list + [gr.update(visible=True)] + [global_dict_State]


def cluster_analyse(images_dir: str, max_cluster_number: int, vectorize_X_and_label_State:list, cluster_model_State):
    """
    读取指定路径下的图片，并依据与图片同名的txt内的tags进行聚类
    将评估从聚类数从 2~max_cluster_number 的效果
    返回matplotlib类型的肘部曲线和轮廓系数
    """
    
    # 实例化搜索器
    searcher = SearchImagesTags(images_dir, tag_file_ext=".txt")
    # 读取图片名字，tag内容
    image_files_list = searcher.image_files()
    tag_content_list = searcher.tag_content(error_then_tag_is="_no_tag")
    images_and_tags_tuple = tuple( zip( image_files_list, tag_content_list ) )
    
    tags_list = [tags for _, tags in images_and_tags_tuple]
    
    # 提取特征标签
    X = vectorize_X_and_label_State[0]  # 向量特征
    
    # 使用肘部法则和轮廓系数确定最优的聚类数
    davies_bouldin_scores_list = [] # 存储每个聚类数对应的davies系数
    silhouette_scores_list = []  # 用于存储不同k值对应的轮廓系数
    
    # 最大聚类数不能超过样本,最多只能样本数-1
    k_range = range( 2, min(max_cluster_number+1, len(tags_list) ) ) # 聚类数的范围(左闭右开)
    
    print("聚类分析开始")
    for k in tqdm(k_range):
        cluster_model = cluster_model_State # 获取模型
        
        # 根据模型的不同设置不同的参数n
        if "n_clusters" in cluster_model.get_params().keys():
            cluster_model.set_params(n_clusters=k)
        elif "min_samples" in cluster_model.get_params().keys():
            cluster_model.set_params(min_samples=k)
        else:
            logging.error("选择的模型中，n参数指定出现问题")
        
        y_pred = cluster_model.fit_predict(X) # 训练模型
        
        # 如果出现只有一个聚类的情况，或者每一个图片都自成一类，就不用继续分析了
        if len( np.unique(y_pred) ) == 1 or len( np.unique(y_pred) ) == len(X):
            break
          
        sil_score = silhouette_score(X, y_pred)  # 计算轮廓系数,聚类数为1时无法计算
        silhouette_scores_list.append(sil_score) # 储存轮廓系数
        
        dav_score= davies_bouldin_score(X,y_pred)  # 计算davies系数
        davies_bouldin_scores_list.append(dav_score) # 储存
    
    print("分析算法", type(cluster_model).__name__)
    
    Silhouette_df = pd.DataFrame( {"x":k_range[0:len(silhouette_scores_list)], "y":silhouette_scores_list} )
    # 注意，这里Davies_df的y值做了一次非线性映射，0 -> 1 ; +inf -> -1
    # Davies_df = pd.DataFrame( {"x":k_range[0:len(davies_bouldin_scores_list)], "y":(1 - 4*np.arctan(davies_bouldin_scores_list) / np.pi) } )
    Davies_df = pd.DataFrame( {"x":k_range[0:len(davies_bouldin_scores_list)], "y":( -1 * np.array(davies_bouldin_scores_list) ) } )
    
    print(davies_bouldin_scores_list)
    print(Davies_df.loc[:,"y"])
    
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
    
    
    final_clusters_number = len( np.unique(y_pred) )  # 实际最大聚类数
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


def create_gr_gallery(max_gallery_number: int) -> list:
    """
    根据指定的最大数，创建相应数量的带Accordion的Gallery组件
    返回一个列表，长度为2*max_gallery_number
    偶数位为Accordion组件，奇数位为Gallery组件
    """
    gr_Accordion_and_Gallery_list = []
    for i in range(max_gallery_number):
        with gr.Accordion(f"聚类{i}", open=True, visible=False) as Gallery_Accordion:
            gr_Accordion_and_Gallery_list.extend( [ Gallery_Accordion, gr.Gallery(value=[]).style(columns=[6], height="auto") ] )
    return gr_Accordion_and_Gallery_list


def confirm_cluster(process_clusters_method:int, global_dict_State: dict):
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
    
    
    def change_ext_to_txt(path: str) -> str:
        # 将一个路径或者文件的扩展名改成.txt
        path_and_name, ext = os.path.splitext(path)
        new_path = path_and_name + ".txt"
        return new_path
    def change_name_with_ext(path: str, new_name: str) -> str:
        # 保留一个路径或者文件的扩展名，更改其文件名
        path_and_name, ext = os.path.splitext(path)
        new_path = os.path.join( os.path.dirname(path_and_name), new_name+ext )
        return new_path
    
    # 获取当前时间
    time_now = datetime.now().strftime('%Y%m%d%H%M%S')
    
    
    def rename_images(images_dir: str, clustered_images_list: list):
        """ 依据clustered_images_list中聚类情况，对images_dir下图片以及同名txt文件重命名 """

        print("重命名原图中，renaming...")
        for cluster_index, cluster in tqdm( enumerate(clustered_images_list) ):
            # 重命名
            for image_index, image_name in enumerate(cluster):
                # 重命名图片
                new_image_name = change_name_with_ext(image_name, f"cluster{cluster_index}-{image_index:06d}-{time_now}")
                try:
                    os.rename( os.path.join(images_dir, image_name), os.path.join(images_dir, new_image_name) )
                except Exception as e:
                    logging.error(f"重命名 {image_name} 失败, error: {e}")
                # 重命名txt
                txt_name = change_ext_to_txt(image_name)
                new_txt_name = change_name_with_ext(txt_name, f"cluster{cluster_index}-{image_index:06d}-{time_now}")
                try:
                    os.rename( os.path.join(images_dir, txt_name), os.path.join(images_dir, new_txt_name) )
                except Exception as e:
                    logging.error(f"重命名 {txt_name} 失败, error: {e}")
        print("重命名完成  Done!")
        
    if process_clusters_method == 0:
        rename_images(images_dir, clustered_images_list)
    
    def copy_or_move_images(images_dir: str, clustered_images_list: list, move=False):
        """
        依据clustered_images_list中聚类情况，将images_dir下图片以及同名txt拷贝或移动至Cluster文件夹
        move=True时为移动
        """
        
        Cluster_folder_dir = os.path.join(images_dir, f"Cluster-{time_now}")
        
        process_func = shutil.move if move else shutil.copy2
        
        # 清空聚类文件夹
        if os.path.exists(Cluster_folder_dir):
            shutil.rmtree(Cluster_folder_dir)
        os.makedirs(Cluster_folder_dir, exist_ok=True)
        
        print("拷贝聚类中，coping...")
        for cluster_index, cluster in tqdm( enumerate(clustered_images_list) ):
            Cluster_son_folder_dir = os.path.join(Cluster_folder_dir, f"cluster-{cluster_index}")
            os.makedirs(Cluster_son_folder_dir, exist_ok=True)
            
            # 拷贝
            for image_name in cluster:
                # 拷贝或移动图片
                try:
                    process_func( os.path.join(images_dir, image_name), os.path.join(Cluster_son_folder_dir, image_name) )
                except Exception as e:
                    logging.error(f"拷贝或移动 {image_name} 失败, error: {e}")
                # 拷贝或移动txtx
                txt_name = change_ext_to_txt(image_name)
                try:
                    process_func( os.path.join(images_dir, txt_name), os.path.join(Cluster_son_folder_dir, txt_name) )
                except Exception as e:
                    logging.error(f"拷贝或移动 {txt_name} 失败, error: {e}")
        print(f"拷贝或移动完成: {Cluster_folder_dir}\nDone!")
        
    if process_clusters_method == 1:
        copy_or_move_images(images_dir, clustered_images_list, move=False)
    if process_clusters_method == 2:
        copy_or_move_images(images_dir, clustered_images_list, move=True)
    
    return gr.update(visible=False)
    



##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

css = """
.attention {color: red  !important}
"""

with gr.Blocks(css=css) as demo:
    
    global_dict_State = gr.State(value={})  # 这个将会起到全局变量的作用，类似于globals()
    """
    全局列表
    global_dict_State["common_duplicate_tags_set"] = common_duplicate_tags_set
    global_dict_State["cluster_feature_tags_list"] = cluster_feature_tags_list
    global_dict_State["clustered_images_list"] = clustered_images_list
    global_dict_State["images_dir"] = images_dir
    """
    
    with gr.Box():
        vectorize_X_and_label_State = gr.State(value=[])  # 用于存放特征向量，和其对应的tag
        cluster_model_State = gr.State()  # 用于存放预处理中生成的聚类模型
        preprocess_Markdown = gr.Markdown("**请先进行预处理再聚类**")
        with gr.Row():
            images_dir = gr.Textbox(label="图片目录")     
        with gr.Row():
            with gr.Column(scale=1):
                vectorizer_method_list = ["TfidfVectorizer", "CountVectorizer"]
                vectorizer_method = gr.Dropdown(vectorizer_method_list, label="特征提取", value=vectorizer_method_list[0], type="index")
            use_comma_tokenizer = gr.Checkbox(label="强制逗号分词", value=True, info="启用后则以逗号划分各个tag。不启用则同时以空格和逗号划分")
            use_binary_tokenizer = gr.Checkbox(label="tag频率二值化", value=True, info="只考虑是否tag出现而不考虑出现次数")
            vectorizer_button = gr.Button("确认预处理", variant="primary")
        with gr.Row():
                cluster_model_list = ["K-Means聚类", "Spectral谱聚类", "Agglomerative层次聚类", "OPTICS聚类"]
                cluster_model = gr.Dropdown(cluster_model_list, label="聚类模型", value=cluster_model_list[0], type="index")
    with gr.Box():
        with gr.Row():
            with gr.Accordion("聚类效果分析", open=True):
                with gr.Row():
                    max_cluster_number = gr.Slider(2, MAX_GALLERY_NUMBER, step=1, value=10, label="分析时最大聚类数 / OPTICS-min_samples")
                    cluster_analyse_button = gr.Button("开始分析")
                with gr.Row():
                    Silhouette_gr_Plot = gr.LinePlot()
                    Davies_gr_Plot = gr.LinePlot()
                with gr.Row():
                    bset_cluster_number_DataFrame = gr.DataFrame(value=[],
                                                                 label="根据轮廓曲线推荐的聚类数（y越大越好）",
                                                                 visible=False
                    )
    with gr.Box():
        with gr.Row():
            with gr.Column(scale=2):
                confirmed_cluster_number = gr.Slider(2, MAX_GALLERY_NUMBER, step=1, value=2, label="聚类数n_cluster / OPTICS-min_samples")
            with gr.Column(scale=1):
                use_cache = gr.Checkbox(label="使用缓存",info="如果cache目录内存在同名图片，则不会重新缓存(可能会造成图片显示不一致)")
            
            cluster_images_button = gr.Button("开始聚类并展示结果", variant="primary")
    with gr.Row():
        with gr.Accordion("聚类图片展示", open=True):
            with gr.Row(visible=False) as confirm_cluster_Row:
                process_clusters_method_choices = ["重命名原图片(不推荐)","在Cluster文件夹下生成聚类副本(推荐)","移动原图至Cluster文件夹(大数据集推荐)"]
                process_clusters_method = gr.Radio(label="图片处理方式",
                                                   value=process_clusters_method_choices[1],
                                                   choices=process_clusters_method_choices,
                                                   type="index",
                )
                confirm_cluster_button = gr.Button(value="确认聚类", elem_classes="attention")
            gr_Accordion_and_Gallery_list = create_gr_gallery(MAX_GALLERY_NUMBER)

    # 特征提取与模型选择
    vectorizer_button.click(fn=vectorizer,
                            inputs=[images_dir, vectorizer_method, use_comma_tokenizer, use_binary_tokenizer, cluster_model],
                            outputs=[vectorize_X_and_label_State, cluster_model_State, preprocess_Markdown]
    )
    # 聚类图像
    cluster_images_button.click(fn=cluster_images,
                                inputs=[images_dir,
                                        confirmed_cluster_number,
                                        use_cache, global_dict_State,
                                        vectorize_X_and_label_State,
                                        cluster_model_State],
                                outputs=gr_Accordion_and_Gallery_list + [confirm_cluster_Row] + [global_dict_State]
    )
    # 聚类分析
    cluster_analyse_button.click(fn=cluster_analyse,
                                 inputs=[images_dir,
                                         max_cluster_number,
                                         vectorize_X_and_label_State,
                                         cluster_model_State],
                                 outputs=[Silhouette_gr_Plot, Davies_gr_Plot, bset_cluster_number_DataFrame]
    )
    # 确定聚类
    confirm_cluster_button.click(fn=confirm_cluster,
                                 inputs=[process_clusters_method, global_dict_State],
                                 outputs=[confirm_cluster_Row],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True,debug=True)
