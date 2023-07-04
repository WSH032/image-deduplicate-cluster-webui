from typing import List, Union

import gradio as gr


from ui.cluster_fn import(
    vectorizer,
    cluster_images,
    cluster_analyse,
    confirm_cluster,
    MAX_GALLERY_NUMBER,
)
from ui.tools.operate_images import cluster_dir_prefix


##############################  常量  ##############################

# 请与ui.clustere_fn.confirm_cluster对应
process_clusters_method_choices = [
    "重命名原图片(不推荐)",
    f"在{cluster_dir_prefix}文件夹下生成聚类副本(推荐)",
    f"移动原图至{cluster_dir_prefix}文件夹(大数据集推荐)",
]

# 请与ui.clustere_fn相应函数对应
vectorizer_method_list = [
    "TfidfVectorizer",
    "CountVectorizer",
    "WD14",
]

# 请与ui.clustere_fn相应函数对应
cluster_model_list = [
    "K-Means聚类",
    "Spectral谱聚类",
    "Agglomerative层次聚类",
    "OPTICS聚类"
]

css = """
.attention {color: red  !important}
.recommendation {color: dodgerblue !important}
"""
blocks_name = "cluster"


############################## Blocks ##############################

def create_gr_gallery(max_gallery_number: int) -> List[Union[gr.Accordion, gr.Gallery]]:
    """
    根据指定的最大数，创建相应数量的带Accordion的Gallery组件

    返回一个列表，长度为 2 * max_gallery_number
    前一半为Accordion，后一半为Gallery
    """
    gr_Accordion_list = []
    gr_Gallery_list = []
    for i in range(max_gallery_number):
        with gr.Accordion(f"聚类{i}", open=True, visible=False) as Gallery_Accordion:
            Gallery = gr.Gallery(value=[]).style(columns=6, height="auto")

            gr_Accordion_list.append(Gallery_Accordion)
            gr_Gallery_list.append(Gallery)
    gr_Accordion_and_Gallery_list = gr_Accordion_list + gr_Gallery_list
    return gr_Accordion_and_Gallery_list


# 函数式创建有助于刷新ui界面
def create_ui() -> gr.Blocks:

    with gr.Blocks(title=blocks_name, css=css) as cluster_ui:

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
            cluster_model_State = gr.State(None)  # 用于存放预处理中生成的聚类模型
            preprocess_Markdown = gr.Markdown("**请先进行预处理再聚类**")
            with gr.Row():
                images_dir = gr.Textbox(label="图片目录")     
            with gr.Row():
                with gr.Column(scale=1):
                    vectorizer_method = gr.Dropdown(
                        vectorizer_method_list,
                        label="特征提取",
                        value=vectorizer_method_list[0],
                        type="index"
                    )
                use_comma_tokenizer = gr.Checkbox(label="强制逗号分词", value=True, info="启用后则以逗号划分各个tag。不启用则同时以空格和逗号划分")
                use_binary_tokenizer = gr.Checkbox(label="tag频率二值化", value=True, info="只考虑是否tag出现而不考虑出现次数")
                vectorizer_button = gr.Button("确认预处理", variant="primary")
            with gr.Row():
                    cluster_model = gr.Dropdown(
                        cluster_model_list,
                        label="聚类模型",
                        value=cluster_model_list[0],
                        type="index"
                    )
        with gr.Box():
            with gr.Row():
                with gr.Accordion("聚类效果分析", open=True):
                    gr.Markdown("**建议多次运行分析，取最大值作为最终聚类数**")
                    with gr.Row():
                        max_cluster_number = gr.Slider(2, MAX_GALLERY_NUMBER, step=1, value=10, label="分析时最大聚类数 / OPTICS-min_samples")
                        cluster_analyse_button = gr.Button(value="请先预处理再分析", interactive=False)
                    with gr.Row():
                        Silhouette_gr_Plot = gr.LinePlot()
                        Davies_gr_Plot = gr.LinePlot()
                    with gr.Row():
                        bset_cluster_number_DataFrame = gr.DataFrame(
                            value=None,
                            label="根据轮廓曲线推荐的聚类数（y越大越好）",
                            visible=False,
                        )
        with gr.Box():
            with gr.Row():
                with gr.Column(scale=2):
                    confirmed_cluster_number = gr.Slider(2, MAX_GALLERY_NUMBER, step=1, value=2, label="聚类数n_cluster / OPTICS-min_samples")
                with gr.Column(scale=1):
                    use_cache = gr.Checkbox(label="使用缓存",info="如果cache目录内存在同名图片，则不会重新缓存(可能会造成图片显示不一致)")
                
                cluster_images_button = gr.Button(value="请先预处理再聚类", interactive=False, variant="primary")
        with gr.Row():
            with gr.Accordion("聚类图片展示", open=True):
                with gr.Row(visible=False) as confirm_cluster_Row:
                    process_clusters_method = gr.Radio(
                        label="图片处理方式",
                        value=process_clusters_method_choices[1],
                        choices=process_clusters_method_choices,
                        type="index",
                    )
                    confirm_cluster_button = gr.Button(value="确认聚类", elem_classes="attention")
                gr_Accordion_and_Gallery_list = create_gr_gallery(MAX_GALLERY_NUMBER)


        ############################## 绑定函数 ##############################

        # 特征提取与模型选择
        vectorizer_button.click(
            fn=vectorizer,
            inputs=[
                images_dir,
                vectorizer_method,
                use_comma_tokenizer,
                use_binary_tokenizer,
                cluster_model
            ],
            outputs=[
                vectorize_X_and_label_State,
                cluster_model_State,
                preprocess_Markdown,
                cluster_analyse_button,
                cluster_images_button,
            ]
        )

        # 聚类图像
        cluster_images_button.click(
            fn = cluster_images,
            inputs = [
                images_dir,
                confirmed_cluster_number,
                use_cache, global_dict_State,
                vectorize_X_and_label_State,
                cluster_model_State,
            ],
            outputs = gr_Accordion_and_Gallery_list + [confirm_cluster_Row] + [global_dict_State] # type: ignore
        )

        # 聚类分析
        cluster_analyse_button.click(
            fn=cluster_analyse,
            inputs=[
                max_cluster_number,
                vectorize_X_and_label_State,
                cluster_model_State
            ],
            outputs=[
                Silhouette_gr_Plot,
                Davies_gr_Plot,
                bset_cluster_number_DataFrame
            ]
        )

        # 确定聚类
        confirm_cluster_button.click(
            fn=confirm_cluster,
            inputs=[
                process_clusters_method,
                global_dict_State
            ],
            outputs=[
                confirm_cluster_Row, # type: ignore
                cluster_analyse_button,
                cluster_images_button,
            ],
        )

    return cluster_ui