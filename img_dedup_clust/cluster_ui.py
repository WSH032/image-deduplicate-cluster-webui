from typing import List, Union

import gradio as gr

from img_dedup_clust.cluster_fn import(
    vectorizer,
    confirm_SVD,
    cluster_images,
    cluster_analyse,
    confirm_cluster,
    MAX_GALLERY_NUMBER,
    FEATURE_EXTRACTION_METHOD_LIST,
    feature_extraction_method_change_trigger,
    wd14_feature_layer_choice_change_trigger,
    DEFAULT_TAGGER_CAPTION_EXTENSION,  # 默认打标文件的扩展名
    WD14_NPZ_EXTENSION,  # 用于保存推理所得特征向量的文件扩展名 # .wd14用来区分kohya的潜变量cache
    WD14_FEATURE_LAYER_CHOICE_LIST,
    TEXT_VECTORIZATION_METHOD_LIST,
    PROCESS_CLUSTERS_METHOD_CHOICES,
    CLUSTER_MODEL_LIST,
    WD14_TAGS_CATEGORY_LIST,
)


##############################  常量  ##############################

WD14_TAGS_TOML_HUGGINGFACE_URL = "https://huggingface.co/WSH032/wd-v1-4-tagger-feature-extractor/blob/main/wd14_tags.toml"

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

            vectorize_X_and_label_State = gr.State(value=None)  # 用于存放特征向量，和其对应的tag
            cluster_model_State = gr.State(None)  # 用于存放预处理中生成的聚类模型

            preprocess_Markdown = gr.Markdown("**请先进行预处理再聚类**")
            with gr.Row():
                images_dir = gr.Textbox(label="图片目录")
            with gr.Row():
                with gr.Box():
                    with gr.Column():
                        feature_extraction_method_Radio = gr.Radio(
                            choices=FEATURE_EXTRACTION_METHOD_LIST,
                            label="聚类特征选择",
                            value=FEATURE_EXTRACTION_METHOD_LIST[0],
                            type="index",
                            interactive=True
                        )
                        cluster_model = gr.Dropdown(
                            CLUSTER_MODEL_LIST,
                            label="聚类模型",
                            value=CLUSTER_MODEL_LIST[0],
                            type="index"
                        )
                        vectorizer_button = gr.Button("确认预处理", variant="primary")

                with gr.Box() as text_feature_extraction_Box:
                    gr.Markdown("**请用 '聚类特征选择' 切换此Box**")
                    text_vectorizer_method = gr.Radio(
                        choices=TEXT_VECTORIZATION_METHOD_LIST,
                        label="tags文本特征提取方法",
                        value=TEXT_VECTORIZATION_METHOD_LIST[0],
                        type="index"
                    )
                    text_feature_file_extension_name = gr.Textbox(label="tag文本文件扩展名", value=DEFAULT_TAGGER_CAPTION_EXTENSION, placeholder=DEFAULT_TAGGER_CAPTION_EXTENSION)
                    with gr.Row():
                        use_comma_tokenizer = gr.Checkbox(label="强制逗号分词", value=True, info="启用后则以逗号划分各个tag。不启用则同时以空格和逗号划分")
                        use_binary_tokenizer = gr.Checkbox(label="tag频率二值化", value=True, info="只考虑是否tag出现而不考虑出现次数")

                with gr.Box() as image_feature_extraction_Box:
                    gr.Markdown("**请用 '聚类特征选择' 切换此Box**")
                    wd14_feature_layer_choice = gr.Radio(
                        choices=WD14_FEATURE_LAYER_CHOICE_LIST,
                        label="wd14特征层选择",
                        value=WD14_FEATURE_LAYER_CHOICE_LIST[1],
                        type="index"
                    )
                    wd14_feature_file_extension_name = gr.Textbox(label="wd14特征向量文件扩展名", value=WD14_NPZ_EXTENSION, placeholder=WD14_NPZ_EXTENSION)
                    # 提醒我0仍然是全向量层
                    assert WD14_FEATURE_LAYER_CHOICE_LIST[0] == "predictions_sigmoid 全向量层"
                    assert WD14_TAGS_CATEGORY_LIST[0] == "rating"
                    tags_category_choices_CheckboxGroup = gr.CheckboxGroup(
                        choices=WD14_TAGS_CATEGORY_LIST,
                        value=WD14_TAGS_CATEGORY_LIST[1:],  # 默认不选第一个rating
                        label=f"用于聚类的标签种类（仅在'{WD14_FEATURE_LAYER_CHOICE_LIST[0]}'可选）",
                        type="index",
                        info=f"您可以在[{WD14_TAGS_TOML_HUGGINGFACE_URL}]查看各个标签种类所包含的tags",
                    )

        with gr.Box():
            confirm_SVD_Markdown = gr.Markdown("**SVD降维（可选，建议特征数大于50时使用降维）**")
            with gr.Row():
                SVD_n_components = gr.Slider(minimum=1, step=1, label="TruncatedSVD n_components", info="SVD降维数", interactive=False)
                confirm_SVD_button = gr.Button("请先预处理在降维", elem_classes="recommendation", interactive=False)

        with gr.Box():
            with gr.Row():
                with gr.Accordion("聚类效果分析", open=True):
                    gr.Markdown("**建议多次运行分析，取最大值作为最终聚类数**")
                    with gr.Row():
                        max_cluster_number = gr.Slider(2, MAX_GALLERY_NUMBER, step=1, value=10, label="分析时最大聚类数 / OPTICS-min_samples")
                        cluster_analyse_button = gr.Button(value="请先预处理再分析", interactive=False)
                    with gr.Row():
                        # 这两个要显示出来，这样第一次才能看到进度条
                        Silhouette_gr_Plot = gr.LinePlot(visible=True)
                        Davies_gr_Plot = gr.LinePlot(visible=True)
                    with gr.Row():
                        bset_cluster_number_DataFrame = gr.DataFrame(visible=False)
                        samples_ScatterPlot = gr.ScatterPlot(visible=False)
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
                        value=PROCESS_CLUSTERS_METHOD_CHOICES[1],
                        choices=PROCESS_CLUSTERS_METHOD_CHOICES,
                        type="index",
                    )
                    confirm_cluster_button = gr.Button(value="确认聚类", elem_classes="attention")
                gr_Accordion_and_Gallery_list = create_gr_gallery(MAX_GALLERY_NUMBER)


        ############################## 绑定函数 ##############################

        # 聚类特征选择
        feature_extraction_method_Radio_trigger_kwargs = dict(
            fn=feature_extraction_method_change_trigger,
            inputs=[feature_extraction_method_Radio],
            outputs=[
                text_feature_extraction_Box,
                image_feature_extraction_Box
            ],
        )
        feature_extraction_method_Radio.change(
            **feature_extraction_method_Radio_trigger_kwargs,  # type: ignore
        )
        cluster_ui.load(
            **feature_extraction_method_Radio_trigger_kwargs,  # type: ignore
        )

        # 通过选择的向量层判断是否显示部分特征聚类功能
        wd14_feature_layer_choice_change_trigger_kwargs = dict(
            fn=wd14_feature_layer_choice_change_trigger,
            inputs=[wd14_feature_layer_choice],
            outputs=[tags_category_choices_CheckboxGroup],
        )
        wd14_feature_layer_choice.change(
            **wd14_feature_layer_choice_change_trigger_kwargs,  # type: ignore
        )
        cluster_ui.load(
            **wd14_feature_layer_choice_change_trigger_kwargs,  # type: ignore
        )

        # 特征提取与模型选择
        vectorizer_button.click(
            fn=vectorizer,
            inputs=[
                images_dir,
                feature_extraction_method_Radio,
                # 文本聚类
                text_vectorizer_method,
                use_comma_tokenizer,
                use_binary_tokenizer,
                text_feature_file_extension_name,
                # wd14特征聚类
                wd14_feature_layer_choice,
                wd14_feature_file_extension_name,
                tags_category_choices_CheckboxGroup,
                # 聚类方法
                cluster_model,
            ],
            outputs=[
                vectorize_X_and_label_State,
                cluster_model_State,
                preprocess_Markdown,
                cluster_analyse_button,
                cluster_images_button,
                SVD_n_components,
                confirm_SVD_button,
            ]
        )

        # SVD降维
        confirm_SVD_button.click(
            fn=confirm_SVD,
            inputs=[
                SVD_n_components,
                vectorize_X_and_label_State,
            ],
            outputs=[
                confirm_SVD_Markdown,
                vectorize_X_and_label_State,
                confirm_SVD_button,
            ],
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
                bset_cluster_number_DataFrame,
                samples_ScatterPlot,  # 样本散点图
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