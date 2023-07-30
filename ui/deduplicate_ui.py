import gradio as gr
import torch

from ui.deduplicate_fn import (
    find_duplicates_images,
    get_choose_image_index,
    confirm,
    cancel,
    confirm_cluster,
    auto_select,
    all_select,
    CLUSTER_DIR_PREFIX,
    PROCESS_CLUSTERS_METHOD_CHOICES,
    IMAGEDEDUP_MODE_CHOICES_LIST,
    imagededup_mode_choose_trigger,
    HASH_METHODS_CHOICES_LIST,
    CNN_METHODS_CHOICES_LIST,
    release_torch_memory,
)
from tag_images_by_wd14_tagger import (
    DEFAULT_TAGGER_CAPTION_EXTENSION,  # 默认打标文件的扩展名
    WD14_NPZ_EXTENSION,  # 用于保存推理所得特征向量的文件扩展名
    WD14_TAGS_TOML_FILE,  # 存储各列向量对应的tag的文件的名字
)


##############################  常量  ##############################

NEED_OPERATED_EXTRA_FILE_EXTENSION_STR = ", ".join( [DEFAULT_TAGGER_CAPTION_EXTENSION, WD14_NPZ_EXTENSION] )

CUDA_IS_AVAILABLE = torch.cuda.is_available()

css = """
.attention {color: red  !important}
.recommendation {color: dodgerblue !important}
"""
blocks_name = "Deduplicate"


##############################  Markdown  ##############################

process_clusters_tips_markdown = f"""
### 处理方式
- 更名：在原文件夹将选中的图片加上{CLUSTER_DIR_PREFIX}前缀
  - 推荐全选，然后在资源管理器中按文件名排序，即可将同一聚类的图片放在一起，自行操作去重
- 移动：将选中的图片移动至{CLUSTER_DIR_PREFIX}子文件夹
  - 可以自动选择，或手动选中不想要的图片，然后移动，相比删除的优点是有备份
  - 也可以全部选择，然后在资源管理器中进入{CLUSTER_DIR_PREFIX}子文件夹，自行操作去重
- 删除：将选中的图片删除
  - 可以自动选择，或手动选中不想要的图片，然后删除，无备份

### 自动选择
自动选择是一种启发式算法，可以在去重的同时尽可能保留不重复的图片

"""


############################## Blocks ##############################

# 函数式创建有助于刷新ui界面
def create_ui() -> gr.Blocks:

    with gr.Blocks(title=blocks_name, css=css) as deduplicate_ui:

        with gr.Accordion("操作Tips", open=False):
            gr.Markdown(process_clusters_tips_markdown)

        with gr.Accordion("预处理", open=True):
            with gr.Row():
                images_dir = gr.Textbox(label="图片目录")
            with gr.Row():
                with gr.Box():
                    # TODO: 最好用demo.load来改写它
                    # 提醒我1是否仍然指的是CNN
                    assert IMAGEDEDUP_MODE_CHOICES_LIST[1] == "CNN - recommend on GPU"
                    imagededup_mode_choose_Dropdown = gr.Dropdown(
                        label="查重模式选择",
                        choices=IMAGEDEDUP_MODE_CHOICES_LIST,
                        value=IMAGEDEDUP_MODE_CHOICES_LIST[1 if CUDA_IS_AVAILABLE else 0],  # 如果有CUDA就默认显示CNN界面
                        type="index",
                    )
                with gr.Box() as hash_Box:
                    hash_methods_choose = gr.Dropdown(
                        label="Hash方法选择",
                        choices=HASH_METHODS_CHOICES_LIST,
                        value=HASH_METHODS_CHOICES_LIST[0],
                        type="index",
                    )
                    max_distance_threshold = gr.Slider(
                        minimum=0,
                        maximum=64,
                        value=10,
                        step=1,
                        label="max_distance_threshold",
                        info="越小越严格，越大耗时越长",
                    )
                with gr.Box() as cnn_Box:
                    cnn_methods_choose = gr.Dropdown(
                        label="CNN模型选择",
                        choices=CNN_METHODS_CHOICES_LIST,
                        value=CNN_METHODS_CHOICES_LIST[0],
                        type="index",
                    )
                    min_similarity_threshold = gr.Slider(
                        minimum=-0.99,
                        maximum=0.99,
                        step=0.01,
                        value=0.9,
                        label="min_similarity_threshold",
                        info="越大越严格",
                    )
        
        with gr.Row():
            use_cache = gr.Checkbox(label="使用缓存")
            release_torch_memory_button = gr.Button("释放Torch显存", elem_classes="recommendation")
            find_duplicates_images_button = gr.Button("扫描重复图片", variant="primary")

        with gr.Box():
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        duplicates_images_gallery = gr.Gallery(label="重复图片", value=[]).style(columns=6, height="auto", preview=False)
                    with gr.Row():
                        confirm_button = gr.Button("选择图片")
                        cancel_button = gr.Button("取消图片")
                    with gr.Row():
                        image_info_json = gr.JSON()
                with gr.Column(scale=1):
                    with gr.Accordion("操作（打开以进行自动选择、删除等操作）", open=False):
                        confirm_cluster_button = gr.Button("确定处理", elem_classes="attention")
                        process_clusters_method = gr.Radio(
                            label="图片处理方式（只对选中图片有效！）",
                            value=PROCESS_CLUSTERS_METHOD_CHOICES[1],
                            choices=PROCESS_CLUSTERS_METHOD_CHOICES,
                            type="index",
                        )
                        with gr.Row():
                            need_operated_extra_file_extension_Textbox = gr.Textbox(
                                label="同时处理的同名文件扩展名",
                                value=NEED_OPERATED_EXTRA_FILE_EXTENSION_STR,
                                placeholder=NEED_OPERATED_EXTRA_FILE_EXTENSION_STR,
                            )
                            need_operated_extra_file_name_Textbox = gr.Textbox(
                                label="同时处理的文件的全名",
                                value=WD14_TAGS_TOML_FILE,
                                placeholder=WD14_TAGS_TOML_FILE,
                            )
                    auto_select_button = gr.Button("自动选择", elem_classes="recommendation")
                    all_select_button = gr.Button("全部选择")
                    selected_images_str = gr.Textbox(label="待操作列表（手动编辑时请保证toml格式的正确）")


        ############################## 绑定函数 ##############################

        # 判断展示Hash还是CNN的Box组件
        imagededup_mode_choose_Dropdown_trigger_kwargs = dict(
            fn=imagededup_mode_choose_trigger,
            inputs=[imagededup_mode_choose_Dropdown],
            outputs=[hash_Box, cnn_Box],
        )
        imagededup_mode_choose_Dropdown.change(
            **imagededup_mode_choose_Dropdown_trigger_kwargs,  # type: ignore
        )
        deduplicate_ui.load(
            **imagededup_mode_choose_Dropdown_trigger_kwargs,  # type: ignore
        )

        # 释放torch显存
        release_torch_memory_button.click(
            fn=release_torch_memory,
            inputs=[],
            outputs=[],
        )

        # 按下后，在指定的目录搜索重复图像，并返回带标签的重复图像路径
        find_duplicates_images_button.click(
            fn=find_duplicates_images,
            inputs=[
                images_dir,
                use_cache,
                imagededup_mode_choose_Dropdown,
                hash_methods_choose,
                max_distance_threshold,
                cnn_methods_choose,
                min_similarity_threshold,
            ],
            outputs=[duplicates_images_gallery, selected_images_str],
        )

        # 点击一个图片后，记录该图片标签于全局变量choose_image_index，并且把按钮更名为该标签;同时显示该图片分辨率等信息
        duplicates_images_gallery.select(
            fn=get_choose_image_index,
            inputs=[selected_images_str],
            outputs=[confirm_button, cancel_button, image_info_json],
        )

        # 按下后，将全局变量choose_image_index中的值加到列表中,同时将取消按钮变红
        confirm_button.click(
            fn=confirm,
            inputs=[selected_images_str],
            outputs=[selected_images_str, confirm_button, cancel_button],
        )

        # 按下后，将全局变量choose_image_index中的值从列表中删除,同时将选择按钮变红
        cancel_button.click(
            fn=cancel,
            inputs=[selected_images_str],
            outputs=[selected_images_str, confirm_button, cancel_button],
        )

        # 按下后，或移动、复制、重命名、删除指定的图像，并清空画廊
        confirm_cluster_button.click(
            fn=confirm_cluster,
            inputs=[
                selected_images_str,
                process_clusters_method,
                need_operated_extra_file_extension_Textbox,
                need_operated_extra_file_name_Textbox,
            ],
            outputs=[duplicates_images_gallery, selected_images_str],
        )

        # 按下后，用启发式算法自动找出建议删除的重复项
        auto_select_button.click(
            fn=auto_select,
            inputs=[],
            outputs=[selected_images_str],
        )

        # 按下后，选择全部图片
        all_select_button.click(
            fn=all_select,
            inputs=[],
            outputs=[selected_images_str]
        )

    return deduplicate_ui
