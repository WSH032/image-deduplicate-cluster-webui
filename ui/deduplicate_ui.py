import gradio as gr

from ui.deduplicate_fn import (
    find_duplicates_images,
    get_choose_image_index,
    confirm,
    cancel,
    confirm_cluster,
    auto_select,
    all_select,
)
from ui.tools.operate_images import cluster_dir_prefix


##############################  Markdown  ##############################

process_clusters_tips_markdown = f"""
### 处理方式
- 更名：在原文件夹将选中的图片加上{cluster_dir_prefix}前缀
  - 推荐全选，然后在资源管理器中按文件名排序，即可将同一聚类的图片放在一起，自行操作去重
- 移动：将选中的图片移动至{cluster_dir_prefix}子文件夹
  - 可以自动选择，或手动选中不想要的图片，然后移动，相比删除的优点是有备份
  - 也可以全部选择，然后在资源管理器中进入{cluster_dir_prefix}子文件夹，自行操作去重
- 删除：将选中的图片删除
  - 可以自动选择，或手动选中不想要的图片，然后删除，无备份

### 自动选择
自动选择是一种启发式算法，可以在去重的同时尽可能保留不重复的图片

"""


##############################  常量  ##############################

# 请与ui.dedupilicate_fn.confirm_cluster对应
process_clusters_method_choices = [
    "更名选中图片（推荐全部选择）",
    "移动选中图片（推荐此方式）",
    "删除选中图片（推荐自动选择）"
]

css = """
.attention {color: red  !important}
"""
blocks_name = "Deduplicate"


############################## Blocks ##############################

# 函数式创建有助于刷新ui界面
def create_ui() -> gr.Blocks:

    with gr.Blocks(title=blocks_name, css=css) as deduplicate_ui:
        with gr.Row():
            with gr.Column(scale=10):
                images_dir = gr.Textbox(label="图片目录")
            with gr.Column(scale=1):
                use_cache = gr.Checkbox(label="使用缓存")
                find_duplicates_images_button = gr.Button("扫描重复图片")
        with gr.Row():
            with gr.Column(scale=15):
                with gr.Row():
                    with gr.Accordion("操作Tips", open=False):
                        gr.Markdown(process_clusters_tips_markdown)
                with gr.Row():
                    duplicates_images_gallery = gr.Gallery(label="重复图片", value=[]).style(columns=6, height="auto", preview=False)
                with gr.Row():
                    confirm_button = gr.Button("选择图片")
                    cancel_button = gr.Button("取消图片")
                with gr.Row():
                    image_info_json = gr.JSON()
            with gr.Column(scale=1):
                with gr.Accordion("操作（打开以进行自动选择、删除等操作）", open=False):
                    process_clusters_method = gr.Radio(
                        label="图片处理方式（只对选中图片有效！）",
                        value=process_clusters_method_choices[1],
                        choices=process_clusters_method_choices,
                        type="index",
                    )
                    confirm_cluster_button = gr.Button("确定处理", elem_classes="attention")
                    auto_select_button = gr.Button("自动选择")
                    all_select_button = gr.Button("全部选择")
                selected_images_str = gr.Textbox(label="待操作列表（手动编辑时请保证toml格式的正确）")


        ############################## 绑定函数 ##############################

        # 按下后，在指定的目录搜索重复图像，并返回带标签的重复图像路径
        find_duplicates_images_button.click(
            fn=find_duplicates_images,
            inputs=[images_dir, use_cache],
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
            inputs=[duplicates_images_gallery, selected_images_str, process_clusters_method],
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
