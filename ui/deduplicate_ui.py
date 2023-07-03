import gradio as gr

from ui.deduplicate_fn import (
    find_duplicates_images,
    get_choose_image_index,
    confirm,
    cancel,
    delet,
    auto_select,
    all_select,
)

css = """
.attention {color: red  !important}
"""

# 函数式创建有助于刷新ui界面
def create_ui() -> gr.Blocks:

    with gr.Blocks(css=css) as deduplicate_ui:
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
                delet_button = gr.Button("删除（不可逆）", elem_classes="attention")
                auto_select_button = gr.Button("自动选择")
                all_select_button = gr.Button("全部选择")
                delet_images_str = gr.Textbox(label="待删除列表（手动编辑时请保证toml格式的正确）")


        ############################## 绑定函数 ##############################

        # 按下后，在指定的目录搜索重复图像，并返回带标签的重复图像路径
        find_duplicates_images_button.click(
            fn=find_duplicates_images,
            inputs=[images_dir, use_cache],
            outputs=[duplicates_images_gallery, delet_images_str],
        )

        # 点击一个图片后，记录该图片标签于全局变量choose_image_index，并且把按钮更名为该标签;同时显示该图片分辨率等信息
        duplicates_images_gallery.select(
            fn=get_choose_image_index,
            inputs=[delet_images_str],
            outputs=[confirm_button, cancel_button, image_info_json],
        )

        # 按下后，将全局变量choose_image_index中的值加到列表中,同时将取消按钮变红
        confirm_button.click(
            fn=confirm,
            inputs=[delet_images_str],
            outputs=[delet_images_str, confirm_button, cancel_button],
        )

        # 按下后，将全局变量choose_image_index中的值从列表中删除,同时将选择按钮变红
        cancel_button.click(
            fn=cancel,
            inputs=[delet_images_str],
            outputs=[delet_images_str, confirm_button, cancel_button],
        )
        
        # 按下后，删除指定的图像，并更新画廊
        delet_button.click(
            fn=delet,
            inputs=[duplicates_images_gallery, delet_images_str],
            outputs=[duplicates_images_gallery, delet_images_str],
        )
        
        # 按下后，用启发式算法自动找出建议删除的重复项
        auto_select_button.click(
            fn=auto_select,
            inputs=[],
            outputs=[delet_images_str],
        )
        # 按下后，选择全部图片
        all_select_button.click(
            fn=all_select,
            inputs=[],
            outputs=[delet_images_str]
        )
    
    return deduplicate_ui
