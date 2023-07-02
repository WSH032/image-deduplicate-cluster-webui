import os
import logging

import gradio as gr

from ui.wd14_fn import (
    use_wd14,
    release_memory,
    wd14_model_dir,
)
from ui.tools import path_tools

############################## markdown ##############################

def show_tensorrt_install_info():
    abs_install_tensortrt_ps1_path = os.path.join(path_tools.CWD, "utils", "run_install_tensorrt_lib.ps1")
    if os.path.exists(abs_install_tensortrt_ps1_path):
        return_str = f"**Win10用户可以运行 [{abs_install_tensortrt_ps1_path}]({abs_install_tensortrt_ps1_path}) 完成tensorrt的安装**"
    else:
        logging.debug(f"show_tensorrt_install_info(): not found {abs_install_tensortrt_ps1_path}")
        return_str = "show_tensorrt_install_info()出错，这应该不是你的问题，可以提issue"
    
    return return_str

experimental_features_markdown = f"""
**实验性功能**

**并发推理似乎可以代替`多进程数据读取`，甚至其在小数据集情况下启动非常快，在使用时建议将`多进程数据读取`设置为0**

**在从普通模式和TensorRT加速模式切换时，请先进行释放模型**

{show_tensorrt_install_info()}
"""

wd14_model_download_info_markdown = f"""
**首次使用会自动下载并编译模型，耗时较久，请耐心等待**

**出现网络问题，你也可以手动下载[https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2)中下述相应内容并放入`{wd14_model_dir}`目录内**

```
{wd14_model_dir}{os.path.sep}
├── variables{os.path.sep}
│   ├── variables.data-00000-of-00001
│   └── variables.index
├── keras_metadata.pb
├── saved_model.pb
└── selected_tags.csv
```

"""

wd14_parameter_reference_markdown = """
**合理选择`batch_size`和`数据读取进程`可以加快推理速度**

冷启动，GTX2060：
| 图片数量 | batch | 数据读取进程数 | 并发推理 | 耗时 |
| --- | --- | --- | --- | --- |
| 204 | 1batch | 0 | ❌ | 64s |
| 204 | 4batch | 2 | ❌ | 51s |
| 204 | 1batch | 0 | ✅ | 44s |
| 204 | 4batch | 0 | ✅ | 28s |
| 204 | 4batch | 2 | ✅ | 49s |

204张图片，热启动，tensor RT，4batch，不开多进程读取 = 2g显存占用，24s
"""

############################## Blocks ##############################


css = """
.attention {color: red  !important}
"""
blocks_name = "WD14 - tagger"


# 函数式创建有助于刷新ui界面
def create_ui() -> gr.Blocks:

    with gr.Blocks(title=blocks_name, css=css) as wd14_ui:

        with gr.Row():
            with gr.Accordion("模型下载说明", open=False):
                gr.Markdown(wd14_model_download_info_markdown)
            with gr.Accordion("性能参数参考", open=False):
                gr.Markdown(wd14_parameter_reference_markdown)
        with gr.Row():
            wd14_finish_Markdown = gr.Markdown(value="如果要使用WD14打标,在图片目录框填入路径后点击")
        with gr.Row():
            train_data_dir = gr.Textbox(label="Tagger目录", value="")
            use_wd14_button = gr.Button("WD14模型打标", elem_classes="attention")
            release_memory_button = gr.Button("释放内存或者显存中的模型")
        with gr.Row():
            repo_id = gr.Dropdown(
                ["SmilingWolf/wd-v1-4-moat-tagger-v2"],
                label="repo_id",
                value="SmilingWolf/wd-v1-4-moat-tagger-v2",
                type="value",
            )
            force_download = gr.Checkbox(label="强制重新下载模型", value=False, info="如果模型已经存在，是否强制下载覆盖")
            # TODO: 先暂时不要让用户选择目录，这个目录会在use_wd14()被强制覆盖
            model_dir = gr.Textbox(label="模型下载目录", value=wd14_model_dir, placeholder=wd14_model_dir, visible=False)
        with gr.Row():
            batch_size = gr.Slider(label="batch_size", value=1, minimum=1, maximum=16, step=1, info="越大显存占用越大")
            max_data_loader_n_workers = gr.Slider(label="多进程进行数据读取", value=0, minimum=0, maximum=16, step=1, info="设置成0则不启用，创建进程也有时间开销，建议30张以下就不要启用了")
        with gr.Row():
            general_threshold = gr.Slider(label="general_threshold", value=0.35, minimum=0, maximum=1.0, step=0.01)
            character_threshold = gr.Slider(label="character_threshold", value=0.35, minimum=0, maximum=1.0, step=0.01)
        with gr.Row():
            caption_extension = gr.Textbox(
                label="tag文件扩展名",
                value=".txt",
                info="如果你想使用聚类功能，扩展名因为设置为'.txt'",
                placeholder=".txt",
            )
            undesired_tags = gr.Textbox(label="undesired_tags", value="", info="不想要的tag，用逗号分隔，不加空格", placeholder="tag0,tag1...")
            remove_underscore = gr.Checkbox(label="remove_underscore", value=True, info="将tag中的下划线替换为空格")
        gr.Markdown( experimental_features_markdown )
        with gr.Row():
            concurrent_inference = gr.Checkbox(
                label="concurrent_inference",
                value=False,
                info="并发推理，可能会加快速度，可能会占用更多内存，建议在GPU模式使用",
            )
            tensorrt = gr.Checkbox(label="tensorrt", value=False, info="使用tensorrt加速，需要安装tensorrt,首次使用需要一段时间的编译模型")
            tensorrt_batch_size = gr.Slider(
                label="tensorrt_batch_size",
                value=2,
                minimum=1,
                maximum=8,
                step=1,
                info="编译后tensorrt模型所支持的最大batch_size，越大编译时间越长，不建议大于4; 改变后需要重新编译，不建议再次改变"
            )


        ############################## 绑定函数 ##############################

        # 使用wd14模型打标
        use_wd14_button.click(
            fn=use_wd14,
            inputs=[
                train_data_dir,
                repo_id,
                force_download,
                model_dir,
                batch_size,
                max_data_loader_n_workers,
                general_threshold,
                character_threshold,
                caption_extension,
                undesired_tags,
                remove_underscore,
                concurrent_inference,
                tensorrt,
                tensorrt_batch_size,
            ],
            outputs=[wd14_finish_Markdown],
        )

        # 释放内存或者显存中的模型
        release_memory_button.click(
            fn=release_memory,
            inputs=[],
            outputs=[],
        )
    return wd14_ui
