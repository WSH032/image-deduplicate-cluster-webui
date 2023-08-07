import os
import logging
from urllib.parse import urljoin

import gradio as gr

from img_dedup_clust.wd14_fn import (
    use_wd14,
    release_memory,
    WD14_MODEL_DIR,  # 模型强制的下载目录
)
from img_dedup_clust.tools import path_tools
from tag_images_by_wd14_tagger import (
    DEFAULT_WD14_TAGGER_REPO,  # huggingface上的WD14模型仓库ID
    WD14_MODEL_TYPE_LIST,  # 允许用户选择的模型种类名字列表，与tag_images_by_wd14_tagger.py后端中一致
    WD14_MODEL_OPSET,  # 模型的opset
    DELIMITER,  # 模型名字的分隔符
    TAG_FILES,  # 模型所用tag文件的名字列表
    DEFAULT_TAGGER_CAPTION_EXTENSION,  # 默认打标文件的扩展名
    DEFAULT_TAGGER_THRESHOLD,  # 默认的tagger阈值
)


##############################  常量  ##############################

css = """
.attention {color: red  !important}
.recommendation {color: dodgerblue !important}
"""
blocks_name = "WD14 - tagger"

HUGGINGFACE_BASE_URL = "https://huggingface.co/"
DEFAULT_WD14_TAGGER_REPO_ULR = urljoin(HUGGINGFACE_BASE_URL, DEFAULT_WD14_TAGGER_REPO)
TENSORRT_INSTALL_GUIDE_URL = "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"


############################## markdown ##############################

########## 实验性功能 ##########
def show_tensorrt_install_info():
    abs_install_tensortrt_ps1_path = os.path.join(path_tools.CWD, "utils", "run_install_tensorrt_lib.ps1")
    if os.path.exists(abs_install_tensortrt_ps1_path):
        return_str = f"Win10用户可以运行 [{abs_install_tensortrt_ps1_path}]({abs_install_tensortrt_ps1_path}) 完成tensorrt的安装"
    else:
        logging.debug(f"show_tensorrt_install_info(): not found {abs_install_tensortrt_ps1_path}")
        return_str = "show_tensorrt_install_info()出错，这应该不是你的问题，可以提issue"
    
    return return_str

experimental_features_markdown = f"""
**实验性功能**

**在从普通模式和TensorRT加速模式切换时，请先进行释放模型**

**{show_tensorrt_install_info()}**

**或者阅读 [{TENSORRT_INSTALL_GUIDE_URL}]({TENSORRT_INSTALL_GUIDE_URL}) 完成安装**
"""


########## 下载提示 ##########
try:
    wd14_model_download_info_markdown = f"""
**首次使用会自动下载并编译模型，耗时较久，请耐心等待**

**出现网络问题，你也可以手动下载[{DEFAULT_WD14_TAGGER_REPO_ULR}]({DEFAULT_WD14_TAGGER_REPO_ULR})中下述相应内容并放入`{WD14_MODEL_DIR}`目录内**

```
{WD14_MODEL_DIR}{os.path.sep}
├── {WD14_MODEL_TYPE_LIST[0]}{os.path.sep}
│   └── {WD14_MODEL_TYPE_LIST[0]}{DELIMITER}opset{WD14_MODEL_OPSET}.onnx
├── {TAG_FILES[0]}
├── {TAG_FILES[1]}
├── {TAG_FILES[2]}
└── {TAG_FILES[3]}
```

"""
# 关于TAG_FILES的部分采用了硬编码，所以这里必须判断是否出错
except Exception as e:
    logging.exception(f"wd14_model_download_info_markdown出现了异常: {e}")
    wd14_model_download_info_markdown = f"wd14_model_download_info_markdown出现了异常: {e}\n这应该不是你的问题，可以提issue"


########## 参数提示 ##########
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

**并发推理似乎可以代替`多进程数据读取`，甚至其在小数据集情况下启动非常快，在使用时建议将`多进程数据读取`设置为0**

"""


############################## Blocks ##############################

# 函数式创建有助于刷新ui界面
def create_ui() -> gr.Blocks:

    with gr.Blocks(title=blocks_name, css=css) as wd14_ui:

        with gr.Row():
            with gr.Accordion("模型下载说明", open=False):
                gr.Markdown(wd14_model_download_info_markdown)
            with gr.Accordion("性能参数参考", open=False):
                gr.Markdown(wd14_parameter_reference_markdown)
        with gr.Row():
            wd14_finish_Markdown = gr.Markdown(value="请在Tagger目录框内填入需要标记的图片文件夹路径")
        with gr.Row():
            train_data_dir = gr.Textbox(label="Tagger目录", value="")
            recursive = gr.Checkbox(label="递归处理子文件夹", value=False, info="勾选后会递归遍历子目录")
        with gr.Row():
            wd14_model_type = gr.Dropdown(
                WD14_MODEL_TYPE_LIST,
                label="模型选择（切换前请先释放模型）",
                value=WD14_MODEL_TYPE_LIST[0],
                type="index",
            )
            keep_updating = gr.Checkbox(label="保持联网更新模型", value=True, info="取消勾选以进行离线模式")
            use_wd14_button = gr.Button("WD14模型打标", variant="primary")
            release_memory_button = gr.Button("释放内存或者显存中的模型", elem_classes="recommendation")
            # TODO: 先暂时不要让用户选择目录
            # model_dir = gr.Textbox(label="模型下载目录", value=wd14_model_dir, placeholder=wd14_model_dir, visible=False)
        with gr.Row():
            batch_size = gr.Slider(label="batch_size", value=1, minimum=1, maximum=16, step=1, info="越大显存占用越多")
            max_data_loader_n_workers = gr.Slider(
                label="启用DataLoader多进程数据读取（建议>300张）",
                value=0,
                minimum=-1,
                maximum=16,
                step=1,
                info="设置成-1则不启用，0则使用DataLoader但不使用多进程，大于0则启用相应个子进程读取数据"
            )
            concurrent_inference = gr.Checkbox(
                label="concurrent_inference",
                value=False,
                info="并发推理，加快速度，但占用更多内存，建议在GPU模式使用",
            )
        with gr.Row():
            general_threshold = gr.Slider(label="general_threshold", value=DEFAULT_TAGGER_THRESHOLD, minimum=0, maximum=1.0, step=0.01, info="常规标签阈值（设置为1则不生成）")
            characters_threshold = gr.Slider(label="characters_threshold", value=DEFAULT_TAGGER_THRESHOLD, minimum=0, maximum=1.0, step=0.01, info="特定角色标签阈值（设置为1则不生成）")
        with gr.Row():
            caption_extension = gr.Textbox(
                label="tag文件扩展名",
                value=DEFAULT_TAGGER_CAPTION_EXTENSION,
                info=f"如果你想使用聚类功能，扩展名应设置为'{DEFAULT_TAGGER_CAPTION_EXTENSION}'",
                placeholder=f"{DEFAULT_TAGGER_CAPTION_EXTENSION}",
            )
            undesired_tags = gr.Textbox(label="undesired_tags", value="", info="不想要的tag，逗号分隔（可以空格）", placeholder="tag0, tag1...")
            remove_underscore = gr.Checkbox(label="remove_underscore", value=True, info="将tag中的下划线替换为空格")
            rating = gr.Checkbox(label="rating", value=False, info="是否标记限制级tags")
        gr.Markdown( experimental_features_markdown )
        with gr.Row():
            if_use_tensorrt = gr.Checkbox(label="use_tensorrt", value=False, info="是否使用tensorrt加速，需要安装tensorrt,首次使用需要一段时间的编译模型")
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
                # 模型下载相关
                wd14_model_type,
                keep_updating,
                # 数据集相关
                train_data_dir,
                batch_size,
                max_data_loader_n_workers,
                recursive,
                # 模型推理参数
                general_threshold,
                characters_threshold,
                caption_extension,
                remove_underscore,
                rating,
                undesired_tags,
                # 推理并发
                concurrent_inference,
                # tensorrt相关
                if_use_tensorrt,
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
