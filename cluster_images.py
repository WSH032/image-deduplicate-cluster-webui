# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:38:31 2023

@author: WSH
"""
from typing import List

import gradio as gr

from ui import cluster_ui, wd14_ui


"""
注意！！！
如果WebUI开启queue，出错时会无限等待结果
但是出错时是不会有结果返回的，因此会一直等待
所以需要处理按钮交互的异常

这里采用的方式是为每个按钮函数添加一个错误处理的装饰器
各装饰器的放回值依据各函数的返回值而定
其实也可以把原输出接在输入函数后面，若出现异常就放回原值就行

可以使用from ui.tools.webui_error_wrapper import webui_error_default_wrapper
"""


############################## markdown ##############################

use_info_markdown = """
# 基于booru风格的tag标签或者WD14提取的特征向量进行聚类

## 对于tag标签的聚类
**可以使用WD14打标，也可以使用booru上下载的tag文本**
- 使用tfidf或者countvectorizer提取方法
- 要求目录下有与图片同名的`.txt`文件，其中内容为booru风格的tag标签

## 对于WD14提取的特征向量的聚类
**必须使用本项目自带的WD14脚本完成特征向量提取**
- 要求目录下有与图片同名的`.wd14.npz`文件，里面记录了每个图片的特征向量
- 要求目录下存在一个`wd14_vec_tag.wd14.txt`文件，里面记录了每个特征向量对应的tag

## WD14模型使用
- 你可以打开并修改`run_tagger.ps1`同时完成上述两个准备，该脚本采用友好交互编写
- 你也可以在`WD14 - tagger`选项卡中完成这个过程，两者是一样的
- 首次运行会下载WD14模型，可能需要等待一段时间
- 运行时候也需要等待，请去终端查看输出

## Credits
我不训练模型，WD14模型来自于这个项目[SmilingWolf/WD14](https://huggingface.co/SmilingWolf)

聚类方法和特征提取来着于sklearn库

tag_images_by_wd14_tagger来自[kohya](https://github.com/kohya-ss/sd-scripts/blob/main/finetune/tag_images_by_wd14_tagger.py)

## 😀Development
项目地址 及 更详细的使用方法请看:
**[WSH032/image-deduplicate-cluster-webui](https://github.com/WSH032/image-deduplicate-cluster-webui)**

如果你觉得此项目有用💪，可以去 [![GitHub Repo stars](https://img.shields.io/github/stars/WSH032/image-deduplicate-cluster-webui?style=social)](https://github.com/WSH032/image-deduplicate-cluster-webui) 点一颗小星星🤤，非常感谢你⭐

遇到问题可以在[Github上提issue ❓](https://github.com/WSH032/image-deduplicate-cluster-webui/issues)

"""


##############################  常量  ##############################

sub_blocks_css_list = [
    wd14_ui.css,
    cluster_ui.css,
]
def get_css_from_sub_blocks(sub_blocks_css_list: List[str]):
    # 去重
    deduplicat_sub_blocks_css_list = list( set(sub_blocks_css_list) )
    # 按照原来的顺序排序
    deduplicat_sub_blocks_css_list.sort(key=sub_blocks_css_list.index)
    return "\n".join(deduplicat_sub_blocks_css_list)


css = get_css_from_sub_blocks(sub_blocks_css_list)
title = "Cluster-Tagger"


############################## Blocks ##############################

def create_ui() -> gr.Blocks:

    with gr.Blocks(title=title, css=css) as demo:
            
        with gr.Accordion(label="使用说明", open=False):
            gr.Markdown(use_info_markdown)

        # 图片聚类 #

        with gr.Tab(cluster_ui.blocks_name):
            cluster_ui.create_ui()

        # WD14模型使用 #
        
        with gr.Tab(wd14_ui.blocks_name):
            wd14_ui.create_ui()
    
    return demo


############################## 命令行启动 ##############################

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(inbrowser=True,debug=True)
