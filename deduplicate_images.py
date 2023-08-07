# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:38:31 2023

@author: WSH
"""


import gradio as gr

from img_dedup_clust import deduplicate_ui

"""
注意！！！
如果WebUI开启queue，出错时会无限等待结果
但是出错时是不会有结果返回的，因此会一直等待
所以需要处理按钮交互的异常

这里采用的方式是为每个按钮函数添加一个错误处理的装饰器
各装饰器的放回值依据各函数的返回值而定
其实也可以把原输出接在输入函数后面，若出现异常就放回原值就行

可以使用from img_dedup_clust.tools.webui_error_wrapper import webui_error_default_wrapper
"""


##############################  常量  ##############################

css = deduplicate_ui.css
title = deduplicate_ui.blocks_name


############################## Blocks ##############################

def create_ui() -> gr.Blocks:

    with gr.Blocks(title=title, css=css) as demo:
        deduplicate_ui.create_ui()
    
    return demo


############################## 命令行启动 ##############################

if __name__ == "__main__":

    # 不在SD-WebUI里运行，没有gradioApp()，所以需要修改BaseJS的类属性以达到全局应用documnet
    from img_dedup_clust.tools.js import BaseJS
    BaseJS.set_cls_attr(is_a1111_webui=False)

    demo = create_ui()
    demo.launch(debug=True, inbrowser=True)
