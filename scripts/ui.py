"""
在SD-WebUI中注册一个tab界面
"""

from modules import script_callbacks # type: ignore # SD-WebUI自带的依赖
import gradio as gr
from cluster_images import demo as cluster_images_demo
from deduplicate_images import demo as deduplicate_images_demo

def create_demo():
	with gr.Blocks("Deduplicate-Cluster-Image") as demo:
		with gr.TabItem("Deduplicate Images"):
			deduplicate_images_demo.render()
		with gr.TabItem("Cluster Images"):
			cluster_images_demo.render()
	return demo	

def ui_tab():
		return [(create_demo(), "Deduplicate-Cluster-Image", "Deduplicate-Cluster-Image")]
		
script_callbacks.on_ui_tabs(ui_tab)
