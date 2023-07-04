"""
在SD-WebUI中注册tab界面
"""

from modules import script_callbacks  # type: ignore # SD-WebUI自带的依赖
import cluster_images
import deduplicate_images


def deduplicate_images_ui_tab():
	return (deduplicate_images.create_ui(), deduplicate_images.title, deduplicate_images.title)

def cluster_images_ui_tab():
	return (cluster_images.create_ui(), cluster_images.title, cluster_images.title)

def ui_tab():
	return [cluster_images_ui_tab(), deduplicate_images_ui_tab()]

script_callbacks.on_ui_tabs(ui_tab)
