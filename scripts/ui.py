"""
在SD-WebUI中注册tab界面
"""

from modules import script_callbacks  # type: ignore # SD-WebUI自带的依赖

import cluster_images
import deduplicate_images
from img_dedup_clust.tools.js import BaseJS


# 起到全局修改效果，用A1111_WebUI提供的gradioApp()代替documnet
BaseJS.set_cls_attr(is_a1111_webui=True)


def deduplicate_images_ui_tab():
	return (deduplicate_images.create_ui(), deduplicate_images.title, deduplicate_images.title)

def cluster_images_ui_tab():
	return (cluster_images.create_ui(), cluster_images.title, cluster_images.title)

def ui_tab():
	"""注意，此函数要求能在 sys.path 已经被还原的情况下正常调用"""
	return [cluster_images_ui_tab(), deduplicate_images_ui_tab()]

script_callbacks.on_ui_tabs(ui_tab)
