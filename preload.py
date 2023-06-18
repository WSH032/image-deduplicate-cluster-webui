"""
需要这个文件的原因是
在 tag_images_by_wd14_tagger.py 中，编译onnx模型时需要创建子进程
如果在SD-WebUI中进行这一过程，会出现 ModuleNotFoundError: No module named 'tag_images_by_wd14_tagger' 的报错
推测是SD-WebUI主进程工作目录和插件工作目录不一致
即使在 cluster_images.py 或者 tag_images_by_wd14_tagger.py 中使用 sys.path.append() 也无法解决
故需要在SD-WebUI启动前，将 tag_images_by_wd14_tagger.py 所在的目录添加到sys.path中
"""

import os
import sys
import logging

py_name = "tag_images_by_wd14_tagger.py"
extension_name = "Deduplicate-Cluster-Image"

__my_dir__ = os.path.dirname( os.path.abspath(__file__) )

if not os.path.exists( os.path.join( __my_dir__, py_name ) ):
    logging.warning(f"{extension_name}: Warning! tag_images_by_wd14_tagger.py is not found in {__my_dir__}")

sys.path.append( __my_dir__ )
print(f"{extension_name}: Add {__my_dir__} to sys.path")
