"""
！！！！警告！！！！

不要移动此文件，它为整个项目提供路径支持
"""

import os

PATH_TOOLS_PATH = os.path.abspath(__file__)

TOOLS_DIR = os.path.dirname(PATH_TOOLS_PATH)

PACKAGE_DIR = os.path.dirname(TOOLS_DIR)  # 做为包的文件夹的路径

CWD = os.path.dirname(PACKAGE_DIR)  # 整个项目的根目录路径
"""
from tools import path_tools
path_tools.CWD
# 请用这个而不是import CWD，这样能保证一起变化
"""
