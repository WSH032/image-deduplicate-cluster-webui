"""
！！！！警告！！！！

不要移动此文件，它为整个项目提供路径支持
"""

import os

PATH_TOOLS_PATH = os.path.abspath(__file__)

TOOLS_DIR = os.path.dirname(PATH_TOOLS_PATH)

UI_DIR = os.path.dirname(TOOLS_DIR)

CWD = os.path.dirname(UI_DIR)  # 整个项目的根目录
"""
from tools import path_tools
path_tools.CWD
# 请用这个而不是import CWD，这样能保证一起变化
"""
