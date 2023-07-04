"""
！！！！警告！！！！

不要移动此文件，它为整个项目提供路径支持
"""

import os

path_tools_path = os.path.abspath(__file__)

tools_dir = os.path.dirname(path_tools_path)

ui_dir = os.path.dirname(tools_dir)

CWD = os.path.dirname(ui_dir)  # 整个项目的根目录
"""
from tools import path_tools
path_tools.CWD
# 请用这个而不是import CWD，这样能保证一起变化
"""
