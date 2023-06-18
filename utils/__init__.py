import sys
import os
import logging

__corrent_path__ = os.path.dirname( os.path.abspath(__file__) )

# 将本文件夹加入到sys.path中
sys.path.append( __corrent_path__  )
logging.info(f"{__file__}: Add {__corrent_path__} to sys.path")
