"""
为baf6946e06249c5af9851c60171692c44ef633e0的A1111-SD-WebUI-Extension-Cluster-Image插件添加依赖
有些环境在SD里自带了，这里就不安装
"""

import launch # type: ignore # SD-WebUI自带的依赖
import os

extension_name = "Deduplicate-Cluster-Image"

# 注意，不能在SD-WebUI里重装gradio，会导致奔溃
pip_list = [### for all ###
            "tqdm",
            "pandas",
            "pillow",
            "numpy",
            ### for deduplicate ###
            "toml",
            "imagededup",
            ### for cluster ###
            "\"scikit-learn >= 1.2.2\"",  # 在shell里执行这个指令需要用双引号括起来
            ### for tagger ###
            "\"tensorflow>=2.10.1, <2.11\"",  # 在shell里执行这个指令需要用双引号括起来
            "huggingface_hub",
            "\"opencv-python>=4.7.0.68\"",
            "onnx",
            "onnxruntime-gpu",
            "tf2onnx",
]

test_list = ["pandas",
             "PIL",
             "toml",
             "imagededup",
             "sklearn",
             "tensorflow",
             "cv2",
             "onnx",
             "onnxruntime",
             "tf2onnx",
]

for test in test_list:
    if not launch.is_installed(test):
        print(f"Pip for {extension_name}")
        print(f"pip install {' '.join(pip_list)}")
        launch.run_pip(f"install {' '.join(pip_list)}")
        print("Done")
        break
