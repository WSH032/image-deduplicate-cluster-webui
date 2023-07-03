from typing import Callable, Union
import logging
import time
import gc
import os

import gradio as gr
import onnxruntime as ort

import tag_images_by_wd14_tagger as tagger
from ui.tools import path_tools

############################## 参数常量 ##############################

wd14_model_dir_name = "wd14_tagger_model"  # webui中默认锁定的下载模型的目录
wd14_model_dir = os.path.join(path_tools.CWD, wd14_model_dir_name)
if not os.path.exists(wd14_model_dir):
    os.mkdir(wd14_model_dir)

############################## 全局变量 ##############################

# 用于保存模型，避免每次都要加载模型，浪费时间
model_in_memory: Union[None, ort.InferenceSession] = None


##############################  tagger 函数  ##############################

def use_wd14_exception_wrapper(func) -> Callable:
    """
    用于处理use_wd14函数的异常
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"{func.__name__}函数出现了异常: {e}")
            return gr.update(value = f"出现了异常: {e}")
    return wrapper

@use_wd14_exception_wrapper
# 运行这个可以启动WD14 tagger脚本来打标
def use_wd14(
    train_data_dir: str,
    repo_id: str,
    force_download: bool,
    model_dir: str,
    batch_size: int,
    max_data_loader_n_workers: int,
    general_threshold: float,
    character_threshold: float,
    caption_extension: str,
    undesired_tags: str,
    remove_underscore: bool,
    concurrent_inference: bool,
    tensorrt: bool,
    tensorrt_batch_size: int,
    ) -> dict:

    global model_in_memory  #  存放对模型的引用，保证其在内存中

    # TODO
    model_dir = wd14_model_dir  # 先把用户输入的目录强制覆盖掉

    use_wd14_start_time = time.time()
    cmd_params_list = [
        train_data_dir,
        f"--repo_id={repo_id}",
        f"--model_dir={model_dir}",
        f"--batch_size={batch_size}",
        f"--caption_extension={caption_extension}",
        f"--general_threshold={general_threshold}",
        f"--character_threshold={character_threshold}",
        f"--undesired_tags={undesired_tags}",
    ]
    if force_download:
        cmd_params_list.append("--force_download")
    # 如果为0，就不会使用多线程读取数据
    if isinstance(max_data_loader_n_workers, int) and max_data_loader_n_workers > 0 :
        cmd_params_list.append(f"--max_data_loader_n_workers={max_data_loader_n_workers}")
    if remove_underscore:
        cmd_params_list.append("--remove_underscore")
    if concurrent_inference:
        cmd_params_list.append("--concurrent_inference")
    if tensorrt:
        cmd_params_list.append("--tensorrt")
        cmd_params_list.append(f"--tensorrt_batch_size={tensorrt_batch_size}")
    
    parser = tagger.setup_parser()
    args = parser.parse_args( cmd_params_list )

    # 首次运行之前，model_in_memory为None; 运行一次过后，tagger.main会返回一个ort.InferenceSession
    # 保持对他引用，可以将模型保存在内存中，下次运行时就不用再加载模型了
    model_in_memory = tagger.main(args, model_in_memory)

    total_time = time.time() - use_wd14_start_time
    print_str = f"完成了，打标地址: {train_data_dir}\n总共用时{total_time:.2f}秒"
    print(print_str)

    return gr.update(value = print_str)


def release_memory() -> None:
    """ 将全局变量model_in_memory赋值为None，让其失去对模型的引用，从而释放显存 """
    global model_in_memory
    model_in_memory = None
    time.sleep(0.1)
    gc.collect()
    print("释放完成")
    return None
