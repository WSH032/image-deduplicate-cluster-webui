from typing import Callable, Union
import logging
import time
import os
import gc

import gradio as gr

from tag_images_by_wd14_tagger import Tagger


############################## 参数常量 ##############################

WD14_MODEL_DIR_NAME = "wd14_tagger_model"  # webui中默认锁定的下载模型的目录名字

WD14_MODEL_DIR = os.path.join(os.getcwd(), WD14_MODEL_DIR_NAME)
if not os.path.exists(WD14_MODEL_DIR):
    os.mkdir(WD14_MODEL_DIR)

############################## 全局变量 ##############################

# 用于保存模型类实例，避免每次都要加载模型，浪费时间
# 注意这个不能放在gr.State中，否则刷新界面后就内存泄漏了
tagger: Union[None, Tagger] = None


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
    # 模型下载相关
    model_type: int,
    keep_updating: bool,
    # 数据集相关
    train_data_dir: str,
    batch_size: int,
    max_data_loader_n_workers: int,
    recursive: bool,
    # 模型推理参数
    general_threshold: float,
    characters_threshold: float,
    caption_extension: str,
    remove_underscore: bool,
    rating: bool,
    undesired_tags: str,
    # 推理并发
    concurrent_inference: bool,
    # tensorrt相关
    if_use_tensorrt: bool,
    tensorrt_batch_size: int,
) -> dict:

    # TODO: 让用户自行指定
    model_dir = WD14_MODEL_DIR  # 先把用户输入的目录强制覆盖掉
    print(f"载入模型: {WD14_MODEL_DIR}")

    global tagger  #  存放对模型的引用，保证其在内存中

    # 如果首次点击推理时，还没初始化过模型，就初始化模型
    if not isinstance(tagger, Tagger):
        tagger = Tagger(
            model_dir = model_dir,
            model_type = model_type,
            keep_updating = keep_updating,
        )

    if not train_data_dir:
        # 发出一个空路径警告，如果希望指定的是当前目录，应该显式指定
        logging.warning(
            (
                "Warning, Empty path input! Tagger will work in the current folder. "
                "If you want to specify the current folder, you should explicitly specify it"
            )
        )

    use_wd14_start_time = time.time()
    
    tagger.inference(
        # 数据集相关
        train_data_dir = train_data_dir,
        batch_size = batch_size,
        max_data_loader_n_workers = max_data_loader_n_workers,
        recursive = recursive,
        # 模型推理参数
        general_threshold = general_threshold,
        characters_threshold = characters_threshold,
        caption_extension = caption_extension,
        remove_underscore = remove_underscore,
        rating = rating,
        debug = False,  # 强制关闭debug模式
        undesired_tags = undesired_tags,
        # 推理并发
        concurrent_inference = concurrent_inference,
        # tensorrt相关
        tensorrt_batch_size = tensorrt_batch_size if if_use_tensorrt else -1,  # -1或者None都可以表示禁用
    )

    total_time = time.time() - use_wd14_start_time
    print_str = f"完成了，打标地址: {train_data_dir}\n总共用时{total_time:.2f}秒"
    print(print_str)

    return gr.update(value = print_str)


def release_memory() -> None:
    """ 将全局变量tagger赋值为None，让其失去对模型的引用，从而释放显存 """
    global tagger

    if isinstance(tagger, Tagger):
        # 使用这个保证卸载模型
        tagger.unload_model()

    # 将tagger赋值为None，可以更换模型种类
    tagger = None
    time.sleep(0.1)
    gc.collect()
    
    print("释放完成")
    return None
