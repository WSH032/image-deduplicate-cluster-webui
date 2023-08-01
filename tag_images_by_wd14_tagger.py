# -*- coding: UTF-8 -*-

# from https://github.com/kohya-ss/sd-scripts/blob/16e5981d3153ba02c34445089b998c5002a60abc/finetune/tag_images_by_wd14_tagger.py


import os
from typing import List, Optional, Tuple, Union
import gc
import time
import logging
from pathlib import Path
import concurrent.futures
import shutil
from collections import Counter


# 把torh放在onnxruntime的前面导入，让它调用它的cuda环境
import torch

from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import toml
from torch.utils.data import DataLoader, Dataset


# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # 设置懒惰启动，加快载入


######################################## 常量 ########################################

# from wd14 tagger
IMAGE_SIZE = 448  # wd14 接受的图片尺寸
IMAGE_MODE = "RGB"  # wd14 接受的图片模式
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]


DEFAULT_WD14_TAGGER_REPO = "WSH032/wd-v1-4-tagger-feature-extractor"
WD14_MODEL_TYPE_LIST  = ["wd-v1-4-moat-tagger-v2", "wd-v1-4-convnextv2-tagger-v2"]
WD14_MODEL_OPSET = 17
DELIMITER = "_"  # 仓库里就是这么命名的，不要改


TAG_FILES = [
    "candidate_labels_scores_pt.npz",
    "candidate_labels_scores_safetensors.npz",
    "wd14_tags.toml",
    "selected_tags.csv",
]

WD14_TAGS_TOML_FILE = TAG_FILES[2]  # 存储各列向量对应的tag的文件的名字
# 用于提醒我是否忘了修改
assert WD14_TAGS_TOML_FILE == "wd14_tags.toml", "WD14_TAGS_TOML似乎不是'wd14_tags.toml'"

MULTI_OUTPUT_NUMBER = 4  # 调整后输出层的数量, 注意，如果你动了这个，需要重构下面代码，包括run_batch函数；模型的固有属性

INPUT_NAME = "input_1"  # 模型输入层名字；模型的固有属性

WD14_NPZ_EXTENSION = ".wd14.npz"  # 用于保存推理所得特征向量的文件扩展名 # .wd14用来区分kohya的潜变量cache
WD14_NPZ_ARRAY_PREFIX = "layer"  # 所保存的npz文件中，特征向量的前缀名字，后面会加上层数，如layer0, layer1, layer2, layer3

TRT_ENGINE_CACHE_DIR = "trt_engine_cache"  # 用于缓存tensorrt引擎的子文件夹名字，将会存放于模型文件夹内

DEFAULT_TAGGER_THRESHOLD = 0.35
DEFAULT_TAGGER_CAPTION_EXTENSION = ".txt"

WD14_TAGS_CATEGORY_LIST = ["rating", "general", "characters"]  # wd14标签的种类


######################################## 全局变量 ########################################





######################################## 函数 ########################################

def read_wd14_tags_toml(wd14_tags_toml_path: str) -> Tuple[List[str], List[str], List[str]]:
    """读取tags文件

    Args:
        wd14_tags_toml_path (_type_): tags文件所在路径

    Returns:
        Tuple[List[str], List[str], List[str]]: 分别为rating_tags, general_tags, character_tags
    """

    # 读取标签
    with open(wd14_tags_toml_path, "r") as f:
        wd14_tags_toml = toml.load(f)
        wd14_tags_list = wd14_tags_toml["tags"]

        # 硬编码，提醒我是否发生改变
        assert wd14_tags_list[0]["name"] == WD14_TAGS_CATEGORY_LIST[0]
        assert wd14_tags_list[1]["name"] == WD14_TAGS_CATEGORY_LIST[1]
        assert wd14_tags_list[2]["name"] == WD14_TAGS_CATEGORY_LIST[2]

        rating_tags = wd14_tags_list[0]["tags"]
        general_tags = wd14_tags_list[1]["tags"]
        character_tags = wd14_tags_list[2]["tags"]
    
    return (rating_tags, general_tags, character_tags)


def preprocess_image(image) -> np.ndarray:
    """ 用于将输入图片预处理成模型可接受形式 """
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)  # wd14要求是32精度
    return image


def load_image(image_path: Union[Path, str]) -> np.ndarray:
    image = Image.open(image_path)
    if image.mode != IMAGE_MODE:
        image = image.convert(IMAGE_MODE)
    image = preprocess_image(image)
    return image


class ImageLoadingPrepDataset(Dataset):
    """ 用于多进程读取图片 """
    def __init__(self, image_paths: Union[ List[str], List[Path] ]):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Optional[Tuple[np.ndarray, str]]:
        img_path = str(self.images[idx])

        try:
            image_ndarray = load_image(img_path)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (image_ndarray, img_path)


def collate_fn_remove_corrupted(batch) -> list:
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def get_tensorrt_engine(trt_engine_cache_path:str ,tensorrt_batch_size: int) -> Tuple[str, dict]:

    Tensorrt_options = {
        "trt_timing_cache_enable": True,  # 时序缓存,可以适用于多个模型
        "trt_engine_cache_enable": True,  # 开启引擎缓存,针对特定模型、推理参数、GPU
        "trt_engine_cache_path":trt_engine_cache_path,
        # "trt_fp16_enable": False,  # FP16模式，需要GPU支持
        # "trt_int8_enable": False,  # INT8模式，需要GPU支持
        # "trt_dla_enable": False,  # DLA深度学习加速器，需要GPU支持
        "trt_build_heuristics_enable" : True,  # 启用启发式构建，减少时间
        "trt_builder_optimization_level": 3,  # 优化等级，越小耗时越少，但优化更差，不建议低于3
        "trt_profile_min_shapes": f"{INPUT_NAME}:1x{IMAGE_SIZE}x{IMAGE_SIZE}x3",  # 最小输入形状
        "trt_profile_max_shapes": f"{INPUT_NAME}:{tensorrt_batch_size}x{IMAGE_SIZE}x{IMAGE_SIZE}x3",  # 最大输入形状
        "trt_profile_opt_shapes": f"{INPUT_NAME}:{tensorrt_batch_size}x{IMAGE_SIZE}x{IMAGE_SIZE}x3",  # 优化输入形状
    }
    Tensorrt_provider = ("TensorrtExecutionProvider", Tensorrt_options)

    if Tensorrt_options["trt_engine_cache_enable"]:
        print(f"""
Your cache files will be stored in {Tensorrt_options["trt_engine_cache_path"]}
Warning: Please clean up any old engine and profile cache files (.engine and .profile) if any of the following changes:
1.Model changes (if there are any changes to the model topology, opset version, operators etc.)
2.ORT version changes (i.e. moving from ORT version 1.8 to 1.9)
3.TensorRT version changes (i.e. moving from TensorRT 7.0 to 8.0)
4.Hardware changes. (Engine and profile files are not portable and optimized for specific Nvidia hardware)
""")
    return Tensorrt_provider


def set_providers(
    tensorrt_batch_size: Optional[int] = None,
    trt_engine_cache_path: Optional[str] = None,
) -> list:
    """
    返回ort.InferenceSession的providers参数
    只有tensorrt_batch_size和trt_engine_cache_path都合法输入，才会调用get_tensorrt_engine()返回tensorrt执行参数

    Args:
        tensorrt_batch_size (Optional[int], optional): 当为大于0的整数时，会尝试以该batch_size调用tensorrt. Defaults to None.
        trt_engine_cache_path (str, Path): 用于缓存tensorrt引擎的目录，实际上是个文件夹

    Returns:
        list: 返回ort.InferenceSession的providers参数
    """

    # 配置执行者
    providers =  ort.get_available_providers()
    # 先判断是否大于等于2，因为有些用户可能没安装gpu版本，就只有CPU执行者
    if len(providers) >= 2:
        # 最后一个一般是CPU，倒数第二个一般是GPU
        providers = providers[-2:] if torch.cuda.is_available() else [ providers[-1] ]
    
    # 不要判断trt_engine_cache_path的类，因为可能是str，或者Path，或者别的什么
    if isinstance(tensorrt_batch_size, int) and tensorrt_batch_size > 0 and trt_engine_cache_path :

        # 加入带缓存参数的TensorRT执行者
        providers = [ get_tensorrt_engine(trt_engine_cache_path, tensorrt_batch_size) ] + providers
        # 提示信息
        print("#"*20 + "\n")
        print("使用TensorRT执行者,首次使用或者tensorrt_batch_side发生改变时，需要重新编译模型，耗时较久，请耐心等待，可以使用任务管理器跟踪显卡的使用")
        print("\n" + "#"*20)

    print("可用设备")
    for name in providers:
        print(name)

    return providers


def glob_images_pathlib(dir_path: Path, recursive: Optional[bool]) -> List[Path]:
    """找出符合模式的文件

    Args:
        dir_path (Path): 所搜寻的目录
        recursive (bool): 是否递归搜寻

    Returns:
        List[Path]: 符合模式的文件Path对象list
    """
    print(f"searching images in {dir_path}")
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # 重複を排除
    image_paths.sort()
    print(f"found {len(image_paths)} images.")

    return image_paths


def check_same_name_path(path_list: Union[List[Path], List[str]]) -> List[List[str]]:
    """检查path_list中的Path对象或者str对象是否有相同的名字（在扩展名不同的情况下）

    Args:
        path_list (Union[List[Path], List[str]]): 需要检查的Path对象或者str对象列表

    Returns:
        List[List[str]]: 相同名字的str对象列表，每一个子列表内是同一个相同名字的str对象集合
    """
    path_list = [str(path) for path in path_list]
    path_without_ext_list = [os.path.splitext(path)[0] for path in path_list]  # 去掉扩展名
    
    counter = Counter(path_without_ext_list)  # 统计每个名字出现的次数
    same_name_path_list = []  # 用于存放相同名字的索引

    for same_name_path_without_ext, count in counter.most_common():
        temp_sub_list = []
        # 因为是降序排列，所以当count <= 1时，代表后面的不重复，就不用再看了
        if count <= 1:
            break
        # 找到相同名字的索引，对应的就是相同名字的路径
        for index, path_without_ext in enumerate(path_without_ext_list):
            if path_without_ext == same_name_path_without_ext:
                temp_sub_list.append(path_list[index])

        if temp_sub_list:
            same_name_path_list.append(temp_sub_list)
    
    return same_name_path_list


def load_data(
    train_data_dir: str,
    recursive: Optional[bool] = None,
    use_torch_dataloader: Optional[bool] = None,
    **kwargs,
) -> Union[ DataLoader, List[ List[ Tuple[None, Path] ] ] ]:
    """
    读取train_data_dir下的图片数据
    由use_torch_dataloader决定是否使用DataLoader
    否则返回一个list，其元素也是list，子list的元素是一个元组，第一个元素为None，第二个元素为图片的Path对象
        后续需要你自己处理图片数据

    Args:
        train_data_dir (str): 所需要读取的图片目录
        recursive (Optional[bool], optional): 递归读取. Defaults to None.
        use_torch_dataloader (Optional[bool], optional): 是否使用DataLoader. Defaults to None.
        **kwargs: DataLoader的参数

    Returns:
        _type_: List[ List[ Tuple[None | np.ndarray, Path] ] ] ]
    """

    # 找出图片文件
    image_paths = glob_images_pathlib(Path(train_data_dir), recursive)
    
    # 因为run_batch中是通过os.path.splitext(image_path)[0] + ".txt"来写入标签文件的
    # 所以同名，但是扩展名的不同的图片，前面的图片会被后面图片的标签文件覆盖
    # 这里给用户一个警告
    same_name_path_list = check_same_name_path(image_paths)
    if same_name_path_list:
        warning_str = "以下文件名字相同，但扩展名不同，每类中只有最后一个会有标签文件\n\n"
        for sub_index, sub_list in enumerate(same_name_path_list):
            warning_str = warning_str + f"第{sub_index}类\n" + "\n".join(sub_list) + "\n"

        logging.warning(warning_str)

    # 読み込みの高速化のためにDataLoaderを使うオプション
    if use_torch_dataloader:
        print("将使用 torch.utils.DataLoader 进行数据读取")
        dataset = ImageLoadingPrepDataset(image_paths)
        data = DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn_remove_corrupted,  # 注意，这个会导致损坏的那一批次的图片数量可能不是batch_size
            drop_last=False,
            **kwargs,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    return data


def undesired_tags_str_to_list(undesired_tags: Optional[str] =  None) -> List[str]:
    """ 输入的undesired_tags使用逗号分割 """
    undesired_tags_list = []
    if isinstance(undesired_tags, str) and undesired_tags:
        undesired_tags_list = undesired_tags.split(",")
        undesired_tags_list = [tag.strip() for tag in undesired_tags_list]
        undesired_tags_list = list( set(undesired_tags_list) )
    return undesired_tags_list


def run_batch(
    path_imgs: List[Tuple[str, np.ndarray]],
    ort_session: ort.InferenceSession,
    rating_tags: List[str],
    general_tags: List[str],
    characters_tags: List[str],
    general_threshold: float = DEFAULT_TAGGER_THRESHOLD,
    characters_threshold: float = DEFAULT_TAGGER_THRESHOLD,
    caption_extension: str = DEFAULT_TAGGER_CAPTION_EXTENSION,
    remove_underscore: Optional[bool] = None,
    rating: Optional[bool] = None,
    debug: Optional[bool] = None,
    undesired_tags_list: List[str] = [],
):
    
    # 如果batch_size > 1，变为一个二维矩阵
    imgs = np.array([im for _, im in path_imgs])

    # 一定是一个元素
    inputs_name_list = [x.name for x in ort_session.get_inputs()]
    # 一定是{MULTI_OUTPUT_NUMBER}个元素
    outputs_name_list = [x.name for x in ort_session.get_outputs()]

    # 进行推理
    ort_inputs = {inputs_name_list[0]: imgs}
    ort_out = ort_session.run(outputs_name_list, ort_inputs)

    # 获得四个层的推理结果
    layer0_output, layer1_output, layer2_output, layer3_output = ort_out

    # 最外层的推理结果
    probs = layer0_output  # kohya原来就是这样命名的，我懒得改下面代码

    # 用于确定各类tags所对应的向量索引区间
    rating_tags_len = len(rating_tags)
    general_tags_len = len(general_tags)

    # 写入文本
    for_index = 0  # 起到循环计数器的作用，代表当前处理的行，即当前处理的样本
    for (image_path, _), prob in zip(path_imgs, probs):

        combined_tags = []  # 用于保存所有的tag
        general_tag_text = ""
        characters_tag_text = ""

        # rating_tags
        if rating:
            rating_prob = prob[ : rating_tags_len ]
            # 找到rating_prob中最大的那个
            max_rating_prob_index = np.argmax(rating_prob)
            combined_tags = [rating_tags[max_rating_prob_index]] + combined_tags
        # general_tags
        for i, p in enumerate( prob[ rating_tags_len : rating_tags_len + general_tags_len ] ):
            if p >= general_threshold:
                tag_name = general_tags[i]
                # 替换下划线为空格
                if remove_underscore and len(tag_name) > 3:  # ignore emoji tags like >_< and ^_^
                    tag_name = tag_name.replace("_", " ")
                # 不在不想要的词列表里就把它加进去
                if tag_name not in undesired_tags_list:
                    general_tag_text += ", " + tag_name
                    combined_tags.append(tag_name)
        # characters_tags
        for i, p in enumerate( prob[ rating_tags_len + general_tags_len :  ] ):
            if p >= characters_threshold:
                tag_name = characters_tags[i]
                # 替换下划线为空格
                if remove_underscore and len(tag_name) > 3:  # ignore emoji tags like >_< and ^_^
                    tag_name = tag_name.replace("_", " ")
                # 不在不想要的词列表里就把它加进去
                if tag_name not in undesired_tags_list:
                    characters_tag_text += ", " + tag_name
                    combined_tags.append(tag_name)

        # 先頭のカンマを取る
        if len(general_tag_text) > 0:
            general_tag_text = general_tag_text[2:]
        if len(characters_tag_text) > 0:
            characters_tag_text = characters_tag_text[2:]

        tag_text = ", ".join(combined_tags)
        
        with open(os.path.splitext(image_path)[0] + caption_extension, "wt", encoding="utf-8") as f:
            f.write(tag_text + "\n")
            if debug:
                print(f"\n{image_path}:\n  Characters tags: {characters_tag_text}\n  General tags: {general_tag_text}")

        # 矩阵写入同名的npz文件
        np_savez_kwargs = {
            f"{WD14_NPZ_ARRAY_PREFIX}0": layer0_output[for_index],
            f"{WD14_NPZ_ARRAY_PREFIX}1": layer1_output[for_index],
            f"{WD14_NPZ_ARRAY_PREFIX}2": layer2_output[for_index],
            f"{WD14_NPZ_ARRAY_PREFIX}3": layer3_output[for_index],
        }

        np.savez(
            os.path.splitext(image_path)[0] + WD14_NPZ_EXTENSION,
            **np_savez_kwargs,
        )
        # 处理下个样本
        for_index += 1



class Tagger:

    def __init__(
        self,
        model_dir: str,
        model_type: int,
        keep_updating: Optional[bool] = None,
    ):
        """初始化，下载或更新模型和读取tags文件

        Args:
            model_dir (str): 模型所在目录，建议为绝对路径；如果不存在模型会尝试下载
            model_type (int): 模型类型的序号，请对照WD14_MODEL_TYPE_LIST
            keep_updating (Optional[bool], optional): 当不为真时，只要文件存在，就不会再下载和尝试更新. Defaults to None.
        """

        ### 注意，以下初始化会有一定的执行顺序要求，修改前请注意 ###

        self.model_dir = model_dir  # 模型项目所在文件夹路径
        self.model_type = model_type  # 模型类型序号
        self.model_type_name = WD14_MODEL_TYPE_LIST[model_type]  # 模型类型名字，也是onnx模型所在的子文件夹名字

        # 存放tags文件的wd14_tags.toml的路径
        self.wd14_tags_toml_path = os.path.join(model_dir, WD14_TAGS_TOML_FILE)

        # onnx模型文件名字，类似：wd-v1-4-moat-tagger-v2_opset17.onnx
        self.model_name = (
            self.model_type_name + 
            DELIMITER +
            f"opset{WD14_MODEL_OPSET}" +
            ".onnx"
        )

        # onnx 模型所在路径
        self.model_path = os.path.join(model_dir, self.model_type_name, self.model_name)

        # 用于保存模型，避免每次都要加载模型，浪费时间
        self.model_in_memory: Union[None, ort.InferenceSession] = None

        #################### 调用实例方法初始化 ####################

        # 先运行一次下载，保证文件一定存在 ！！！
        self.download_model(keep_updating=keep_updating)

        # 读取标签
        # 所有模型都共用同样的标签文件，所以可以直接在这里初始化
        # 请先运行self.download_model()以保证文件一定存在  ！！！
        self.tags = read_wd14_tags_toml(self.wd14_tags_toml_path)


    def download_model(self, keep_updating: Optional[bool] = None) -> None:
        """
        从抱脸下载相关的tag文件和模型

        Args:
            keep_updating (Optional[bool], optional): 当不为真时，只要文件存在，就不会再下载和尝试更新. Defaults to None.

        这个实例方法不需要你来调用，一般在 __init__() 中调用
        """

        # hf_hub_downloadをそのまま使うとsymlink関係で問題があるらしいので、キャッシュディレクトリとforce_filenameを指定してなんとかする
        # depreacatedの警告が出るけどなくなったらその時
        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/issues/22

        download_dir = self.model_dir  # 下载到指定的模型项目目录
        wd14_model_dir_name_in_repo = self.model_type_name  # 模型类型名字与模型在repo中的文件夹同名
        model_name = self.model_name  # onnx模型文件名称
        model_path = self.model_path  # 预计onnx模型会被下载到的位置

        # 下载与tags有关的文件
        for file in TAG_FILES:
            if not os.path.exists( os.path.join(download_dir, file) ) or keep_updating:
                print(f"尝试下载{file}中")
                hf_hub_download(
                    repo_id = DEFAULT_WD14_TAGGER_REPO,
                    filename = file,
                    local_dir = download_dir,
                )
        
        # 下载模型
        if not os.path.exists( model_path ) or keep_updating:  # 实际上 model_path == os.path.join( download_dir, wd14_model_dir_name_in_repo, model_name )
            print(f"尝试下载{model_name}中")
            hf_hub_download(
                repo_id = DEFAULT_WD14_TAGGER_REPO,
                filename = model_name,
                subfolder = wd14_model_dir_name_in_repo,
                local_dir = download_dir,  # 下载到同一个目录下的{subfolder}文件夹内
            )


    def load_model(self, tensorrt_batch_size: Optional[int] = None) -> ort.InferenceSession:
        """强制载入模型，并赋值给self.model_in_memory

        Args:
            tensorrt_batch_size (Optional[int], optional): 当为大于0的整数时，会尝试以该batch_size调用tensorrt；如果不希望使用tensorrt，则不要输入此参数. Defaults to None.
            
        """

        model_dir = self.model_dir  # 模型项目所在文件夹
        model_name = self.model_name  # onnx模型文件名字
        model_path = self.model_path  # onnx模型文件路径

        # 载入模型
        print(f"载入新的模型： {model_name}")
        InferenceSession_time_start = time.time()

        # 设置执行者参数
        # 如果启用了tensorrt，则会将引擎缓存至 模型目录/{TRT_ENGINE_CACHE_DIR}/模型文件名(不带扩展名) 文件夹内
        trt_engine_cache_path = os.path.join(model_dir, TRT_ENGINE_CACHE_DIR, os.path.splitext(model_name)[0])
        os.makedirs(trt_engine_cache_path, exist_ok=True)  # 手动创建一个文件夹，否则会报错
        # 由set_providers()内部通过tensorrt_batch_size判断是否使用tensorrt执行者
        providers = set_providers(
            trt_engine_cache_path = trt_engine_cache_path,
            tensorrt_batch_size = tensorrt_batch_size,
        )

        model_in_memory = ort.InferenceSession(model_path, providers=providers)
        print("载入模型用时", time.time() - InferenceSession_time_start, "秒")

        # 检查是否是本项目的模型
        outputs_name_list = [x.name for x in model_in_memory.get_outputs()]
        inputs_name_list = [x.name for x in model_in_memory.get_inputs()]

        assert len(inputs_name_list) == 1, "onnx模型输入层不止一个，可能你使用的模型不是本项目的模型"
        assert inputs_name_list[0] == INPUT_NAME, f"onnx模型输入层名字不是{INPUT_NAME}，可能你使用的模型不是本项目的模型"
        assert len(outputs_name_list) == MULTI_OUTPUT_NUMBER, f"onnx模型输出层不是{MULTI_OUTPUT_NUMBER}个，可能你使用的模型不是本项目的模型"
        
        total_len = sum(len(sub_tags) for sub_tags in self.tags)
        layer0_output_shape = model_in_memory.get_outputs()[0].shape[1]
        assert total_len == layer0_output_shape, f"tags数量{total_len}和预测向量维度{layer0_output_shape}不一致"

        self.model_in_memory = model_in_memory

        return self.model_in_memory


    def inference(
        self,
        # 数据集相关
        train_data_dir: str,
        batch_size: int = 1,
        max_data_loader_n_workers: Optional[int] = None,
        recursive: Optional[bool] = None,
        # 模型推理参数
        general_threshold: float = DEFAULT_TAGGER_THRESHOLD,
        characters_threshold: float = DEFAULT_TAGGER_THRESHOLD,
        caption_extension: str = DEFAULT_TAGGER_CAPTION_EXTENSION,
        remove_underscore: Optional[bool] = None,
        rating: Optional[bool] = None,
        debug: Optional[bool] = None,
        undesired_tags: Optional[str] =  None,
        # 推理并发
        concurrent_inference: Optional[bool] = None,
        # tensorrt相关
        tensorrt_batch_size: Optional[int] = None,
    ) -> None:
        """进行图片tagger推理

        Args:
            train_data_dir (str): 需要推理的图片所在目录
            batch_size (int, optional): 推理batch_size大小. Defaults to 1.
            max_data_loader_n_workers (Optional[int], optional): Torch.DataLoader的进程数，设置成0则使用DataLoader但不使用子进程，大于0的整数则使用相应的子进程数载入数据，其余值则不使用DataLoader. Defaults to None.
            recursive (Optional[bool], optional): 递归推理子文件夹. Defaults to None.
            general_threshold (float, optional): 常规描述tag的推理阈值，越大越准确，但tag数量越少. Defaults to DEFAULT_TAGGER_THRESHOLD.
            characters_threshold (float, optional): 特定任务tag的推理阈值，越大越准确，但tag数量越少. Defaults to DEFAULT_TAGGER_THRESHOLD.
            caption_extension (str, optional): 输出的tag文件扩展名. Defaults to DEFAULT_TAGGER_CAPTION_EXTENSION.
            remove_underscore (Optional[bool], optional): 是否将tag字符串中的下划线'_'替换为空格' '. Defaults to None.
            rating (Optional[bool], optional): 是否标记上限制级tag. Defaults to None.
            debug (Optional[bool], optional): debug模式，将会在每次处理完一张图片后，立马输出标签内同. Defaults to None.
            undesired_tags (Optional[str], optional): 不希望标记的tag，以逗号分割，可以有空格 e.g "1girl, solo". Defaults to None.
            concurrent_inference (Optional[bool], optional): 载入数据的同时进行推理，建议在GPU模式下使用. Defaults to None.
            tensorrt_batch_size (Optional[int], optional): tensorrt执行者的batch_size，需要tensorrt环境的支持；设置为大于0的整数生效，会启用tensorrt执行者. Defaults to None.

        Returns:
            _type_: None
        """

        # 如果没载入过模型，就载入一次
        # 如果tensorrt_batch_size是有效值，则会启用tensorrt执行者
        if not isinstance(self.model_in_memory, ort.InferenceSession):
            self.load_model(tensorrt_batch_size)
        
        # 启用了tensorrt的话，batch_size应该被限制在tensorrt_batch_size以内，不然需要重新编译
        if isinstance(tensorrt_batch_size, int) and tensorrt_batch_size > 0 :
            inference_batch = min(batch_size, tensorrt_batch_size)
        else:
            inference_batch = batch_size

        # 获取不想要的tags列表
        undesired_tags_list = undesired_tags_str_to_list(undesired_tags)

        # 数据集载入器
        data = load_data(
            train_data_dir = train_data_dir,
            recursive = recursive,
            # 决定是否使用DataLoader
            # 注意，这里请用>=判断，因为DataLoader接受0，代表不用子进程
            use_torch_dataloader = bool(isinstance(max_data_loader_n_workers, int) and max_data_loader_n_workers >= 0),
            # DataLoader的参数，如果不用DataLoader则不起作用
            batch_size = inference_batch,
            num_workers = max_data_loader_n_workers,
        )

        def run_batch_wrapper(path_imgs: List[Tuple[str, np.ndarray]]):
            """ 用于包装run_batch，因为run_batch的参数太多了，不好传递 """
            return run_batch(
                path_imgs=path_imgs,
                ort_session=self.model_in_memory,  # type: ignore
                general_threshold=general_threshold,
                characters_threshold=characters_threshold,
                rating_tags=self.tags[0],
                general_tags=self.tags[1],
                characters_tags=self.tags[2],
                caption_extension=caption_extension,
                remove_underscore=remove_underscore,
                rating=rating,
                debug=debug,
                undesired_tags_list=undesired_tags_list,
            )
        
        def copy_wd14_tags_toml():
            """ 拷贝一份tag文件到数据集文件夹 """
            wd14_tags_toml_path = self.wd14_tags_toml_path
            wd14_tags_toml_name = os.path.basename(wd14_tags_toml_path)
            copy_wd14_tags_toml_path = os.path.join(train_data_dir, wd14_tags_toml_name)
            try:
                shutil.copy2(wd14_tags_toml_path, copy_wd14_tags_toml_path)
            except Exception as e:
                logging.warning(f"Warning! Copy tag file: {wd14_tags_toml_name} Failed\nError: {e}")
        
        # 如果使用并行推理，那么就创建线程池来管理进程
        # 设置成1就行了，主要是不让GPU推理阻塞CPU上数据集读取
        # 更快的并发应该通过调大batch_size来实现，而不是通过多线程
        # ！！！ 如果不为1，会出现争抢问题，会带来死锁；需要把ort.InferenceSession放在run_batch里 ！！！
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1) if concurrent_inference else None
        if pool is not None:
            print("并发推理已启用，将使用将使用线程池进行推理")

        try:

            copy_wd14_tags_toml()  # 拷贝一份tag文件，用于方便聚类做特征重要性分析

            pool_futures_list = []  # 用于跟踪进程完成进度
            b_imgs = []  # 起到类似队列的作用
            tqdm.write("分配进程中...")
            for data_entry in tqdm( data, smoothing=0.0, total=len(data) ):
                for data in data_entry:
                    # kohya原来就是这样写的，我也不知道为什么要再做一次判断
                    if data is None:
                        continue

                    image, image_path = data

                    # 如果是None，代表没有用DataLoader，那么就要手动处理图片
                    if image is None:
                        try:
                            image = load_image(image_path)
                        except Exception as e:
                            print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
                            continue
                    
                    # 将处理好的图片加到队列中
                    b_imgs.append((image_path, image))

                    # 一旦凑满了一个批，就进行推理
                    if len(b_imgs) >= inference_batch:
                        # 如果启用并发推理，则提交任务到线程池，避免同步任务阻塞
                        if pool is not None:
                            # 注意这里必须要存在一个新list里，不然直接用b_imgs.clear()，会出现线程还没调用，内存中b_imgs就已经被删了
                            b_imgs_batch = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
                            pool_futures_list.append( pool.submit(run_batch_wrapper, b_imgs_batch) )  # 分配任务给进程池
                        # 没有并发就正常调用
                        else:
                            run_batch_wrapper(b_imgs)
                        # 清理以释放内存
                        b_imgs.clear()

            # TODO: 对于残缺批，考虑手动补充一些空数据，这样可以避免batch变化时的巨大开销
            # 不能删掉这段，因为可能会有些小于batch_size的图片没有被推理
            if len(b_imgs) > 0:
                # 如果启用并发推理，则提交任务到线程池，避免同步任务阻塞
                if pool is not None:
                    b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
                    pool_futures_list.append( pool.submit(run_batch_wrapper, b_imgs) )  # 分配任务给进程池
                else:
                    print(f"处理余下{len(b_imgs)}张图片中...")
                    run_batch_wrapper(b_imgs) # 同步任务
                    print("处理完成")

            # 显示完成进度
            if len(pool_futures_list) > 0:  # 大于零代表启用了并发推理
                tqdm.write("Waiting for processes to finish...")
                e_num = 0
                for future in tqdm( concurrent.futures.as_completed(pool_futures_list), smoothing=0.0, total=len(pool_futures_list) ):
                    try:
                        future.result()
                    # 检查是否某个任务发送了错误而无正常返回
                    except Exception as e:
                        tqdm.write(f"Error: {e}")
                        e_num += 1
                        continue
                print(f"Error count: {e_num}")
            
            print("done!")

        finally:
            # 释放线程池
            if pool is not None:
                pool.shutdown(wait=True)


    def unload_model(self) -> None:
        """释放模型内存，方便更换模型种类或减少占用"""
        self.model_in_memory = None
        time.sleep(0.1)
        gc.collect()
