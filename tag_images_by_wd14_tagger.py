# -*- coding: UTF-8 -*-

# from https://github.com/kohya-ss/sd-scripts/blob/16e5981d3153ba02c34445089b998c5002a60abc/finetune/tag_images_by_wd14_tagger.py


import argparse
import csv
import glob
import os
from typing import List, Optional, Tuple
import gc
import time
import logging
from pathlib import Path
import multiprocessing
import concurrent.futures

# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # 设置懒惰启动，加快载入

# 把torh放在tensorflow的前面导入，让它调用cuda环境给tensorflow
import torch

from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model  # type: ignore
from tensorflow.keras.backend import clear_session  # type: ignore
from huggingface_hub import hf_hub_download
import tf2onnx
import onnxruntime as ort



# from wd14 tagger
IMAGE_SIZE = 448

# wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2/ wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2 / SmilingWolf/wd-v1-4-moat-tagger-v2
DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]

ONNX_NAME = "wd14.onnx"  # 转换的onnx模型的名字
MULTI_OUTPUT_NUMBER = 4  # 调整后输出层的数量, 注意，如果你动了这个，需要重构下面代码，包括run_batch和change_keras_to_onnx函数


def preprocess_image(image):
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

    image = image.astype(np.float32)
    return image


class ImageLoadingPrepDataset(torch.utils.data.Dataset): # type: ignore
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess_image(image)
            tensor = torch.tensor(image)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (tensor, img_path)


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch

def download_keras_model(repo_id: str, download_dir: str) -> None:
    print(f"downloading wd14 tagger model from hf_hub. id: {repo_id}")
    for file in FILES:
        hf_hub_download(repo_id, file, cache_dir=download_dir, force_download=True, force_filename=file)
    for file in SUB_DIR_FILES:
        hf_hub_download(repo_id,
                        file,
                        subfolder=SUB_DIR,
                        cache_dir=os.path.join(download_dir, SUB_DIR),
                        force_download=True,
                        force_filename=file,
        )

def change_keras_to_onnx(input_keras_model_dir, output_onnx_model_path: str) -> None:
    time_start = time.time()
    print("开始将kera模型转换成onnx格式，请耐心等待...")
    # 载入模型
    print("载入keras模型...")
    keras_model = load_model(input_keras_model_dir)
    print("载入keras模型完成")
    # 获取中间层
    layer0 = keras_model.get_layer(index=-1) # 获取最后一层的对象，predictions_sigmoid 预测层
    layer1 = keras_model.get_layer(index=-2) # 获取倒数第二层的对象
    layer2 = keras_model.get_layer(index=-3) # 获取倒数第三层的对象，predictions_norm 层
    layer3 = keras_model.get_layer(index=-4) # 获取倒数第四层的对象
    # 重构为四层输出
    multi_ouput_model = Model(inputs=keras_model.input,
                                outputs=[layer0.output,
                                        layer1.output,
                                        layer2.output,
                                        layer3.output,
                                ]
                        )
    # 构建onnx输入层
    multi_ouput_model_input = multi_ouput_model.input
    # name=multi_ouput_model_input.name  # 似乎不指定他也能自己推断出来
    spec = [tf.TensorSpec(shape = multi_ouput_model_input.shape,
                            dtype = multi_ouput_model_input.dtype,
                            name = "input_1", # 不能改变，因为后面进行tensorrt动态转换时候需要同样的名字
            ) # type: ignore
    ]
    # 转换为onnx模型
    print("开始转换为onnx模型...")
    tf2onnx.convert.from_keras(multi_ouput_model,
                               input_signature=spec,  # input_signature=None # 不指定似乎他也能推断出来
                               output_path=output_onnx_model_path,
    )
    print(f"转换完成，总共用时{time.time() - time_start}秒")

def get_tensorrt_engine(trt_engine_cache_path:str ,tensorrt_batch_size: int) -> Tuple[str, dict]:
    # 缓存在与onnx模型同目录下的'trt_engine_cache'文件夹中
    Tensorrt_options = {"trt_timing_cache_enable": True,  # 时序缓存,可以适用于多个模型
                        "trt_engine_cache_enable": True,  # 开启引擎缓存,针对特定模型、推理参数、GPU
                        "trt_engine_cache_path":trt_engine_cache_path,
                        # "trt_fp16_enable": False,  # FP16模式，需要GPU支持
                        # "trt_int8_enable": False,  # INT8模式，需要GPU支持
                        # "trt_dla_enable": False,  # DLA深度学习加速器，需要GPU支持
                        "trt_build_heuristics_enable" : True,  # 启用启发式构建，减少时间
                        "trt_builder_optimization_level": 3,  # 优化等级，越小耗时越少，但优化更差，不建议低于3
                        "trt_profile_min_shapes": "input_1:1x448x448x3",  # 最小输入形状
                        "trt_profile_max_shapes": f"input_1:{tensorrt_batch_size}x448x448x3",  # 最大输入形状
                        "trt_profile_opt_shapes": f"input_1:{tensorrt_batch_size}x448x448x3",  # 优化输入形状
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

def main(args) -> None :
    # hf_hub_downloadをそのまま使うとsymlink関係で問題があるらしいので、キャッシュディレクトリとforce_filenameを指定してなんとかする
    # depreacatedの警告が出るけどなくなったらその時
    # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/issues/22

    onnx_model_path = os.path.join(args.model_dir, ONNX_NAME)
    keras_model_sub_dir = os.path.join(args.model_dir, SUB_DIR)
    
    # 如果下自文件夹下没有{SUB_DIR}这个子目录，代表没下载过keras模型，则启动下载；如果要求强制下载，也启动下载
    if not os.path.exists(keras_model_sub_dir) or args.force_download:
        download_keras_model(args.repo_id, args.model_dir)
    # 如果onnx_model_path不存在，待变没转换过onnx模型，则启动转换；如果要求强制转换，也启动转换
    if not os.path.exists(onnx_model_path) or args.force_download:
        # 放到另一个进程里，因为tensorflow无法正常释放内存，要把进程结束了才能释放
        p = multiprocessing.Process( target=change_keras_to_onnx, args=(args.model_dir, onnx_model_path,) )
        p.start()
        p.join()
    else:
        print("using existing wd14 tagger model")

    # label_names = pd.read_csv("2022_0000_0899_6549/selected_tags.csv")
    # 依存ライブラリを増やしたくないので自力で読むよ

    with open(os.path.join(args.model_dir, CSV_FILE), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        header = l[0]  # tag_id,name,category,count
        rows = l[1:]
    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"unexpected csv format: {header}"

    general_tags = [row[1] for row in rows[1:] if row[2] == "0"]
    character_tags = [row[1] for row in rows[1:] if row[2] == "4"]

    # 画像を読み込む
    
    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]
    
    def glob_images_pathlib(dir_path, recursive):
        image_paths = []
        if recursive:
            for ext in IMAGE_EXTENSIONS:
                image_paths += list(dir_path.rglob("*" + ext))
        else:
            for ext in IMAGE_EXTENSIONS:
                image_paths += list(dir_path.glob("*" + ext))
        image_paths = list(set(image_paths))  # 重複を排除
        image_paths.sort()
        return image_paths
    
    train_data_dir_path = Path(args.train_data_dir)
    print(f"searching images in {train_data_dir_path}")
    image_paths = glob_images_pathlib(train_data_dir_path, args.recursive)

    # tag保留一份用于聚类时候进行特征重要性分析
    _all_tag_list = [ row[1] for row in rows ]  # 读取所有的tag名字
    np.savetxt( os.path.join(train_data_dir_path, "wd14_vec_tag.wd14.txt"),
                np.array(_all_tag_list),
                delimiter=',',
                fmt='%s'
    )
    
    print(f"found {len(image_paths)} images.")

    tag_freq = {}

    undesired_tags = set(args.undesired_tags.split(","))

    # 配置执行者
    providers =  ort.get_available_providers()
    if len(providers) >= 2:
        # 最后一个一般是CPU，倒数第二个一般是GPU
        providers = providers[-2:] if torch.cuda.is_available() else [ providers[-1] ]
    if args.tensorrt:
        # 加入带缓存参数的TensorRT执行者
        trt_engine_cache_path = os.path.join( os.path.dirname(onnx_model_path), "trt_engine_cache" )
        providers = [ get_tensorrt_engine(trt_engine_cache_path, args.tensorrt_batch_size) ] + providers
        print("#"*20)
        print("\n")
        print("使用TensorRT执行者,首次使用或者tensorrt_batch_side发生改变时，需要重新编译模型，耗时较久，请耐心等待，可以使用任务管理器跟踪显卡的使用")
        print("\n")
        print("#"*20)
    print("可用设备")
    for name in providers:
        print(name)
   

    # 载入模型
    InferenceSession_time_start = time.time()
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers) # type: ignore
    outputs_name_list = [x.name for x in ort_session.get_outputs()]
    inputs_name_list = [x.name for x in ort_session.get_inputs()]
    print("载入模型用时", time.time() - InferenceSession_time_start, "秒")

    assert len(inputs_name_list) == 1, "onnx模型输入层不止一个，可能你使用的模型不是本项目的模型"
    assert len(outputs_name_list) == MULTI_OUTPUT_NUMBER, f"onnx模型输出层不是{MULTI_OUTPUT_NUMBER}个，可能你使用的模型不是本项目的模型"

    def run_batch(path_imgs):
        imgs = np.array([im for _, im in path_imgs])

        ort_inputs = {inputs_name_list[0]: imgs}
        ort_out = ort_session.run(outputs_name_list, ort_inputs)

        layer0_output, layer1_output, layer2_output, layer3_output = ort_out

        probs = layer0_output  # kohya原来就是这样命名的，我懒得改下面代码

        # 写入文本
        for_index = 0  # 起到循环计数器的作用
        for (image_path, _), prob in zip(path_imgs, probs): # type: ignore
            # 最初の4つはratingなので無視する
            # # First 4 labels are actually ratings: pick one with argmax
            # ratings_names = label_names[:4]
            # rating_index = ratings_names["probs"].argmax()
            # found_rating = ratings_names[rating_index: rating_index + 1][["name", "probs"]]

            # それ以降はタグなのでconfidenceがthresholdより高いものを追加する
            # Everything else is tags: pick any where prediction confidence > threshold
            combined_tags = []
            general_tag_text = ""
            character_tag_text = ""
            for i, p in enumerate(prob[4:]):
                if i < len(general_tags) and p >= args.general_threshold:
                    tag_name = general_tags[i]
                    if args.remove_underscore and len(tag_name) > 3:  # ignore emoji tags like >_< and ^_^
                        tag_name = tag_name.replace("_", " ")

                    if tag_name not in undesired_tags:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        general_tag_text += ", " + tag_name
                        combined_tags.append(tag_name)
                elif i >= len(general_tags) and p >= args.character_threshold:
                    tag_name = character_tags[i - len(general_tags)]
                    if args.remove_underscore and len(tag_name) > 3:
                        tag_name = tag_name.replace("_", " ")

                    if tag_name not in undesired_tags:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        character_tag_text += ", " + tag_name
                        combined_tags.append(tag_name)

            # 先頭のカンマを取る
            if len(general_tag_text) > 0:
                general_tag_text = general_tag_text[2:]
            if len(character_tag_text) > 0:
                character_tag_text = character_tag_text[2:]

            tag_text = ", ".join(combined_tags)
            
            with open(os.path.splitext(image_path)[0] + args.caption_extension, "wt", encoding="utf-8") as f:
                f.write(tag_text + "\n")
                if args.debug:
                    print(f"\n{image_path}:\n  Character tags: {character_tag_text}\n  General tags: {general_tag_text}")


            # 矩阵写入同名的npz文件
            np.savez(os.path.splitext(image_path)[0] + ".wd14.npz",  # wd14用来区分kohya的潜变量cache
                    layer0=layer0_output[for_index],
                    layer1=layer1_output[for_index],
                    layer2=layer2_output[for_index],
                    layer3=layer3_output[for_index]
            )
            for_index += 1

     
    # 読み込みの高速化のためにDataLoaderを使うオプション
    if args.max_data_loader_n_workers is not None:
        print(f"启用max_data_loader_n_workers，将使用{args.max_data_loader_n_workers}个进程进行数据读取")
        dataset = ImageLoadingPrepDataset(image_paths)
        """
        # 在子进程里使用需要在import一次，我也不知道为什么，别删
        # 我现在把它删了，报错也没再次出现，很奇怪？？？
        # 或许我们需要改变上下文继承为fork？
        # import torch
        """
        data = torch.utils.data.DataLoader( # type: ignore
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    

    # 启用了tensorrt的话，batch_size应该被限制在tensorrt_batch_size以内，不然需要重新编译
    if args.tensorrt:
        inference_batch = min(args.batch_size, args.tensorrt_batch_size)
    else:
        inference_batch = args.batch_size

    # 如果使用并行推理，那么就创建线程池来管理进程
    # 设置成1就行了，主要是不让GPU推理阻塞CPU上数据集读取
    # 更快的并发应该通过调大batch_size来实现，而不是通过多线程
    # ！！！ 如果不为1，会出现争抢问题，会带来死锁；需要把ort.InferenceSession放在run_batch里 ！！！
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1) if args.concurrent_inference else None
    if pool is not None:
        print("并发推理已启用，将使用将使用线程池进行推理")

    pool_futures_list = []  # 用于跟踪进程完成进度
    b_imgs = []
    tqdm.write("分配进程中...")
    for data_entry in tqdm( data, smoothing=0.0, total=len(data) ):
        for data in data_entry:
            if data is None:
                continue

            image, image_path = data
            if image is not None:
                image = image.detach().numpy()
            else:
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = preprocess_image(image)
                except Exception as e:
                    print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
                    continue
            b_imgs.append((image_path, image))

            if len(b_imgs) >= inference_batch:
                # 如果启用并发推理，则提交任务到线程池，避免同步任务阻塞
                if pool is not None:
                    # 注意这里必须要存在一个新list里，不然直接用b_imgs.clear()，会出现线程还没调用，内存中b_imgs就已经被删了
                    b_imgs_batch = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
                    pool_futures_list.append( pool.submit(run_batch, b_imgs_batch) )  # 分配任务给进程池
                else:
                    run_batch(b_imgs)
                b_imgs.clear()

    # 不能删掉这段，因为可能会有些小于batch_size的图片没有被推理
    if len(b_imgs) > 0:
        # 如果启用并发推理，则提交任务到线程池，避免同步任务阻塞
        if pool is not None:
            b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
            pool_futures_list.append( pool.submit(run_batch, b_imgs) )  # 分配任务给进程池
        else:
            print(f"处理余下{len(b_imgs)}张图片中...")
            run_batch(b_imgs) # 同步任务
            print("处理完成")
    
    # 显示完成进度
    if len(pool_futures_list) > 0:  # 大于零代表启用了并发推理
        tqdm.write("Waiting for processes to finish...")
        e_num = 0
        for future in tqdm( concurrent.futures.as_completed(pool_futures_list), smoothing=0.0, total=len(pool_futures_list) ):
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"Error: {e}")
                e_num += 1
                continue
        print(f"Error count: {e_num}")

    if args.frequency_tags:
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        print("\nTag frequencies:")
        for tag, freq in sorted_tags:
            print(f"{tag}: {freq}")

    """不起作用
    def _release(model):
        del model
        clear_session()  # 释放模型
        try:
            torch.cuda.empty_cache() # 释放显存
        except:
            logging.warning("释放显存失败，可能是因为没有显卡")
        gc.collect()
    _release(model)
    """

    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_WD14_TAGGER_REPO,
        help="repo id for wd14 tagger on Hugging Face / Hugging Faceのwd14 taggerのリポジトリID",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="wd14_tagger_model",
        help="directory to store wd14 tagger model / wd14 taggerのモデルを格納するディレクトリ",
    )
    parser.add_argument(
        "--force_download", action="store_true", help="force downloading wd14 tagger models / wd14 taggerのモデルを再ダウンロードします"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）",
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）",
    )
    parser.add_argument("--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    parser.add_argument("--thresh", type=float, default=0.35, help="threshold of confidence to add a tag / タグを追加するか判定する閾値")
    parser.add_argument(
        "--general_threshold",
        type=float,
        default=None,
        help="threshold of confidence to add a tag for general category, same as --thresh if omitted / generalカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ",
    )
    parser.add_argument(
        "--character_threshold",
        type=float,
        default=None,
        help="threshold of confidence to add a tag for character category, same as --thres if omitted / characterカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ",
    )
    parser.add_argument("--recursive", action="store_true", help="search for images in subfolders recursively / サブフォルダを再帰的に検索する")
    parser.add_argument(
        "--remove_underscore",
        action="store_true",
        help="replace underscores with spaces in the output tags / 出力されるタグのアンダースコアをスペースに置き換える",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--undesired_tags",
        type=str,
        default="",
        help="comma-separated list of undesired tags to remove from the output / 出力から除外したいタグのカンマ区切りのリスト",
    )
    parser.add_argument("--frequency_tags", action="store_true", help="Show frequency of tags for images / 画像ごとのタグの出現頻度を表示する")
    parser.add_argument("--concurrent_inference", action="store_true",
                        help="Concurrently read dataset and inference, may increase RAM usage, recommend to use in GPU mode / 并发的进行数据集读取和模型推理，可能会增加RAM占用，建议在GPU模式下使用"
    )
    parser.add_argument("--tensorrt", action="store_true", help="Use TensorRT for inference, it is recommended to specify tensorrt_batch_size explicitly at the same time / 使用TensorRT进行推理,建议同时现式的指定tensorrt_batch_size")
    parser.add_argument("--tensorrt_batch_size", type=int, default=2,
                        help="Max batch size for TensorRT model compilation, the larger the longer the compilation time, not recommended to be greater than 4, can be different with batch_size / TensorRT模型编译时所需要支持的最大batch size，越大编译时间越长，不建议大于4； 可以与batch_size不同"
    )
    
    return parser

if __name__ == "__main__":

    start_time = time.time()
    print("WD14 Tagger 开始")

    parser = setup_parser()
    
    args = parser.parse_args()

    # スペルミスしていたオプションを復元する
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention

    if args.general_threshold is None:
        args.general_threshold = args.thresh
    if args.character_threshold is None:
        args.character_threshold = args.thresh

    main(args)
    print("time used: ", time.time() - start_time)
