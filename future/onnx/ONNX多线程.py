# -*- coding: UTF-8 -*-

# from https://github.com/kohya-ss/sd-scripts/blob/16e5981d3153ba02c34445089b998c5002a60abc/finetune/tag_images_by_wd14_tagger.py


import argparse
import csv
import glob
import os
from typing import List, Optional, Tuple
import gc

# 把torh放在tensorflow的前面导入，让它调用cuda环境给tensorflow
import torch

from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
from huggingface_hub import hf_hub_download
from pathlib import Path
import logging

import onnx
import onnxruntime as rt


import time

# from wd14 tagger
IMAGE_SIZE = 448

# wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2/ wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2 / SmilingWolf/wd-v1-4-moat-tagger-v2
DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]


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

def main(args) -> None :
    # hf_hub_downloadをそのまま使うとsymlink関係で問題があるらしいので、キャッシュディレクトリとforce_filenameを指定してなんとかする
    # depreacatedの警告が出るけどなくなったらその時
    # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/issues/22
    if not os.path.exists(args.model_dir) or args.force_download:
        print(f"downloading wd14 tagger model from hf_hub. id: {args.repo_id}")
        for file in FILES:
            hf_hub_download(args.repo_id, file, cache_dir=args.model_dir, force_download=True, force_filename=file)
        for file in SUB_DIR_FILES:
            hf_hub_download(
                args.repo_id,
                file,
                subfolder=SUB_DIR,
                cache_dir=os.path.join(args.model_dir, SUB_DIR),
                force_download=True,
                force_filename=file,
            )
    else:
        print("using existing wd14 tagger model")

    # 画像を読み込む
    model = onnx.load( os.path.join(args.model_dir, "model.onnx") )
    
    if model is None:
        raise ValueError("model is None")

    # 用于获取两个层的输出，方便聚类使用倒数第三层的数据
    for node in model.graph.node[-2:-5:-1]:  # type: ignore
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])  # type: ignore

    # 输出参数shape
    # print(layer0.output_shape)
    # print(layer3.output_shape)

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

    """
    已经弃用，因为不再考虑通过import来进行numpy矩阵传递
    # 如果传入了force_images_list，就按这个列表处理指定的图片
    try:
        image_paths = args.force_images_list
        print("正在进行聚类分析，不再处理tags文本")
    except:
        pass
    """
    
    print(f"found {len(image_paths)} images.")

    tag_freq = {}

    undesired_tags = set(args.undesired_tags.split(","))

    # 载入模型
    InferenceSession_time_start = time.time()
    providers =  rt.get_available_providers()
    providers = providers if torch.cuda.is_available() else [ providers[-1] ]  # type: ignore
    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("可用设备",providers)
    ort_session = rt.InferenceSession(model.SerializeToString(), providers=providers)  # type: ignore
    outputs = [x.name for x in ort_session.get_outputs()]
    print("载入模型用时", time.time() - InferenceSession_time_start, "秒")
    

    def run_batch(path_imgs):

        imgs = np.array([im for _, im in path_imgs])

        layer0_output = []
        layer1_output = []
        layer2_output = []
        layer3_output = []

        # onnx里面只能拆成单个单个样本逐个推理
        for i in imgs:
            i = np.expand_dims(i, axis=0)
            ort_inputs = {ort_session.get_inputs()[0].name: i,}
            ort_out = ort_session.run(outputs, ort_inputs)

            layer0_output.append(ort_out[0])
            layer1_output.append(ort_out[1])
            layer2_output.append(ort_out[2])
            layer3_output.append(ort_out[3])
        
        layer0_output = np.vstack(layer0_output)
        layer1_output = np.vstack(layer1_output)
        layer2_output = np.vstack(layer2_output)
        layer3_output = np.vstack(layer3_output)

        probs = layer0_output  # kohya原来就是这样命名的，我懒得改下面代码
        
        """
        layer0_output_list.extend( layer0_output.numpy() )
        layer1_output_list.extend( layer1_output.numpy() )
        layer2_output_list.extend( layer2_output.numpy() )
        layer3_output_list.extend( layer3_output.numpy() )
        """

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

            """
            我们考虑用numpy的npz文件来储存，读取会更快
            # 将写入同名的toml文本
            with open(os.path.splitext(image_path)[0] + ".toml", "wt", encoding="utf-8") as f:
                layer_output_dict = {"layer0": layer0_output[for_index].tolist(),
                                        "layer1": layer1_output[for_index].tolist(),
                                        "layer2": layer2_output[for_index].tolist(),
                                        "layer3": layer3_output[for_index].tolist(),          
                }
                # 把layer_output_dict以toml格式写入文件
                toml.dump(layer_output_dict, f)
            """

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
        dataset = ImageLoadingPrepDataset(image_paths)
        import torch  # 在子进程里使用需要在import一次，我也不知道为什么，别删
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

    import concurrent.futures
    # 只能用线程池，进程池无法正确传递模型和无法被pickable
    # 如果要用进程池，需要把run_batch移到main之外，然后每个进程单独建一个onnxruntime.session
    # 但是速度慢的瓶颈不在CPU，而在GPU，所以没必要？
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pool_futures_list = []  # 用于跟踪进程完成进度
        b_imgs = []  # 用于暂时存放待生成batch时候的数据
        tqdm.write("分配进程中...")
        for data_entry in tqdm(data, smoothing=0.0):
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

                if len(b_imgs) >= args.batch_size:
                    # 注意这里必须要存在一个新list里，不然直接用b_imgs.clear()，会出现线程还没调用，内存中b_imgs就已经被删了
                    b_imgs_batch = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
                    pool_futures_list.append( pool.submit(run_batch, b_imgs_batch) )  # 分配任务给进程池
                    # run_batch(b_imgs) # 同步任务
                    b_imgs.clear()

        if len(b_imgs) > 0:
            b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
            pool_futures_list.append( pool.submit(run_batch, b_imgs) )  # 分配任务给进程池
            # run_batch(b_imgs) # 同步任务

        # 显示完成进度
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
    
    """
    已经弃用，因为现在改用npz文件来进行numpy矩阵传递
    # 结构为List[ List[np.array, np.array], List[str] ]
    # List[np.array, np.array]
    #   第一个元素为一个矩阵，(样本图片数量 * 所选的开头层参数长度 )
    #   第一个元素为一个矩阵，(样本图片数量 * 所选的结尾层参数长度 )
    # List[str]
    #   里面的元素为从selected_tags.csv读取的tags，包含了分级tag
    ### 请注意，这里以一个元素请保证为List，因为后续聚类已经进行了硬编码 ###
    # return [ [ np.array(layer1_output_list), np.array(layer2_output_list) ],  [ row[1] for row in rows ] ]
    """

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
    
    """
    已经弃用，因为不再通过函数传递numpy矩阵，而是改为通过npz文件传递
    # 注意，这个参数不能添加进去，因为代码里面是通过判断其存不存在来决定是否在进行聚类，而不是看其是否为None
    # parser.add_argument("--force_images_list", type=str, default=None, help="指定要处理的图片列表，将会按顺序处理")  # 有bug
    """
    return parser


def WD14tagger(train_data_dir: str) -> None :
    
    train_data_dir = train_data_dir
    repo_id = "SmilingWolf/wd-v1-4-moat-tagger-v2"
    model_dir = r"wd14_tagger_model"
    batch_size = 8
    max_data_loader_n_workers = 4
    caption_extension = ".txt"
    general_threshold = 0.35
    character_threshold = 0.35
    
    cmd_params_list = [train_data_dir,
                        f"--repo_id={repo_id}",
                        f"--model_dir={model_dir}",
                        f"--batch_size={batch_size}",
                        f"--max_data_loader_n_workers={max_data_loader_n_workers}",
                        f"--caption_extension={caption_extension}",
                        f"--general_threshold={general_threshold}",
                        f"--character_threshold={character_threshold}",
    ]
    
    parser = setup_parser()
    args = parser.parse_args( cmd_params_list )
    main(args)

if __name__ == "__main__":
    start_time = time.time()
    WD14tagger(r"E:\GitHub\图片\宫子和会长")
    print("耗时：", time.time() - start_time, "秒")
