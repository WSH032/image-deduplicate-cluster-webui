###### 基础功能 ######
$train_data_dir = "images"    # 需要处理的图片目录 | need to process images directory
$repo_id = "SmilingWolf/wd-v1-4-moat-tagger-v2"    # wd14 tagger 模型的 repo id, 只在moat上做过测试 | repo id for wd14 tagger on Hugging Face, only tested on moat
$model_dir = "wd14_tagger_model"    # wd14 tagger 模型的下载目录 | directory to download wd14 tagger model

# 选取合适的batch_size和max_data_loader_n_workers，使显存占用率在80%，可以获取最高的速度 |
# batch_size and max_data_loader_n_workers should be set to make GPU memory usage at 80% to get the highest speed
$batch_size = 1    # GPU # 推理时的 batch size，取绝与你的显存大小，6g显存可以选4，4g显存选2 | batch size in inference,  6g GPU memory can choose 4, 4g memory can choose 2
$max_data_loader_n_workers = 2    # CPU # 设置成0则不启用，载入数据集的线程数，2/4 | set to 0 to disable, number of threads to load dataset, 2/4

$caption_extension = ".txt"    # 标注文件的扩展名，聚类需要txt格式 | extension of caption file, clustering needs txt format
$general_threshold = 0.35    # 一般标签的阈值，对tag聚类有影响，对wd向量聚类无影响 | threshold of confidence to add a tag for general category
$character_threshold = 0.35    # 人物标签的阈值，对tag聚类有影响，对wd向量聚类无影响 | threshold of confidence to add a tag for character category

$undesired_tags = ""    # 不需要的标签，用英文逗号分割，不要有空格 e.g. "tag1,tag2" | undesired tags, split by comma, no space. e.g. "tag1,tag2"
$remove_underscore = 0    # 是否移除tags中的下划线 | whether to remove underscore in tags
$recursive = 0    # 是否处理子文件夹 | whether to process subfolders
$force_download = 0    # 是否强制重新下载模型 | whether to force to download model again


###### 实验性功能 ######

<#
############
可以运行 .\utils\run_install_tensorrt_lib.ps1 完成tensorrt的安装
专为 TensorRT 8.6 GA for Windows 10 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7 and 11.8 编写
不满足Win10条件的请阅读 https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html 完成安装
############
#>

# 是否在读取数据的时候同时进行推理，开启可能会加快速度，但是会增加RAM内存占用，建议在GPU模式下使用 |
# whether to perform inference while reading data, which may speed up the process, but will increase the RAM memory usage, it is recommended to use in GPU mode
$concurrent_inference = 0

# 是否使用tensorrt加速，需要tensorrt环境的支持，首次使用会自动进行一段时间的模型编译
# TensorRT能在达到最大速度的情况下，显著减小显存占用; 最大速度可能受限于CPU的瓶颈而不是GPU |
# whether to use tensorrt to accelerate, it requires tensorrt environment support, the model will be compiled automatically for a period of time when used for the first time
# TensorRT can significantly reduce GPU memory usage while achieving maximum speed, it is recommended to set batch_size to be greater than or equal to tensorrt_batch_size when using
$tensorrt = 0

# tensorrt编译出的模型所能支持的最大tensorrt推理batch_size，可以和上面那个batch_size不同，上面那个batch会自动分割来处理；设置越大编译时间越长，设置成4会需要5分钟的编译时间，不建议设置得更大 |
# the maximum TensorRT inference batch size supported by the model compiled by tensorrt, which can be different from batch_size, it will be automatically split to process;
# the larger the setting, the longer the compilation time, 
# it will take 5 minutes to compile if set to 4, it is not recommended to set it larger
$tensorrt_batch_size = 2



########## 不要修改 | do not edit ##########
.\venv\Scripts\activate

$ext_args = [System.Collections.ArrayList]::new()

if ($max_data_loader_n_workers) {
  [void]$ext_args.Add("--max_data_loader_n_workers=$max_data_loader_n_workers")
}
if ($undesired_tags) {
  [void]$ext_args.Add("--undesired_tags=$undesired_tags")
}
if ($remove_underscore) {
  [void]$ext_args.Add("--remove_underscore")
}
if ($force_download) {
  [void]$ext_args.Add("--force_download")
}
if ($recursive) {
  [void]$ext_args.Add("--recursive")
}

# 实验性功能
if ($concurrent_inference) {
  [void]$ext_args.Add("--concurrent_inference")
}
if ($tensorrt) {
  [void]$ext_args.Add("--tensorrt")
  [void]$ext_args.Add("--tensorrt_batch_size=$tensorrt_batch_size")
}

python tag_images_by_wd14_tagger.py `
  $train_data_dir `
  --repo_id=$repo_id `
  --model_dir=$model_dir `
  --batch_size=$batch_size `
  --caption_extension=$caption_extension `
  --general_threshold=$general_threshold `
  --character_threshold=$character_threshold `
  $ext_args
  
pause

########## 更多参数 | more parameters ##########

<#

    "train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")

    "--repo_id", type=str, default=DEFAULT_WD14_TAGGER_REPO, help="repo id for wd14 tagger on Hugging Face / Hugging Faceのwd14 taggerのリポジトリID")
    
    "--model_dir", type=str, default="wd14_tagger_model", help="directory to store wd14 tagger model / wd14 taggerのモデルを格納するディレクトリ",)
    
    "--force_download", action="store_true", help="force downloading wd14 tagger models / wd14 taggerのモデルを再ダウンロードします")
    
    "--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    
    "--max_data_loader_n_workers", type=int, default=None, help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）",)
    
    "--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    
    "--general_threshold", type=float, default=None, help="threshold of confidence to add a tag for general category, same as --thresh if omitted / generalカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ",)
    
    "--character_threshold", type=float, default=None, help="threshold of confidence to add a tag for character category, same as --thres if omitted / characterカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ",)
    
    "--recursive", action="store_true", help="search for images in subfolders recursively / サブフォルダを再帰的に検索する")
    
    "--remove_underscore", action="store_true", help="replace underscores with spaces in the output tags / 出力されるタグのアンダースコアをスペースに置き換える",)
    
    "--debug", action="store_true", help="debug mode")
    
    "--undesired_tags", type=str, default="", help="comma-separated list of undesired tags to remove from the output / 出力から除外したいタグのカンマ区切りのリスト",)
    
    "--frequency_tags", action="store_true", help="Show frequency of tags for images / 画像ごとのタグの出現頻度を表示する")
    
    "--concurrent_inference", action="store_true", help="Concurrently read dataset and inference, may increase RAM usage, recommend to use in GPU mode / 并发的进行数据集读取和模型推理，可能会增加RAM占用，建议在GPU模式下使用")
    
    "--tensorrt", action="store_true", help="Use TensorRT for inference, it is recommended to specify tensorrt_batch_size explicitly at the same time / 使用TensorRT进行推理,建议同时现式的指定tensorrt_batch_size")
    
    "--tensorrt_batch_size", type=int, default=2, help="Max batch size for TensorRT model compilation, the larger the longer the compilation time, not recommended to be greater than 4, can be different with batch_size / TensorRT模型编译时所需要支持的最大batch size，越大编译时间越长，不建议大于4； 可以与batch_size不同")

#>
