$train_data_dir = "images"    # 需要处理的图片目录 | need to process images directory
# wd14 tagger 模型的 repo id, 只在moat上做过测试 | repo id for wd14 tagger on Hugging Face, only tested on moat
$repo_id = "SmilingWolf/wd-v1-4-moat-tagger-v2"
$model_dir = "wd14_tagger_model"    # wd14 tagger 模型的下载目录 | directory to download wd14 tagger model
$batch_size = 2    # 推理时的 batch size，取绝与你的显存大小，6g显存可以选8 | batch size in inference,  6g GPU memory can choose 8
$max_data_loader_n_workers = 2    # 载入数据集的线程数，2/4/8 | number of threads to load the dataset, 2/4/8
$caption_extension = ".txt"    # 标注文件的扩展名，聚类需要txt格式 | extension of caption file, clustering needs txt format
$general_threshold = 0.35    # 一般标签的阈值，对tag聚类有影响，对wd向量聚类无影响 | threshold of confidence to add a tag for general category
$character_threshold = 0.35    # 人物标签的阈值，对tag聚类有影响，对wd向量聚类无影响 | threshold of confidence to add a tag for character category

########## 不要修改 | do not edit ##########
.\venv\Scripts\activate

python tag_images_by_wd14_tagger.py `
  $train_data_dir `
  --repo_id=$repo_id `
  --model_dir=$model_dir `
  --batch_size=$batch_size `
  --max_data_loader_n_workers=$max_data_loader_n_workers `
  --caption_extension=$caption_extension `
  --general_threshold=$general_threshold `
  --character_threshold=$character_threshold `
  
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

#>