$train_data_dir = "images"
$repo_id = "SmilingWolf/wd-v1-4-moat-tagger-v2"
$model_dir = "wd14_tagger_model"
$batch_size = 8
$max_data_loader_n_workers = 2
$caption_extension = ".txt"
$general_threshold = 0.35
$character_threshold = 0.35


<# 


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
    

    parser.add_argument("--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    
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

#>

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