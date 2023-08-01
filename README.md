# image-deduplicate-cluster-webui
 A `A1111-WebUI` extension for deduplicates images or cluster images. 

 一个用于图像查重和图片聚类的 `A1111-WebUI` 扩展

## 现在我们有什么？
 - 基于imagededup库，进行图片去重的WebUI
 - 基于sklearn库，以tags文本为特征，或者以WD14 tagger模型提取的特征向量进行图片聚类WebUI
 - 高效的 WD14 图片标记脚本及WebUI

## 部分展示
### 查重演示

<details>
<summary>旧版查重演示</summary>

![deduplicate_demo](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/093198c5-89b3-4f88-8ccc-4412e662cf35)

</details>

![deduplicate_demo](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/fc1774cd-c705-4e6b-b4cd-07ccd543e2dc)

### 聚类演示
![images_cluster_show_0](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/213b1370-80b5-4c41-8e49-0c8003acbd62)

### WD14-Tagger 演示
![wd14_show_1](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/29b52bf7-316d-4bb6-850d-e40e1caa3afc)

### 其他
**另一张复杂情况的8聚类效果请看[这里`images_cluster_show_1`](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/1b7ecffb-14a8-492c-a629-e4d93850e54c)**

[聚类-Tagger说明 `title_show`](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/4dab7fcf-3048-4cce-8441-347d8a3c2a68)

## Credit
 - 我不训练模型，WD14模型来自于这个项目[SmilingWolf/WD14](https://huggingface.co/SmilingWolf)
 - 聚类方法和特征提取来着于sklearn库
 - 查重方法来自于imagededup库
 - tag_images_by_wd14_tagger来自[kohya/sd-scripts](https://github.com/kohya-ss/sd-scripts/blob/main/finetune/tag_images_by_wd14_tagger.py)

## Change History
如果你不会使用git命令，可以运行`update.ps1`完成更新，并再次运行`install.ps1`来更新依赖环境

### V 2.0.0
01 Aug.2023 2023/08/01
#### New Features:
- **[破坏性]** 大更新，优化并调整几乎全部WebUI部分代码结构、组织
  - 所有ui界面和几乎所有后端函数被移至 `./ui` 文件夹下
- 为聚类增加更多参数
  - 支持自定义的 `tag文件`、`npz矩阵文件` 扩展名选择
  - 增加SVD降维功能
  - 聚类分析中增加样本分布可视化图，帮助确认n值
  - wd14聚类支持更多参数
    - 可选择 `压缩层` 或 `全向量` 层做为特征向量
	- 在 `全向量` 模式下，可以自定义选择部分类别特征
	  - 您可以在[https://huggingface.co/WSH032/wd-v1-4-tagger-feature-extractor/blob/main/wd14_tags.toml](https://huggingface.co/WSH032/wd-v1-4-tagger-feature-extractor/blob/main/wd14_tags.toml)查看各个标签种类所包含的tags
	  - `rating` 依据 nsfw 进行聚类
	  - `general` 依据图片标签tag进行聚类
	  - `characters` 依据booru上已知的角色tag进行聚类
- 为查重增加更多参数
  - 增加 `js` 的支持
    - 可通过键盘快捷键 `Enter` 和 `Backspace` 来进行选择和取消
  - 支持 `CNN模型` 查重
    - 如果有GPU的支持，CNN模型查重将会比原来的 `Hash查重` 更快
  - 增加查重阈值设置
  - 查重结果新增 `移动`，`重命名` 选项
    - 现在查重结果的操作将采用和聚类一样的操作方式
	- 增加附带文件处理功能，可以一起 `移动`，`重命名`，`删除` 附带文件
- 重构 `WD14-tagger` 的 `tag_images_by_wd14_tagger` 为类
  - 将直接下载huggingface上编译好的onnx模型，不再进行编译
  - 删除 `tensorflow`, `onnx`, `tf2onnx` 等和编译模型有关的部分
  - 增加离线模式选项
  - 增加 `wd-v1-4-convnextv2-tagger-v2` 模型
    - 除此，还有先前的 `wd-v1-4-moat-tagger-v2` 模型可以选择
  - 暂时弃用并删除命令行WD14-tagger脚本 `run_tagger.ps1`
    - 可能会在未来回归
- 不再支持只安装部分模块，默认安装全部依赖
#### Other Changes:
- 优化ui显示
- A1111-WebUI中现在`聚类-WD14`和`查重`被分为两个选项卡
  - `聚类-WD14` 对应 `Cluster-Tagger`
  - `查重` 对应 `Deduplicate`
- 删除 `./docs` 中所有展示图片，更换为 `github/assets` 链接以减小仓库体积
  - 使用 `git clone --depth=1` 时，将不再下载多余图片，以方便快速克隆
- 调整部分logging日志等级
- 在使用WD14-Tagger时，对于重名但扩展名不同的图片，其的标签文本将会被相互覆盖，此时会发出警告
#### Bug Fixes:
- 为所有后端函数添加异常处理装饰器，以避免未知错误造成 A1111-WebUI 中永久无返回结果
- 当缓存图片失败时，将自动退回到原图模式
- 修正特征重要性分析算法中的错误
  - 现在，在某个聚类结果中，只有某个特征tag占比超过或低于设置的阈值时，才会被判断为 `prompt` 和 `negetive`


### 以前的更新内容请查看[CHANGELOG.md](CHANGELOG.md)

## 安装 Install

### Colab使用
[![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://colab.research.google.com/github/WSH032/sd-webui-fast-dataset-maker/blob/main/fast_dataset_maker.ipynb)

### （一）做为[AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)的扩展使用

**请尽量在`gradio>=3.31.0`的版本下使用，此扩展在 2023年第22周 [`SD-WebUI v1.3.1`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/b6af0a3809ea869fb180633f9affcae4b199ffcf)下经过测试**

`https://github.com/WSH032/kohya-config-webui.git`

将这个仓库连接复制到SD-WebUi的`扩展 extensions`->`从网址安装 Install from URL`界面下载完成后，**重启SD-WebUI**即可，会自动安装所需依赖

 - 中国区用户可尝试用`https://ghproxy.com/https://github.com/WSH032/kohya-config-webui.git`代理加速下载
 - 在SD-WebUI中使用本扩展的`WD14tagger`功能时，若出现**模型下载失败**情况，可以按照本扩展WebUI内的指示从`https://huggingface.co/WSH032/wd-v1-4-tagger-feature-extractor`手动下载模型并放入WebUI内指定的文件夹中
 - tensorrt自动增量脚本[utils/run_install_tensorrt_lib.ps1](utils/run_install_tensorrt_lib.ps1)在此仓库做为扩展使用时不被支持，但你仍可以自行手动安装tensorrt环境来使用加速功能
![Install-SD-WebUI-extension](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/0b5d628f-91f4-471f-bf0f-e56965193446)

 #### 关于依赖环境
 做为SD-WebUI扩展使用时，会**自动安装和使用SD-WebUI的依赖环境**，**本项目的三个模块都会被安装**,不需要任何操作

 首次安装时，中国区用户可能会因网络原因安装较久，请耐心等待

### （二）独立使用
 <blockquote> 

**[2323/07/04] 在独立使用时，强烈建议使用此整合项目[WSH032/sd-webui-fast-dataset-maker](https://github.com/WSH032/sd-webui-fast-dataset-maker)，以获得更好的体验**

图示为整合了[zanllp/sd-webui-infinite-image-browsing](https://github.com/zanllp/sd-webui-infinite-image-browsing)
![sd-webui-infinite-image-browsing](https://github.com/WSH032/sd-webui-fast-dataset-maker/assets/126865849/b98a293f-a3a0-4d59-a997-cae86e7f25b4)

 </blockquote> 

**在python3.9下完成编写，可能之前版本也可以使用，不保证**

克隆本仓库
```shell
git clone https://github.com/WSH032/image-deduplicate-cluster-webui.git  # --depth=1
```

每个模块所需的依赖已经写在[requirements.txt](requirements.txt)

**一键运行`install.ps1`即可**

<details>
<summary>旧版 <code>install.ps1</code> 界面</summary>

![install](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/eacb082e-55e1-40c6-b5a4-cea1a6a202fb)

</details>

![install](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/bf421b54-5547-4426-a018-03b46199cfb6)

### 关于 `Torch` 和 `CUDA` 环境
`requirements.txt`中并没指定 `Torch` 的版本，请根据你的需求自行决定并修改
> 推荐安装带有cuda依赖的torch版本
> ```shell
> torch==2.0.0+cu118  --extra-index-url https://download.pytorch.org/whl/cu118
> ```

涉及到深度学习模型的部分，可以进行CPU或者GPU的推理，其中GPU的推理速度快，但是要求 `CUDA` 环境

运行`install.ps1`时会提问你是否需要安装 `Torch==2.0.0 + cuda118`，其能够在虚拟环境中配置 `CUDA` 环境

如果你配置过系统级的 `CUDA` 环境，或者你不需要使用GPU进行推理，可以选择否

### 关于WD14模型下载失败
请按照本项目 `Tagger` WebUI 中的 `模型下载说明` 选项卡指示进行操作
![model_download_info](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/93221e33-06bf-49ff-97d9-c712a362776d)

## 使用Tips
### SD-WebUI内使用演示
与独立使用方法相同
 - [SD-WebUI 查重演示](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/a4ad9926-8ee4-47d9-9f5a-bd2a3515563f)
 - [SD-WebUI 聚类演示](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/f62fb1a6-88a6-4aad-8071-16b2f22dd248)
 - [SD-WebUI WD14演示](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/0dc9ed5e-8d6b-4f3c-b383-620ceaf7d715)

### 图片查重
图片查重不依赖任何tag文本或者WD14模型

一键运行`run_deduplicate_images.ps1`，或在 `A1111-WebUI` 中以扩展形式使用，其将会生成一个WebUI，进行操作即可

![deduplicate_tips](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/fb37e242-ecc2-4847-99e0-433e775951c3)

 ![deduplicate_demo](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/fc1774cd-c705-4e6b-b4cd-07ccd543e2dc)


### 图片聚类
一键运行`run_cluster_images.ps1`，或在 `A1111-WebUI` 中以扩展形式使用，其将会生成一个WebUI，进行操作即可

图片聚类依赖与图片同名的 `txt文本` **或者** `npz文件` 进行聚类，这取决于你在WebUI中选择的特征提取方式

![vectorize_method](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/b0f81098-3de0-4924-88cd-9e8f673f32f1)

#### 选择 `tf-idf` 或者 `countvectorizer` 提取特征，则需要 `txt文本`
其中txt内容为与图片对应的booru风格的tag标签，例如
`1girl, solo, yellow eyes`

![image_with_tag](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/499e7e19-4eb2-4bd3-8337-0cb9168faa76)

#### `wd14` 提取特征，则需要 `npz文件`
其中 `npz文件` 储存着WD14模型提取的特征向量矩阵，其的生成**必须**使用本项目自带的 `tag_images_by_wd14_tagger.py` ，或者在 `WD14-Tagger` WebUI中生成


![image_witg_npz](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/a83eb86c-709d-416f-826f-eb6478f60de7)

### WD14 tagger
本项目自带的WD14 tagger模型来自于[SmilingWolf/WD14](https://huggingface.co/SmilingWolf)，我对其结构进行了调整，增加了后四层的输出，并采取倒数第三层做为特征向量

你可以使用`tag_images_by_wd14_tagger.py`进行图片打标，获取txt文本，这与[toriato/stable-diffusion-webui-wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger)的打标结果并无太大差异


同时其会输出同名的npz文件，其中包含了WD14模型的倒数前四层的输出，你可以在聚类模块中使用

~~[run_tagger](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/9681a7d1-b66e-4ac2-a47e-dc443145cd90)~~(弃用)

**2023/06/6，新增WD14的WebUI界面**

![wd14_show_1](https://github.com/WSH032/image-deduplicate-cluster-webui/assets/126865849/ff6ef49c-e5a4-4cc7-b5c7-24357e4ffa83)

## Todo

- [x] 在Colab上部署
- [x] 完成本地部署封装
- [x] 完成A111-SD-WebUI部署
- [x] 增加WD14提取特征和生成tags文本
	- [x] 使用WD14生成特征向量到npz文件，避免多次聚类时重复运行耗时
	- [x] 增加释放模型功能
	- [x] 增加使用倒数第三层模型
- 聚类
	- [x] 为图片聚类增加SVD降维
	- ~~[ ] 增加tags文本字符串预处理（如将空格变为逗号）~~
	  - 已经取消，因为Gelbooru-API-Downloader已经自带了这个功能
	- [ ] 为聚类文件夹增加标签
	- [x] 修正特征重要性分析的标签错误(应用占比50判断)
	- [ ] 增加手动选择是否应用某个聚类的功能
	- [x] 增加更多分析方法
	  - 现在有轮廓系数和Davids系数
	- [x] 更多聚类方式
	  - 现在有kmeans，谱聚类、层次聚类，OPTICS聚类
	  - [ ] 为不同聚类方式增加参数选择功能
	  - ~~[ ] 将聚类方法选择独立出来~~
	    - 已经取消,因为使用npz文件后，读取特征向量已经很快了
- 查重
	- [x] 为查重添加选择全部选项
	- [x] 为查重添加更多查重方式和查重阈值
	- [x] 重写查重启发式选择算法
	- [x] 为查重添加移动图片功能
	- [x] 为查重删除添加删除tag文本功能

## 结尾

 **考虑使用onnx加载模型，两个方案**
  - [x] 先把keras模型下载，调整输出后再本地导出onnx格式
  - ~~[ ] 直接下载onnx模型，用onnx库来调输出层~~（似乎层不一样）
