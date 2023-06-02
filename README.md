# image-deduplicate-cluster-webui
 A WebUI script that deduplicates images or clusters them by tags.  一个用于图像查重和基于tags聚类的WebUI脚本

## 现在我们有什么？
 - 基于imagededup库，进行图片去重的WebUI
 - 基于sklearn库，以tags为特征的图片聚类WebUI

## Todo

- [ ] 在Colab上部署
- [x] 完成本地部署封装
- [ ] 完成A111-SD-WebUI部署
- [x] 增加WD14提取特征和生成tags文本
	- [x] 使用WD14生成特征向量到npz文件，避免多次聚类时重复运行耗时
	- [ ] 增加释放模型功能
	- [x] 增加使用倒数第三层模型
- 聚类
	- [ ] 为图片聚类增加SVD降维
	- [o] 增加tags文本字符串预处理（如将空格变为逗号）
	  - 已经取消，因为Gelbooru-API-Downloader已经自带了这个功能
	- [ ] 为聚类文件夹增加标签
	- [ ] 修正特征重要性分析的标签错误(应用占比50判断)
	- [ ] 增加手动选择是否应用某个聚类功能
	- [x] 增加更多分析方法
	  - 现在有轮廓系数和Davids系数
	- [x] 更多聚类方式
	  - 现在有kmeans，谱聚类、层次聚类，OPTICS聚类
	  - [ ] 为不同聚类方式增加参数选择功能
	  - [o] 将聚类方法选择独立出来
	    - 暂时取消,因为使用npz文件后，读取特征向量已经很快了
- 查重
	- [x] 为查重添加选择全部选项
	- [ ] 为查重添加更多查重方式和查重阈值
	- [x] 重写查重启发式选择算法
	- [ ] 为查重添加移动图片功能
	- [ ] 为查重删除添加删除tag文本功能

## 寻求帮助

**tensorflow，在gradio中使用wd14无法释放显存**
**现已经用multiprocesing解决了，但是有几个疑惑**

 - 为什么用concurrent.futures.ProcessPoolExecutor创建子进程也无法释放？
   - 会不会是multiprocesing不需要和父进程通信，意味着结束后完全无引用，所以自动释放了
   - 会不会是主进程最开头中导入了tag_images_by_wd14_tagger，造成的显存资源共享，因此WD结束后主进程仍然保存着显存的引用？
 - 如何增加保留模型在显存中以进行快速下一次推理功能
   - **计划使用onnxruntime实现**
 - 考虑是不是torch.tensor的占用问题
   - 基本确定是tensorflow无法释放
 - 为什么tag_images_by_wd14_tagger.main里要再一次import torch
   - 我又把main里的torch删了，问题却没出现？为什么？
 - 考虑删除model和tensor，配合gc.collect()回收
 - 考虑tensorflow的sesson

 **考虑使用onnx加载模型，两个方案**
  - [ ] 先把keras模型下载，调整输出后再本地导出onnx格式
    - **采纳此方案**
    - 在colab中实验时观察到
	  - 对于onnx，似乎用for循环和一次性批量输入相差时间不大
	  - 对于tf，for循环比批量输入要慢的多
	  - 对于批量，tf和onnx似乎时间差不多
  - [x] 直接下载onnx模型，用onnx库来调输出层（似乎层不一样）
    - 已经实现，但仍有问题
	  - 很难使用多进程
	  - 多线程带来的一点提升似乎是在写入文件的IO上，因为GPU已经跑满了，当然也有可能是载入显存能更快进行推算，而减小CPU到GPU的通信
	  - 虽然启动比tensorflow快，但在超大数据集（应该要大于400张）的并发表现不如tf
  - 我把调整好的onnx上传到huggingface

 **考虑使用torch加载模型**

## 部分展示
![deduplicate_demo](./docs/deduplicate_demo.png)

![cluster_demo_0](./docs/cluster_demo_0.png)

![cluster_demo_1](./docs/cluster_demo_1.png)
