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
	- [ ]**计划使用WD14生成特征向量toml文件，后续将直接从文件中读取特征，避免聚类时多次运行模型**
	- [ ] 增加释放模型功能
	- [x] 增加使用倒数第三层模型
- 聚类
	- [ ] 为图片聚类增加SVD降维
	- [ ] 增加tags文本字符串预处理（如将空格变为逗号）
	- [ ] 为聚类文件夹增加标签
	- [ ] 修正特征重要性分析的标签错误(应用占比50判断)
	- [ ] 增加手动选择是否应用某个聚类功能
	- [x] 增加更多分析方法
	- [x] 更多聚类方式
		- [ ] 为不同聚类方式增加参数选择功能
		- [ ] 将聚类方法选择独立出来
- 查重
	- [x] 为查重添加选择全部选项
	- [ ] 为查重添加更多查重方式和查重阈值
	- [x] 重写查重启发式选择算法
	- [ ] 为查重添加移动图片功能
	- [ ] 为查重删除添加删除tag文本功能


## 部分展示
![deduplicate_demo](./docs/deduplicate_demo.png)

![cluster_demo_0](./docs/cluster_demo_0.png)

![cluster_demo_1](./docs/cluster_demo_1.png)
