# image-deduplicate-cluster-webui
 A WebUI script that deduplicates images or clusters them by tags.  一个用于图像查重和基于tags聚类的WebUI脚本

## 现在我们有什么？
 - 基于imagededup库，进行图片去重的WebUI
 - 基于sklearn库，以tags为特征的图片聚类WebUI

## Todo
- [ ] 在Colab上部署
- [x] 完成本地部署封装
- [ ] 完成A111-SD-WebUI部署
- [ ] 增加WD14提取特征和生成tags文本
- 聚类
	- [ ] 为图片聚类增加SVD降维，更多聚类方式与参数
	- [ ] 增加tags文本字符串预处理（如将空格变为逗号）
	- [ ] 为聚类文件夹增加标签
	- [ ] 增加更多分析方法
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
