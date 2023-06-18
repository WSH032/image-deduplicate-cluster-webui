### 6 Jun.2023 2023/06/6
new:
 - 增加WebUI中WD14-tagger的UI界面，支持与`run_tagger.ps1`几乎一样的参数设置，详细请看[wd14_show_0](./docs/wd14_show_0.png)

bugfix:
 - 注释掉类型注解，以解决[1#issue](https://github.com/WSH032/image-deduplicate-cluster-webui/issues/1)

### 5 Jun.2023 2023/06/05
**因为添加tf2onnx库的相关代码，需要更新依赖，请再次运行`install.ps1`完成更新**

new:
 - 增加并发推理功能,可以在读取数据时候进行推理，建议在GPU模式下使用，详细请看`run_tagger.ps1`中相关部分
 - 增加tensorrt执行者的选项，可在大幅降低显存同时达到GPU几乎最高推理性能，详细请看`run_tagger.ps1`中相关部分
   - 需要tensorrt的支持，请阅读[docs.nvidia](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip)
     - 如果你不会安装，下个版本将添加自动化安装tensorrt脚本的支持，请等待
   - 首次使用需要进行一段时间的编译，请耐心等待
 - 关于并发推理和tensorrt的测试结果请看[tensorrt_speed_experiment.md](./docs/tensorrt/tensorrt_speed_experiment.md)
 - 增加`run_tagger.ps1`中更多tagger参数的设置
 - 因为高batch情况下可能会有显存溢出的问题，现在`run_tagger.ps1`中默认推理的batch为1，请自行按需修改
   - 合理的batch，达到75%的显存占用率时可以达到最大推理速度
   - 聚类WebUI中tagger暂不支持设置推理参数，下个版本添加该功能支持
 - 更新脚本由`update.bat`更改为`update.ps1`

bugfix:
 - 修改`gradio >= 3.31.0` 以解决类型注解的问题[1#issue](https://github.com/WSH032/image-deduplicate-cluster-webui/issues/1)
