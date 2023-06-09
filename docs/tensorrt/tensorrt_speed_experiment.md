| batch | 同步推理 | 并发推理 | 同步推理+tensorrt | 并发推理+tensorrt |
|-------|----------|----------|--------------------|--------------------|
| 1     | 72       | 53       | 40                 | 43                 |
| 2     | 75       | 75       | 39                 | nan                |
| 3     | 58       | 59       | 40                 | nan                |
| 4     | 39       | 39       | 42                 | 41                 |
| 8     | nan      | nan      | 42                 | nan                |

测试条件
 - 在GTX2060 6G下进行测试，其中nan部分为未进行测试
 - 数据集读取线程数为2
 - 205张图片

不启用tensorrt的情况下，随着batch增大速度会增加，但是显存占用也会增加
 - 在batch=4时几乎占用了5G显存，batch=2占用3.5G显存
 - 并发推理能在GPU推理时候，充分利用限制的CPU进行数据集读取
 - 合理的设置batch和数据集读取线程数，在达到75%的显存占用率时，速度最快

启用tensorrt的情况下，随着batch的变化，速度基本不变，显存占用也基本不变
 - 所有测试基本都是40s左右，推测已经到达了极限，可能瓶颈在于CPU性能而不是GPU
