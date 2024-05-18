应用：
利用ted2020的数据(70mb数据量，有两份文件raw.en、raw.zh)来训练一个中文字翻译成英文字的应用

硬体使用：

    云端google colab A100（RAM 83.5 GB、GPU RAM 40.0 GB）

实验1步驟：

    1.训练一个简单的RNN Seq2Seq模型。

    2.切换到Transformer模型以提升性能。

    3.应用回译（Back-translation）进一步提升性能。

    4.使用TED2020数据集进行训练和演示。

失败原因：
运行到RNN系统跑崩，我在RNN就因为embedding_dim = 256，units = 512跟最后run epochs=10跑崩，出现内存不足（Out of Memory，OOM）错误。为了解决的这问题我减少模型
