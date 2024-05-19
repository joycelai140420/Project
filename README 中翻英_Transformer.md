应用：
利用ted2020的数据(70mb数据量，有两份文件raw.en、raw.zh)来训练一个中文字翻译成英文字的应用，想利用RNN Seq2Seq模型跟Transformer来进行比较。

硬体使用：

    云端google colab A100（RAM 83.5 GB、GPU RAM 40.0 GB）

实验1步驟：

    1.训练一个简单的RNN Seq2Seq模型。

    2.切换到Transformer模型以提升性能。

    3.应用回译（Back-translation）进一步提升性能。

    4.使用TED2020数据集进行训练和演示。

失败原因：

运行到RNN系统跑崩，我在RNN就因为embedding_dim = 256，units = 512跟最后run epochs=10跑崩，出现内存不足（Out of Memory，OOM）错误。为了解决的这问题我减少模型我修改

        1,减少模型复杂性：减少LSTM单元数量嵌入维度的大小还有epochs

        2,数据生成器：分批加载数据以减少内存占用
        
一直降到 embedding_dim = 32，units = 64跟最后run epochs=5才跑起来，但因为太久导致 colab 只保留最长24小时（还在后台），12小时消失前台，最后断线，导致没跑完，100元终究是错付，1个RNN epochs跑2小时，不想再跑，有兴趣看两个模型比较结果可以参考中翻英_RNN_Transformer.ipynb。所以只实作Transformer部分，也就是实验2，不过也是花了我少钱。

实验2步驟：

        1.训练一个Transformer模型。

        2.应用回译（Back-translation）进一步提升性能。

        3.评估Transformer模型的性能，并演示它的翻译结果。

结果：

        我只执行一个epoch  

        loss: 0.2516 - accuracy: 0.9740

        Total params: 22782551 (86.91 MB)

        Trainable params: 22782551 (86.91 MB)


        
