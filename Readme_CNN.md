今天實作CNN的範例，但這範例我分成簡單、普通、進階，能明白可以差異。

首先先講簡單的CNN架構與設計（參考Easy_CNN.py）

步骤：
    
    准备数据集：加载并预处理MNIST数据集。
    
    建立CNN模型：构建一个简单的CNN模型。

    训练模型：使用训练数据训练模型。在這裡要注意我為了讓精度較差，只進行少量訓練。

    评估模型：在测试数据上评估模型的性能。

    可视化结果：展示一些测试样本的预测结果和模型的训练过程。

結果：

Test accuracy: 0.97079998254776

只能說經過很好ETL，會讓MNIST数据集結果不差，但也有可能是圖片很簡單。

![70ac4b7eadf3a0b14cd5e71824d2ce6](https://github.com/joycelai140420/Project/assets/167413809/7c41068d-6cd4-4009-8404-93fc616ef020)


再來看預測的結果，Label 5 預測成6，這樣的錯誤。但大部分還是對的。

![image](https://github.com/joycelai140420/Project/assets/167413809/4a403d68-f59a-4a1d-b9dd-766575e944bc)

普通的CNN架構與設計（參考Medium_CNN.py）

优化方向(設計更好的架構或採用不同的資料增強來提高效能)

1.改进CNN架构：

        增加卷积层和池化层的数量:更多的卷积层可以提取更多的特征，提高模型的表达能力。池化层可以减少参数数量，防止过拟合。
        使用更多的卷积核:更多的卷积核可以捕捉更多的特征细节。
        增加一个全连接层:更多的全连接层可以增强模型的分类能力。

2.数据增强：

        通过旋转、平移、缩放等方式增强数据，以增加数据多样性，防止过拟合。

結果：

Test accuracy: 0.9934999942779541

![image](https://github.com/joycelai140420/Project/assets/167413809/d663ebc1-7173-48c7-92b6-dd03d335b99a)

数字辨识也正确

![image](https://github.com/joycelai140420/Project/assets/167413809/202f58a0-e481-47e4-8032-a7620908de08)

進階的CNN架構與設計（參考Hard_CNN.py）
这次实验的是提供未標記資料來獲得更好的結果。

步骤：

        准备数据集：加载并预处理MNIST数据集，并应用数据增强。

        建立和训练初始CNN模型：在标记数据上进行初步训练。

        生成伪标签：使用初始模型预测未标记数据的标签。
        
        重训练模型：使用标记数据和带伪标签的未标记数据进行训练。
        
        评估和可视化结果：在测试数据上评估模型性能并可视化结果。

結果：

其中一个Initial是没有包含未标记数据的伪标签，Final开头是包含未标记数据的伪标签进行训练。包含未标记数据的标签进行训练，不管是精确度跟loss 在低的epoch都有很好的表现。

![image](https://github.com/joycelai140420/Project/assets/167413809/52dd7132-e239-4a55-b33c-d49f33f8a7bb)

数字辨识也正确

![image](https://github.com/joycelai140420/Project/assets/167413809/082a2f6f-380b-4f62-b64a-d4253655fc71)

优化方向

        1.未标记数据：通过加入未标记数据并生成伪标签，可以利用更多的数据来提升模型的泛化能力。

        2.自训练方法：初步训练模型后，使用未标记数据生成伪标签，再使用这些伪标签和原始标记数据一起重训练模型，可以有效利用未标记数据的信息。
        
        3.数据增强：在整个训练过程中，始终使用数据增强技术，以增加数据的多样性，防止模型过拟合。

通过这些改动，模型可以更好地利用标记数据和未标记数据，提高模型的性能。