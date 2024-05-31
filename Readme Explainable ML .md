 Explainable ML

在现代社会，机器学习（ML）已经深刻地融入了我们的日常生活，从推荐系统到自动驾驶，再到医疗诊断，无处不在。然而，尽管这些技术带来了前所未有的便利和效益，但它们的“黑箱”性质却引发了广泛的关注和争议。人们开始意识到，仅仅依赖于机器学习模型的高准确性是不够的，更重要的是理解这些模型如何做出决策。这就是可解释的机器学习（Explainable Machine Learning，简称XML）兴起的背景和动因。

可解释的机器学习旨在揭示和解释机器学习模型内部的工作机制，帮助我们理解模型如何从输入数据中提取特征，并最终做出预测。

Saliency Map

    首先先介紹這篇論文，这篇名为“Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps”的论文由Karen Simonyan, Andrea Vedaldi, 和 Andrew Zisserman在2014年的ICLR会议上发表。

    作者提出了一种可视化方法，帮助理解CNN模型如何进行图像分类，以及识别图像中哪些区域对分类结果贡献最大。具体来说，Saliency Map显示了输入图像中每个像素的梯度大小，这些梯度表示该像素值的微小变化如何影响最终的分类分数。

    主要方法：

        梯度计算：作者通过计算输入图像相对于目标类别得分的梯度，生成Saliency Map。这些梯度可以通过反向传播算法高效地计算得到。
        
        显著性图的可视化：通过将计算得到的梯度值映射到图像像素上，生成显著性图。高梯度值对应图像中对分类结果影响较大的区域。

   从右下图可知道图像每个像素对损失的梯度，得到一个梯度张量，然後取每个像素位置在颜色通道上梯度的最大值，生成一个二维的Saliency Map，显示图像中哪些区域对分类结果最重要。所以红色就是看出哪些区域对分类结果最重要。左下是原始图
    
![image](https://github.com/joycelai140420/Project/assets/167413809/c463b55b-8f92-415c-b435-14a3a12203c7)


    程式範例可參考Saliency Map.py

        
