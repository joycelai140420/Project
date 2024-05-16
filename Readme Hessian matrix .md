Hessian matrix

我们怎么知道在训练模型的时候，是否卡在local minima还是saddle point，还是其他，可以参考范例Hessian matrix_example.py2，我训练一个神经网络，这个神经网络模型的目的是进行分类任务。具体来说，我们使用的是著名的Iris数据集，该数据集包含四个特征（花萼长度、花萼宽度、花瓣长度、花瓣宽度），目标是将数据集中的鸢尾花分为三类（山鸢尾、变色鸢尾、维吉尼亚鸢尾）。并将训练数据计算Hessian矩阵并判断极值类型，可以修改神经网络模型让他变差来看一下例如改一下activation或最佳化的时候不用adam，loss用mse等等，再来观察有几个local min 、local max、saddle point，还是其他。


![1715854487277](https://github.com/joycelai140420/Project/assets/167413809/d2f7553c-bd57-4d47-b88d-d0fd1a739c95)
