![image](https://github.com/joycelai140420/Project/assets/167413809/8235b8e9-2547-45d2-ab73-7632c42b9705)


应用：
是利用GAN来生成动漫脸孔

步骤：
  
  1.随机种子：先将随机种子设定为特定值可以实现重覆性。

  2.将影响大小调整为 (64, 64)将值从 [0, 1] 线性映射到 [-1, 1]。

  3.显示一些图片，请注意这些值范围已经转成是 [-1, 1]，我们应该将他们移到有效的范围 [0, 1]，以便正确显示。

  4.使用DCGAN作为模型架构。这边可以随意改变自己的模型架构。同时要注意，输入/输出形状的N代表批次的大小。

  5.在训练的过程中，我们定期储存10张图片来监控生成器目前的效能，并定期记录检查点。

  6.使用训练好的模型生成动漫面孔，并显示模型生成后的动漫面孔图片 跟真實圖片進行比較。

  
DCGAN（Deep Convolutional Generative Adversarial Network）是一种利用深度卷积神经网络（CNN）来生成图像的生成对抗网络（GAN）架构。

DCGAN的架构

生成器（Generator）：

    生成器接受一个随机噪声向量（通常是多维高斯分布）作为输入，通过一系列反卷积（ConvTranspose）层将其转换为一张图像。
    每一层通常包括反卷积、批归一化（Batch Normalization）和ReLU激活函数，最后一层使用tanh激活函数将输出值限制在[-1, 1]。
    
![image](https://github.com/joycelai140420/Project/assets/167413809/f1352b0f-6bcb-40e3-ac2f-03ab95d6c12f)

判别器（Discriminator）：

    判别器接受一张图像作为输入，通过一系列卷积（Convolution）层对其进行处理，最终输出一个概率值，表示输入图像是真实的还是生成的。
    每一层通常包括卷积、批归一化和Leaky ReLU激活函数，最后一层使用sigmoid激活函数输出一个0到1之间的值。

![image](https://github.com/joycelai140420/Project/assets/167413809/5b8a8ed4-234f-4b8c-8353-8dbb38ae0975)


实验成果展现

    epochs一共50次，所以由低到高来展示“抽样”贴上每个检查点的图片
    
![image](https://github.com/joycelai140420/Project/assets/167413809/b41be98f-5c8f-44c6-995e-e8cbaa61159d)
![image](https://github.com/joycelai140420/Project/assets/167413809/80b7a108-0d27-46bd-8a7b-1acb897b8876)
![image](https://github.com/joycelai140420/Project/assets/167413809/94a96fa0-4006-4813-9dee-706841874012)
![image](https://github.com/joycelai140420/Project/assets/167413809/b6c932da-860b-4429-96d4-8535317fb074)
![image](https://github.com/joycelai140420/Project/assets/167413809/c94080c2-02aa-4da1-90d5-3f9db7d69180)
![image](https://github.com/joycelai140420/Project/assets/167413809/5c9a1cb3-1b78-4a16-a0a9-ba5c9898500b)
![image](https://github.com/joycelai140420/Project/assets/167413809/917898ff-6bc0-4272-ab4e-7c5a6c7ac0c2)
![image](https://github.com/joycelai140420/Project/assets/167413809/7c491364-f19e-476f-81ab-10c2b4889ebb)
![image](https://github.com/joycelai140420/Project/assets/167413809/210bd3ab-df51-4730-9218-cc3fce4dfa79)
![image](https://github.com/joycelai140420/Project/assets/167413809/09a62525-61b3-4a0c-a451-81c534e3dadf)
![image](https://github.com/joycelai140420/Project/assets/167413809/0b5ae46d-5f50-479f-8f3e-2df22aad0e03)
![image](https://github.com/joycelai140420/Project/assets/167413809/934ec520-0996-4c63-a0a8-0582c1e86cb7)
![image](https://github.com/joycelai140420/Project/assets/167413809/bc77e851-d9da-48c5-992d-f2a3cc8f391e)
![image](https://github.com/joycelai140420/Project/assets/167413809/7f3e897a-973f-4256-aee3-56aeded9b2e4)
![image](https://github.com/joycelai140420/Project/assets/167413809/5595a773-e044-4adb-ab8d-a6c09d7114e7)
![image](https://github.com/joycelai140420/Project/assets/167413809/a49be8d0-d8ba-4bb8-905a-acaf1aa14f76)
![image](https://github.com/joycelai140420/Project/assets/167413809/0b07e763-c488-496e-a5dd-857073a26ddf)
![image](https://github.com/joycelai140420/Project/assets/167413809/600f1528-2816-4ba5-a1cf-62e5200e9f78)
![image](https://github.com/joycelai140420/Project/assets/167413809/6b4c8c84-7e59-4b5d-8e8e-658a68c6fa88)
![image](https://github.com/joycelai140420/Project/assets/167413809/25961ca6-4473-498a-9abb-65a2c62107e9)
![image](https://github.com/joycelai140420/Project/assets/167413809/dd75e288-d9e9-4b23-bca7-a6b133d62e5f)
![image](https://github.com/joycelai140420/Project/assets/167413809/2f98944f-8aac-4873-bf9c-f00f8d6c1065)
![image](https://github.com/joycelai140420/Project/assets/167413809/04f87cca-5567-4f17-898d-1611db796721)
![image](https://github.com/joycelai140420/Project/assets/167413809/511d9f3c-c33d-4196-af1d-0cc1408dd3e5)
![image](https://github.com/joycelai140420/Project/assets/167413809/2b3f5853-5e09-4378-a1c6-06f9e66902ef)
![image](https://github.com/joycelai140420/Project/assets/167413809/cabae4d9-b1f9-4fb1-931b-c15b9899f749)
![image](https://github.com/joycelai140420/Project/assets/167413809/665b01e9-c535-4559-b4d2-001f75fabe97)
![image](https://github.com/joycelai140420/Project/assets/167413809/87670018-77bb-490b-aea6-469fbf716cee)
![image](https://github.com/joycelai140420/Project/assets/167413809/a42a04f1-3d4e-4b4f-9cc3-ba523f600b42)
![image](https://github.com/joycelai140420/Project/assets/167413809/47e6bd7f-974a-4a61-8611-0ef7209b68ba)
![image](https://github.com/joycelai140420/Project/assets/167413809/f62a4a79-9770-40ec-a17a-0ee8c63ca4e8)
![image](https://github.com/joycelai140420/Project/assets/167413809/79699c33-15af-470f-b160-07e8dbecff53)



^^
代码详见：GAN_anime_face.ipynb





































































    

