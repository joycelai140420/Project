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
![image](https://github.com/joycelai140420/Project/assets/167413809/3005c242-f723-42ad-ae43-d1fb52562a36)
![image](https://github.com/joycelai140420/Project/assets/167413809/97ad1331-ff59-420b-a08b-024e024cb0f5)
![image](https://github.com/joycelai140420/Project/assets/167413809/49e83b21-e39b-42df-b359-376fa03d4325)
![image](https://github.com/joycelai140420/Project/assets/167413809/db4bd765-120a-4e9c-be50-2c648fa34870)
![image](https://github.com/joycelai140420/Project/assets/167413809/8abde22e-1190-4921-ad19-237590fcaead)
![image](https://github.com/joycelai140420/Project/assets/167413809/55449d7a-3670-4432-b972-ea997055b4fd)
![image](https://github.com/joycelai140420/Project/assets/167413809/6b8c8f34-b2a8-429a-ac99-1ebfbf42ea7e)
![image](https://github.com/joycelai140420/Project/assets/167413809/c9de4104-9d39-4be9-8729-b40f04c3dacc)
![image](https://github.com/joycelai140420/Project/assets/167413809/2af3b57f-caef-4f3f-8a53-8414e3ef8e95)
![image](https://github.com/joycelai140420/Project/assets/167413809/fc19760b-7edc-4bc9-91c6-898e8fcab7b3)
![image](https://github.com/joycelai140420/Project/assets/167413809/1e48bfb0-e5ca-46f9-ad76-7d4dfb46efd4)
![image](https://github.com/joycelai140420/Project/assets/167413809/59cf26d2-b82f-47a1-8253-3ffdfae532a0)
![image](https://github.com/joycelai140420/Project/assets/167413809/84516486-209e-456a-a295-21db13ff518d)
![image](https://github.com/joycelai140420/Project/assets/167413809/7bf3fe76-57d4-467f-a098-22004c33fbad)
![image](https://github.com/joycelai140420/Project/assets/167413809/ffe0310e-36a3-4be2-b2e4-1936f822a74d)
![image](https://github.com/joycelai140420/Project/assets/167413809/87e1da79-c071-4b8b-8bc4-6bace7acb21b)

^^
代码详见：GAN_anime_face.ipynb


在生成对抗网络（GAN）中，生成器有时会生成不理想的图像，例如出现多余的眼睛、面部扭曲或图像噪点等。这些问题可以通过多种方法进行改善和控制：

1. 数据增强和预处理

   数据清理：确保训练数据集高质量。去除模糊、不完整或有缺陷的图像。
   数据增强：对训练数据进行增强，如随机裁剪、旋转、翻转等，以增加数据多样性。

2. 网络架构改进
   
   更深或更宽的网络：增加生成器和判别器的层数或通道数，提升模型的学习能力。
   自注意力机制：引入自注意力机制（Self-Attention）使模型能关注到图像的全局信息，改善生成效果。

3. 损失函数优化

   改进的损失函数：使用WGAN（Wasserstein GAN）、LSGAN（Least Squares GAN）等改进的损失函数，提升训练稳定性和生成效果。
   标签平滑：对真实和生成的标签进行平滑处理，避免判别器过于自信，提升生成图像的质量。

4. 训练策略调整

   增加训练时间：延长训练时间，让生成器和判别器有更多的时间互相学习。
   动态调整学习率：使用学习率衰减策略，逐渐降低学习率，避免生成器和判别器震荡。
   平衡生成器和判别器训练：确保生成器和判别器的训练进度平衡，避免一方过强或过弱。

5. 监督和引导生成

   添加条件信息：使用条件GAN（Conditional GAN），将额外的条件信息（如类别标签）输入生成器和判别器，控制生成图像的特定特征。

6. 后处理

   图像修复：生成图像后进行后处理，如图像平滑、修复等，去除多余的细节和噪点。
   选择性展示：生成多张图像，选择质量较高的进行展示。


根据上面所诉，我进行了以下改进：
    
    数据增强：在读取图片数据集时增加了transforms.RandomHorizontalFlip()，对图片进行随机水平翻转，以增加数据多样性。
    
    标签平滑：在训练过程中使用了标签平滑，将真实标签设为0.9，假标签设为0.1，以提升训练稳定性。

可以看到有效地提升DCGAN生成图像的质量，减少生成异常图像的概率。

![image](https://github.com/joycelai140420/Project/assets/167413809/28f2f731-f7c7-43fb-931e-c672f291e31d)

^^
代码详见：GAN_anime_face_v2.ipynb




























































    

