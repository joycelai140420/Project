应用： Compare the performance of finetune_bert and finetune_llama3 in Question Answering

步骤：

  1.数据准备：挂载Google Drive，下载并解压数据集。

  2.环境配置：查看GPU信息，安装必要的库。

  3.模型和分词器加载：加载预训练的BERT模型和分词器。

  4.数据处理：读取数据集，添加答案的开始和结束位置，进行分词处理。

  5.定义数据集类：定义用于处理数据的自定义数据集类。
  
  6.创建数据加载器：创建训练集、验证集和测试集的数据加载器。
  
  7.定义评估函数：定义用于评估模型的函数。

  8.训练和验证：进行模型训练，并在每个epoch后进行验证。

  9.保存模型：将训练好的模型保存到Google Drive。

  10.评估测试集：评估测试集，并将结果保存到CSV文件中。

  11.将保存的模型进行简单的QA问答。

实验一：

    设置训练参数：
    
        num_epochs = 3
        logging_step = 100
        learning_rate = 1e-4
