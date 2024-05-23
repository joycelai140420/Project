应用： Compare the performance of finetune_bert and finetune_llama3 and Qwen in Question Answering

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

实验：

    设置训练参数：
    
        num_epochs = 3
        logging_step = 100
        learning_rate = 1e-4

    硬体：
        L4

    数据集：
    
        SQuAD v2.0数据集


      

实验结果：

    参数量:（bert-base-uncased胜）

        Qwen1.5-7B-Chat：大约 70 亿（Billion）个参数。
        Meta-Llama-3-8B-Instruct：大约 80 亿（Billion）个参数。
        bert-base-uncased：参数量大约是 1.1 亿（110 million）个参数。

    loss :




    用时：
    
        bert-base-uncased：大约四小时





    QA回答表现：
    
        bert-base-uncased：
        
            context = "我最近购买了一台高清电视机，但不知道如何进行护理。"
            Question: 高清电视机怎么进行护理？
            Answer: 高

            context = "我计划去旅行，想购买一个背包。"
            Question: 旅行背包有内外袋吗？
            Answer: [UNK] 行 [UNK] [UNK] 有 内

            context = "我正在开发一个软件，想知道如何评估测试用例的有效性。"
            Question: LLM生成测试用例的有效性如何？
            Answer: llm 生

            context = "我在处理一些实例管理问答的任务，需要一些指导。"
            Question: 如何对实例管理问答测试？
            Answer: [UNK]

            context = "BERT是由Google开发的一种用于自然语言处理的预训练模型。它可以用于各种任务，如问答、文本分类等。"
            Question: BERT是由谁开发的？
            Answer: bert

            context = "我生日是2000年1月1日，今年是2023年"
            Question: 我生日是？
            Answer: 2000 年 1 月 1 日

            接下来是截取SQuAD v2.0数据集

            context="Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in Santa Clara, California, the home stadium of the San Francisco 49ers."
            Question: Who won Super Bowl 50?
            Answer: denver broncos

            context="The Moon is Earth's only natural satellite. It is the fifth-largest satellite in the Solar System and the largest and most massive relative to its parent planet. The Moon is thought to have formed approximately 4.5 billion years ago, not long after Earth. The most widely accepted explanation is that the Moon formed from the debris left over after a giant impact between Earth and a Mars-sized body called Theia."
            Question: What is the most widely accepted explanation for the Moon's formation?
            Answer: theia










            
    
        
