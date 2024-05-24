应用： Compare the performance of finetune bert and llama3 and Qwen in Question Answering

finetune bert的步骤：

  1.数据准备：挂载Google Drive，下载并解压数据集。

  2.环境配置：查看GPU信息，安装必要的库。

  3.模型和分词器加载：加载预训练的模型。

  4.数据处理：读取数据集，添加答案的开始和结束位置，进行分词处理。

  5.定义数据集类：定义用于处理数据的自定义数据集类。
  
  6.创建数据加载器：创建训练集、验证集和测试集的数据加载器。
  
  7.定义评估函数：定义用于评估模型的函数。

  8.训练和验证：进行模型训练，并在每个epoch后进行验证。

  9.保存模型：将训练好的模型保存到Google Drive。

  10.评估测试集：评估测试集，并将结果保存到CSV文件中。

  11.将保存的模型进行简单的QA问答。

参数量:（bert-base-uncased胜）

    Qwen1.5-7B-Chat：大约 70 亿（Billion）个参数。
    Meta-Llama-3-8B-Instruct：大约 80 亿（Billion）个参数。
    bert-base-uncased：参数量大约是 1.1 亿（110 million）个参数。

实验bert-base-uncased：參考代碼QA_finetune_bert_1.ipynb


    设置训练参数：
    
        num_epochs = 3
        logging_step = 100
        learning_rate = 1e-4

    硬体：
        L4

    数据集：
    
        SQuAD v2.0数据集


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




实验 Meta-Llama-3-8B-Instruct：參考代碼QA_Llama3_1.ipynb

     硬体：
            L4
     內存：
            內存不夠需要超過64G，硬體不夠支持，所以這裡只採用call Meta-Llama-3-8B-Instruct模型不進行finetune
            
    QA回答表现：
   
            context = "我最近购买了一台高清电视机，但不知道如何进行护理。"
            Question: 高清电视机怎么进行护理？
            Answer: Congratulations on your new high-definition TV! To maintain its performance and extend its lifespan, here are some tips for caring for your new TV:
            1. **Clean the screen regularly**: Use a soft, dry cloth to wipe down the screen. Avoid using liquids, chemicals, or abrasive materials that can damage the screen or leave streaks.
            2. **Dust the TV regularly**: Use a soft-bristled brush or a can of compressed air to gently remove dust and debris from the TV's vents, grills, and other crevices.
            3. **Avoid direct sunlight**: Direct sunlight can cause the TV's screen to fade or become discolored over time. If possible, place the TV away from direct sunlight or use window treatments to filter the light.
            4. **Keep the TV away from moisture**: Avoid placing the TV near sources of moisture, such as bathrooms, kitchens, or laundry rooms, as moisture can damage the TV's internal components.
            5. **Avoid physical stress**: Be gentle when moving or handling the TV, as physical stress can cause damage to the screen or internal components.
            6. **Update the TV's software**: Regularly check for and install software updates to ensure your TV remains secure and has the latest features.
            7. **Use a surge protector**: Connect your TV to a surge protector to protect it from power surges and spikes, which can damage the TV's internal components.
            8. **Avoid overheating**: Ensure good airflow around the TV to prevent overheating, which can cause damage to the internal components.
            9. **Check the TV's cables**: Regularly check the TV's cables and connections to ensure they are secure and not damaged.
            10. **Read the user manual**: Familiarize yourself with your TV's user manual to understand its features, settings, and maintenance requirements.

            By following these tips, you can help maintain your TV's performance, extend its lifespan, and ensure it remains a reliable source of entertainment for years to come.


            context = "我计划去旅行，想购买一个背包。"
            Question: 旅行背包有内外袋吗？
            Answer: 是的，很多旅行背包都有内外袋的设计。内袋通常用于存放小件物品，如钱包、手机、耳机等，而外袋则用于存放大件物品，如衣服、鞋子等。这样可以帮助你更好地组织你的行李，并且可以快速地找到需要的物品。

            context = "我正在开发一个软件，想知道如何评估测试用例的有效性。"
            Question: LLM生成测试用例的有效性如何？
            Answer: Evaluating the effectiveness of test cases generated by LLMs (Large Language Models) is a crucial step in ensuring the quality of your software. Here are some ways to assess the effectiveness of LLM-generated test cases:

            1. **Code Coverage**: Measure the percentage of your code that is covered by the generated test cases. A higher code coverage indicates that the test cases are more effective.
            2. **Test Case Quality**: Assess the quality of the generated test cases by evaluating their relevance, completeness, and accuracy. You can use metrics such as test case complexity, test case length, and test case diversity to evaluate their quality.
            3. **Test Case Effectiveness**: Evaluate the effectiveness of the generated test cases by measuring their ability to detect defects and bugs in your software. You can use metrics such as test case failure rate, test case pass rate, and test case effectiveness ratio to evaluate their effectiveness.
            4. **Comparison with Human-Generated Test Cases**: Compare the generated test cases with human-generated test cases to evaluate their effectiveness. You can use metrics such as test case similarity, test case overlap, and test case uniqueness to evaluate their effectiveness.
            5. **Test Case Maintenance**: Evaluate the maintainability of the generated test cases by assessing their ability to be updated and modified as the software evolves. You can use metrics such as test case complexity, test case length, and test case maintenance cost to evaluate their maintainability.
            6. **Defect Detection**: Evaluate the ability of the generated test cases to detect defects and bugs in your software. You can use metrics such as defect detection rate, defect detection accuracy, and defect detection effectiveness to evaluate their ability to detect defects.
            7. **Test Case Reusability**: Evaluate the reusability of the generated test cases by assessing their ability to be reused across different software versions and releases. You can use metrics such as test case reusability rate, test case reuse ratio, and test case reusability cost to evaluate their reusability.

            By using these metrics and evaluating the effectiveness of LLM-generated test cases, you can ensure that your software is of high quality and meets the required standards.

            context = "我在处理一些实例管理问答的任务，需要一些指导。"
            Question: 如何对实例管理问答测试？
            Answer: 对实例管理问答测试可以通过以下步骤进行：
            1. 了解实例管理问答的基本概念和要求：了解实例管理问答的基本概念和要求，可以帮助您更好地理解问题和回答。
            2. 读懂问题：读懂问题，包括理解问题的背景、关键词和要点，帮助您更好地回答问题。
            3. 选择合适的回答：选择合适的回答，根据问题的内容和要求选择合适的回答。
            4. 检查回答：检查回答，确保回答准确、完整和有逻辑性。
            5. 评估回答：评估回答，根据回答的内容和要求评估回答的质量。
            6. 改进回答：改进回答，根据评估结果改进回答，直到满足要求。

            总之，实例管理问答测试需要您对问题和回答有深入的理解和分析能力，同时也需要您有良好的语言和逻辑能力。通过这些步骤，您可以更好地处理实例管理问答测试。

            context = "BERT是由Google开发的一种用于自然语言处理的预训练模型。它可以用于各种任务，如问答、文本分类等。"
            Question: BERT是由谁开发的？
            Answer: BERT (Bidirectional Encoder Representations from Transformers) is developed by Google. It was introduced in a research paper by Google in 2018. The paper was titled "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" and was published at the 2018 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL). The authors of the paper were Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT is based on the transformer architecture and uses a combination of masked language modeling and next sentence prediction tasks to pre-train the model. It has been widely used in natural language processing tasks and has achieved state-of-the-art results in many areas.

            context = "我生日是2000年1月1日，今年是2023年"
            Question: 我生日是？
            Answer: 23

            接下来是截取SQuAD v2.0数据集

            context="Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in Santa Clara, California, the home stadium of the San Francisco 49ers."
            Question: Who won Super Bowl 50?
            Answer: denver broncos

            context="The Moon is Earth's only natural satellite. It is the fifth-largest satellite in the Solar System and the largest and most massive relative to its parent planet. The Moon is thought to have formed approximately 4.5 billion years ago, not long after Earth. The most widely accepted explanation is that the Moon formed from the debris left over after a giant impact between Earth and a Mars-sized body called Theia."
            Question: What is the most widely accepted explanation for the Moon's formation?
            Answer: theia



tips:
    
    1.如果內存支撐夠64G以上，可以Meta-Llama-3-8B-Instruct开源模型为模型基座，通过开源程序LLaMA-Factory来进行中文的微调，提高Llama3的中文能力！
    
        
