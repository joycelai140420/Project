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

参数量:

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
    蓝色是Epoch 1 ，橘色是Epoch 2 ,灰色是Epoch 3 

![image](https://github.com/joycelai140420/Project/assets/167413809/2e388bd7-e7c9-402a-a79b-78c01b3b0da8)



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

实验 Qwen/Qwen1.5-7B-Chat：參考代碼QA_Qwen_1.ipynb

     硬体：
            L4
     內存：
            因为懒，所以這裡只採用call Qwen/Qwen1.5-7B-Chat模型不進行finetune，后面再看看进行finetune。
            
    QA回答表现：
    
            context = "我最近购买了一台高清电视机，但不知道如何进行护理。"
            Question: 高清电视机怎么进行护理？
            Answer: 高清电视机的护理主要是为了保持其良好的显示效果和延长使用寿命，以下是一些基本的护理建议：

            1. **清洁表面**：
            - 使用专用的高清电视清洁布，避免使用含有氨、漂白剂或者研磨性的清洁剂，这些都可能损坏屏幕。
            - 清洁时轻轻擦拭，不要用力按压或拖拽。

            2. **防尘**：
            - 定期用软布或者专用的防尘刷清理电视机周围的灰尘，特别是通风口和底座。
            - 避免在电视机前摆放容易积累灰尘的物品，如书本、花瓶等。

            3. **环境调节**：
            - 保持观看环境的温度适宜，过高过低都会影响画质，一般来说，18-24℃为宜。
            - 避免阳光直射，防止屏幕过热。

            4. **避免震动**：
            - 不要随意移动电视机，尤其是刚买回来的那段时间，应该固定好位置。
            - 不要在电视上放置重物，以防震动。

            5. **正确开关**：
            - 关闭电视机时，要按照遥控器的指示操作，避免突然断电对屏幕造成损害。

            6. **定期维护**：
            - 如果你有保修，可以定期联系专业人员进行深度检查和维护。
            - 对于智能电视，记得定期更新操作系统和软件，以保持最新功能和安全。

            7. **正确连接与使用**：
            - 确保所有接线正确且稳固，特别是HDMI线，过松或过紧都可能导致信号问题。

            8. **避免长时间观看**：
            - 长时间观看后，让电视休息一会儿，减少内部元件的负荷。

            遵循以上步骤，你的高清电视机就能得到很好的保护，保持良好的工作状态。


            context = "我计划去旅行，想购买一个背包。"
            Question: 旅行背包有内外袋吗？
            Answer: 是的，旅行背包通常会设计有内外袋。内袋通常位于背包的主体部分，用来存放日常用品，如手机、钱包、水瓶、洗漱用品等私人物品，它们通常分隔得较为有序，方便取用。外袋则位于背包的外部，通常有拉链或者魔术贴，可以用来存放外套、雨伞、相机、零食或者其他大的、不常用但又需要随时取用的物品。有些背包还会有专门的电脑仓或者是侧袋，以满足不同物品的收纳需求。此外，现在的旅行背包还会考虑到便携性和舒适性，比如有腰带系统、背部支撑和可调节肩带等设计。在选择背包时，可以根据自己的行程和个人需求来挑选合适的款式。
            
            context = "我正在开发一个软件，想知道如何评估测试用例的有效性。"
            Question: LLM生成测试用例的有效性如何？
            Answer: LLM (Learning-based Language Model) 本身并不能直接生成测试用例并评估其有效性。然而，你可以利用LLMs来辅助和优化测试用例设计。以下是一些步骤和方法来将LLM与测试用例的有效性评估结合起来：

            1. **需求理解**：
               - 首先，确保你对软件的功能和业务需求有深入的理解。LLM可以帮助你从用户文档、需求规格书或业务流程中提取关键信息。

            2. **语义分析**：
               - 使用LLM（如BERT、GPT等）进行文本理解，解析用户故事、功能描述或错误案例，以识别潜在的边界情况和异常情况。

            3. **生成测试场景**：
               - 利用LLM自动生成可能的输入数据或场景，比如针对特定函数的边界值、正常操作或错误输入。这些场景应覆盖预期的行为和异常情况。

            4. **人工审核**：
               - 尽管LLM可以生成大量候选，但重要的是人工检查和验证这些场景是否有效，包括是否覆盖所有功能点，是否符合预期逻辑，以及是否具有实际意义。

            5. **性能和覆盖率**：
               - 评估LLM生成的测试用例在自动化测试工具中的执行效果，比如它们能够覆盖多少代码路径，或者在回归测试中的失败率。

            6. **反馈迭代**：
               - 根据测试结果调整LLM模型，使其更好地适应你的测试目标。如果某些测试用例效果不佳，可以调整模型的训练数据或参数。

            7. **持续优化**：
               - LLM是一个动态发展的技术，随着模型的训练和更新，你可能需要定期评估其生成测试用例的效果。

            总的来说，LLM可以作为一个有力的辅助工具，帮助你生成测试用例，但它不能完全取代人工的测试策略和经验。有效的测试用例应该是全面、细致且具有挑战性的，而LLM可以作为其中的一部分，提高测试的效率和质量。

            context = "我在处理一些实例管理问答的任务，需要一些指导。"
            Question: 如何对实例管理问答测试？
            Answer: 实例管理问答测试通常涉及以下几个步骤来确保系统的准确性和效率。以下是一些具体的指导：

            1. **明确目标**：首先，你需要明确测试的目标。这可能包括验证系统是否能正确理解用户问题，提供准确的答案，以及处理各种复杂情况等。

            2. **设计测试用例**：创建一个全面的测试案例库，覆盖不同类型的实例，比如基础知识、复杂问题、边缘情况和异常处理。每个案例应包含问题描述、预期答案和可能的输入变体。

            3. **数据准备**：为系统提供各种类型的数据，包括常见问题、罕见问题、模糊不清的问题以及用户可能使用的多种表达方式。

            4. **功能测试**：测试系统的核心功能，如搜索、推荐、过滤、排序等，确保它们在不同实例中的表现一致。

            5. **性能测试**：模拟大量并发用户或大规模数据，检查系统响应速度和稳定性，以评估其在高负载下的表现。

            6. **用户体验测试**：让真实用户或潜在用户进行测试，观察他们在使用过程中的操作流程和反馈，找出可能的人机交互问题。

            7. **错误和异常处理**：测试系统在遇到无效输入、网络中断、服务器故障等情况时的表现，看它如何正确地处理这些问题并给出提示。

            8. **持续集成与持续部署（CI/CD）**：频繁地在开发环境中进行测试，确保每次代码更改都能顺利通过测试。

            9. **数据分析**：收集测试结果，分析系统性能瓶颈和用户反馈，以便改进。

            10. **修订与优化**：根据测试结果调整系统，优化算法或者改进用户界面，以提高服务质量。

            记住，测试是一个迭代的过程，随着系统的不断优化，测试策略也需要随之调整。

            context = "BERT是由Google开发的一种用于自然语言处理的预训练模型。它可以用于各种任务，如问答、文本分类等。"
            Question: BERT是由谁开发的？
            Answer: BERT是由Google开发的。

            context = "我生日是2000年1月1日，今年是2023年"
            Question: 我生日是？
            Answer: 您的生日是2000年1月1日。

            接下来是截取SQuAD v2.0数据集

            context="Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in Santa Clara, California, the home stadium of the San Francisco 49ers."
            Question: Who won Super Bowl 50?
            Answer: The Denver Broncos won Super Bowl 50. They defeated the Carolina Panthers with a score of 24-10. This victory marked their third Super Bowl title.

            context="The Moon is Earth's only natural satellite. It is the fifth-largest satellite in the Solar System and the largest and most massive relative to its parent planet. The Moon is thought to have formed approximately 4.5 billion years ago, not long after Earth. The most widely accepted explanation is that the Moon formed from the debris left over after a giant impact between Earth and a Mars-sized body called Theia."
            Question: What is the most widely accepted explanation for the Moon's formation?
            Answer: The most widely accepted explanation for the Moon's formation is the giant impact theory. According to this theory, around 4.5 billion years ago, a Mars-sized body named Theia collided with Earth. The impact caused a massive amount of debris to be thrown up into space, which eventually coalesced and formed the Moon. This scenario explains the Moon's size, relative mass, and position as a natural satellite orbiting Earth.
    


总结：
很难评断，假设finetune bert多一点训练跟资料量，表现应该比现在好很多，然Meta-Llama-3-8B-Instruct跟Qwen/Qwen1.5-7B-Chat是因为已经获得大量资料进行训练，所以在回答上都不错，但Meta-Llama-3-8B-Instruct遇到没训练过的，如生日题，就回答错。总体而言我个人感官Qwen/Qwen1.5-7B-Chat中文与英文的理解还有答案都比较让我喜欢。

tips:
    
    1.如果內存支撐夠64G以上，可以Meta-Llama-3-8B-Instruct开源模型为模型基座，通过开源程序LLaMA-Factory来进行中文的微调，提高Llama3的中文能力！
    
        
