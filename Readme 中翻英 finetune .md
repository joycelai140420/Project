本人大实话：

说实话这个项目没做好（花时间又花钱），等后面有机会在修改。但你们还想看，就看下去吧！

目的：

    微调预训练过的t5-small模型进行中翻英，其中分别用了两个预训练分词器（t5-small跟bert-base-chinese）进行比较。
    使用的数据集wmt18，zh-en数据，并只是用所有数据集1%数据量
    硬体跟金钱有限，所以不跑太多数据量与epochs，分别是1次跟10次进行效能的比较。

T5ForConditionalGeneration介绍

T5ForConditionalGeneration类是用于文本生成任务的T5模型。T5（Text-to-Text Transfer Transformer）由Google提出，可以将几乎所有NLP任务转换为文本到文本的格式。以下是一些主要特性：

    统一框架：T5将各种NLP任务（如翻译、摘要、问答等）都统一到文本生成任务上。

    多种模型规模：提供了从t5-small到t5-11b多种规模的模型，用户可以根据需求选择合适的模型。

    预训练和微调：T5模型预训练在大规模数据集上，用户可以在特定任务上进行微调以提高性能。

预训练模型和微调

    预训练模型：

        模型在大规模文本数据上进行预训练，学习通用的语言表示。
        t5-small是预训练模型的一个版本，具有60M参数，是T5家族中较小的版本。

    微调（Fine-tuning）：

        在特定任务上对预训练模型进行训练，使其性能在该任务上得到优化。
        你当前的任务是微调t5-small模型，使其能够更好地完成特定的翻译任务。



代码解析：

model = T5ForConditionalGeneration.from_pretrained('t5-small', force_download=True).to(device)

加载一个预训练的T5模型，并将其移动到指定设备（例如GPU或CPU）上。

T5ForConditionalGeneration:

    这是Hugging Face Transformers库中的一个类，专门用于生成任务（如文本生成、翻译等）的T5模型。
    它继承自PreTrainedModel，并提供了特定的生成方法，如generate，用于执行文本生成任务。

from_pretrained:

    这是一个类方法，用于加载预训练的模型。
    't5-small'参数指定了模型的版本。T5模型有多个版本，如t5-small、t5-base、t5-large、t5-3b、t5-11b，它们的大小和能力依次增加。
    加载的模型是由Google在大规模文本数据上预训练的，可以用于许多NLP任务。

force_download=True:

    这个参数强制重新下载模型，即使模型已经存在于缓存中。这在模型更新或缓存损坏时特别有用。

to(device):

    device指示模型应该加载到哪个设备上。常见的设备有：
    'cuda'：如果有GPU可用，加载到GPU上，以加快计算速度。
    'cpu'：如果没有GPU，则加载到CPU上。

============================================================================================================================================

AutoTokenizer介绍

AutoTokenizer类是Hugging Face Transformers库中的一部分，旨在简化加载和使用各种预训练分词器的过程。它可以自动检测和加载适合特定模型的分词器。

主要功能：

    自动选择并加载适当的分词器。
    支持多种模型，包括BERT、GPT-2、T5等。
    通过简化分词器的选择和加载过程，减少用户的工作量。
    
主要方法：

    from_pretrained：加载预训练的分词器。
    from_pretrained方法
    from_pretrained方法用于加载预训练的分词器或模型。它会从Hugging Face的模型库中下载预训练的分词器或模型参数，并进行初始化。

主要参数：

    pretrained_model_name_or_path：预训练模型或分词器的名称或路径。
    force_download：是否强制重新下载模型或分词器（可选）。
    cache_dir：指定缓存目录（可选）。
    

代码解析：
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    当前的任务是加载一个预训练的中文BERT分词器，以便将中文文本转换为模型可以处理的token ID。

AutoTokenizer:

    AutoTokenizer是Hugging Face Transformers库中的一个类，用于自动选择适当的分词器。
    它可以根据模型名称自动加载对应的分词器，简化了使用不同模型时选择分词器的过程。
    
from_pretrained:

    这是一个类方法，用于加载预训练的分词器。
    'bert-base-chinese'参数指定了分词器的版本。是BERT模型的中文版本。专门针对中文进行了训练。
    


============================================================================================================================================


T5Tokenizer介绍

    T5Tokenizer类是Hugging Face Transformers库的一部分，专门用于T5模型的分词任务。它使用了特定的词汇表和分词算法，使其与T5模型的预训练和架构相兼容。

主要功能：

    将输入文本转换为token ID序列，模型可以处理这些ID。
    将模型生成的token ID序列转换回文本。
    处理特殊标记（如起始标记、结束标记、填充标记等）。

主要方法：

    from_pretrained：加载预训练的分词器。
    encode：将文本转换为token ID序列。
    decode：将token ID序列转换回文本。
    from_pretrained方法
    from_pretrained方法用于加载预训练的分词器或模型。它会从Hugging Face的模型库中下载预训练的分词器或模型参数，并进行初始化。

主要参数：
    
    pretrained_model_name_or_path：预训练模型或分词器的名称或路径。
    force_download：是否强制重新下载模型或分词器（可选）。
    cache_dir：指定缓存目录（可选）。

代码解析：
    
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    正在加载一个预训练的T5分词器t5-small，该分词器适用于T5模型。这个分词器将文本转换为模型可以处理的token ID，为后续的模型输入做准备。


============================================================================================================================================


大模型
  
    大模型：
       
        通常指具有大量参数的模型版本，如t5-large、t5-3b、t5-11b、bert-large、gpt-2-xl等。
        大模型通常在性能上更强，但需要更多的计算资源进行训练和推理。
        在这个例子中，t5-small，这是T5模型家族中较小的版本。
        在这个例子中，'bert-base-chinese'是一个基础模型版本，具有较少的参数，但已经足够处理许多任务。

效能分析(硬体 L4 GPU RAM  22.5 GB ,wmt18，zh-en数据集大小: 251603)

bert-base-chinese预训练的分词器与预训练的t5-small模型

1 epochs

    training_loss=2.986716564407426
    示例 1
    中文: 令人宽心的
    英文: ,,
    --------------------
    示例 2
    中文: 1929年
    英文: 
    --------------------

    示例 1 有学到了,, 

10 epochs 训练约5hr

![1716213813272](https://github.com/joycelai140420/Project/assets/167413809/1e6a884e-e423-49a9-9bca-5d1207188899)

    示例 1
    中文: 巴黎-随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。
    英文: , the s s.
    --------------------
    示例 2
    中文: 一开始，很多人把这次危机比作1982年或1973年所发生的情况，这样得类比是令人宽心的，因为这两段时期意味着典型的周期性衰退。
    英文: , the s s s s.
    --------------------
    示例 3
    中文: 今天是个好天气
    英文: 
    --------------------
    示例 4
    中文: 令人宽心的
    英文: 
    --------------------
    示例 5
    中文: 1929年还是1989年?
    英文: 
    --------------------
    示例 1跟2 好像学到某些英文字 例如 the 或是某些单字里面的s


t5-small预训练的分词器与预训练的t5-small模型

1 epochs
training_loss=忘了记录
![043dae4960649972aee7e388ff8369c](https://github.com/joycelai140420/Project/assets/167413809/02dff06d-b5c0-416f-8b28-3946d792e2b5)



10 epochs 训练约4.5hr

![1716294560899](https://github.com/joycelai140420/Project/assets/167413809/c0b647dc-e19d-469a-a542-427865fc6841)

    示例 1
    中文: 巴黎-随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。
    英文: .
    --------------------
    示例 2
    中文: 一开始，很多人把这次危机比作1982年或1973年所发生的情况，这样得类比是令人宽心的，因为这两段时期意味着典型的周期性衰退。
    英文: As a result, in 1982 and 1973, the United States, the United States, the United States, and the United States, the United States, the United States, and the United Kingdom, the United States, the United States, and the United Kingdom, the United States, the United States, and the United Kingdom, the United States, the United States, and the United Kingdom, the United States, the United Kingdom, and the United Kingdom, the United States, the United States, and the United Kingdom, have been unable to achieve their goals
    --------------------
    示例 3
    中文: 今天是个好天气
    英文: But this is not the case.
    --------------------
    示例 4
    中文: 令人宽心的
    英文: But this is not the case.
    --------------------
    示例 5
    中文: 1929年还是1989年?
    英文: What happened in 1929 and 1989?
    --------------------
    示例 6
    中文: 巴黎
    英文: But this is not the case.
    --------------------


tips:

    1.使用特定的分词器来处理输入文本。我后面是使用t5-small进行训练，所以在分词器选择t5-small比bert-base-chinese更适配
    
    2.用A100显卡，在使用分词器将输入文本转换为token ID序列，指定最大长度为128就差不多，512打死紧绷。虽然理论上是可以设很大，但是在硬体设施有限情况下最多512再多就爆内存。
    
    3.我后来仔细看为什么是学习到整段句子翻译，没有学到有效的语法转换或单词意，所以输入完整的示例找到是对应整句的中文，原因是训练数据量可能不足以让模型学到有效的转换规则。训练时间可能不足，导致模型没有充分学习到有效的转换规则。当然调整超参数也是有影响，但是这里主要的问题是数据量不足跟训练时间不足。参考以下的paper。
    
    https://arxiv.org/pdf/2203.15556
    
    ![image](https://github.com/joycelai140420/Project/assets/167413809/978a00fb-5215-4ccˇa-a172-2afc6c42ab8a)

    4.官方网站说T5模型在使用AdamW优化器进行微调时，通常需要比默认设置略高的学习率。但这里我没有使用AdamW优化器。默认学习率一般为5e-5。建议的学习率范围通常为1e-4（0.0001）到3e-4（0.0003），这些值在大多数问题上（如分类、摘要、翻译、问答、问题生成）效果较好。T5模型在预训练时使用了AdaFactor优化器，AdaFactor是一种在内存和计算效率方面更优的优化器，特别适合大规模训练。所以我取中间值2e-5。
    
    https://huggingface.co/docs/transformers/model_doc/t5

    5.per_device_train_batch_size跟per_device_eval_batch_size我设定是128，因为再大会爆内存（我使用L4）,128算高了，两个值要设大小一致，保证评估时效率较高。
    
    6.epochs，较长的训练时间可以让模型更充分地学习，但也会增加训练时间。可以根据模型的收敛情况调整。可以参考第三点的paper。
    
    7.fp16:混合精度训练可以加速训练并减少显存使用，对于现代GPU非常有效。
    
    8.任务前缀在进行多任务训练时非常重要。如果你的任务与T5预训练时使用的监督任务相似或相关，任务前缀同样重要。所以我这个实验在1 epochs跟bert-base-chinese 1 epochs跟10 epochs，忘记写前缀，所以机器的回答非常差。这是很重大的问题。
    
    https://huggingface.co/docs/transformers/model_doc/t5
    
    9.在TPU上训练时，建议将数据集中的所有示例填充到相同的长度，或者使用pad_to_multiple_of参数设置为固定的桶大小。动态地将每个批次填充到最长示例的长度不推荐，因为这会导致每次遇到新的批次形状时重新编译，显著减慢训练速度。
    
    https://huggingface.co/docs/transformers/model_doc/t5
    
    10.weight_decay=0.01可以尝试不同的值，如0.001或0.1，根据模型的过拟合情况进行调整。gradient_accumulation_steps默认值：通常为1，表示每一步都进行梯度更新。适用于显存较大的环境，因为批次大小不受限制。设定较大值（例如2或更多）可以通过累积多个小批次的梯度进行一次更新，从而减小每一步的显存占用。每进行一次完整的梯度更新需要更多的训练步骤，训练时间可能增加。


