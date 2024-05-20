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

效能分析
硬体 L4 GPU RAM  22.5 GB 

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




























