Batch Norm
Reference：

https://www.zhihu.com/question/553541499/answer/1892702115723452465

https://zhuanlan.zhihu.com/p/54530247
### ICS
内部协变量偏移（Internal Covariate Shift, ICS）指神经网络在训练过程中，由于前一层的参数更新导致后续层的输入分布发生显著变化的现象。这种分布的不稳定性会降低训练效率，增加模型收敛的难度。这种现象在深层网络中尤为明显， 是导致训练不稳定、收敛缓慢甚至[梯度消失](https://zhida.zhihu.com/search?content_id=721744268&content_type=Answer&match_order=1&q=%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1&zhida_source=entity)/爆炸的原因之一。

       ICS发生的主要原因：

        1）网络参数更新导致分布变化；

        2) 非线性激活函数的敏感性问题；[ReLU](https://zhida.zhihu.com/search?content_id=721744268&content_type=Answer&match_order=1&q=ReLU&zhida_source=entity)、[Sigmoid](https://zhida.zhihu.com/search?content_id=721744268&content_type=Answer&match_order=1&q=Sigmoid&zhida_source=entity)、[Tanh](https://zhida.zhihu.com/search?content_id=721744268&content_type=Answer&match_order=1&q=Tanh&zhida_source=entity)等非线性激活函数对不同输入范围的响应不同，如果前一层的输出分布偏移到激活函数的饱和区（如Sigmoid的两端), 梯度会变得极小(梯度消失),  导致训练停滞。

        3）深度网络的累积效应； 即使每一层变化很小，由于多层叠加后， 输入分布的偏移会被放大，导致深层网络的输入分布剧烈波动；

        4）梯度下降的依赖问题； 梯度下降算法假设输入数据的分布是稳定的， 但内部协变量偏移打破了这一假设， 使得优化过程变得不稳定。 由于每一层的输入分布不断变化，优化器需要不断调整学习率， 导致训练收敛变慢。

Batch Normalization（BN）通过对每一层的输入进行标准化（Normalization），使得数据分布更加稳定，是解决『内部协变量偏移』的一种有效方式。


### BN limitation
![image](/blogs/images/HwFPbofd6og0O6xuDjlcoeJhnEc.png)

### LN：Layer Norm
![image]({{ "images/WoqIbAeSFocPP6xvN19cda14nfc.png" | relURL }})
```python
# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
# 指定归一化的维度
layer_norm = nn.LayerNorm(embedding_dim)
# 进行归一化
layer_norm(embedding)
 
# Image Example
N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W)
# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
layer_norm = nn.LayerNorm([C, H, W])
output = layer_norm(input)
```
RMSNorm应该是LayerNorm砍掉算均值这一步，不强求中心一致。faster
#### 1. 稳定训练，防止梯度爆炸/消失
- 深层网络中每一层输出的尺度会不断变化，容易导致：
	- 梯度爆炸（数值变得特别大）
		- 梯度消失（数值趋近于0）
	
- RMSNorm 通过统一每个 token 向量的“长度”，让每层的输出保持在稳定范围内。

类比：像做体操前“先把身体拉开”，让后续动作更安全、流畅。
---
#### 2. 加快收敛，提升训练速度
- 归一化后每层的输入分布更“标准”，优化器收敛更快；

- 在不需要预热很长时间的情况下就能达到稳定训练效果；

- RMSNorm 的实现更简单，计算量小于 LayerNorm，对硬件友好。

类比：就像跑步前热身，让你跑得更快、避免受伤。
---
#### 3. 提升模型泛化能力
- 归一化后，模型不会偏向某些维度/特征；

- 输出分布在各维度上更均匀，有助于模型在验证集、测试集上表现更好。

---
## Positional Encoding
Reference：

https://jalammar.github.io/illustrated-transformer/

https://zhuanlan.zhihu.com/p/427388113

https://spaces.ac.cn/archives/10352

把一个词转换成向量，就好像把一个词映射到了一个高维空间的位置，意思相近的词会在高维空间内比较靠近，而加上位置向量，会让位置相近的词更靠近，位置远的词离得更远。为什么用cos，sin这种方式，使用sin和cos编码可以得到词语之间的相对位置。
### RoPE (Rotary Position Embedding)
![image]({{ "images/LHeYbyvZzoGNMXxvdnocjdFEn4L.png" | relURL }})

```python
# 传统方法：位置信息加到向量上
x_with_pos = x + pos_embedding

# RoPE：通过旋转变换编码位置
x_with_pos = rotate(x, position_angle)
# 对于位置m的token，其query/key向量被旋转θ*m角度
def rope_rotation(x, position, dim):
    # 计算旋转角度
    theta = 10000 ** (-2 * np.arange(0, dim, 2) / dim)
    
    # 位置m的旋转角度
    angles = position * theta
    
    # 构造旋转矩阵并应用
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    
    # 旋转变换（简化版）
    x_rotated = apply_rotation(x, cos_vals, sin_vals)
    return x_rotated
```
### multi-modality position多模态位置
#### 多模位置
多模态模型居然连位置编码都没有形成共识。对于文本LLM，目前主流的位置编码是[RoPE](https://spaces.ac.cn/archives/8265)（RoPE就不展开介绍了，假设读者已经熟知），更准确来说是RoPE-1D，因为原始设计只适用于1D序列。后来我们推导了[RoPE-2D](https://spaces.ac.cn/archives/8397)，这可以用于图像等2D序列，按照RoPE-2D的思路我们可以平行地推广到RoPE-3D，用于视频等3D序列。

然而，以上说的只是单一模态输入，当多种模态混合输入时，困难就出现了：文本是1D序列，所以它的位置只是一个标量nn；图像是2D的（“宽”和“高”），所以表达它的位置需要一个二维向量(x,y)(x,y)；视频则在图像的基础上新增了一个时间维度（或者说“帧”），所以它的位置是一个三维向量(x,y,z)(x,y,z)。当我们希望用同一个模型去处理三种模态的数据时，就要想办法糅合这三种不同形式的位置信息。

RoPE在实现上是绝对位置编码，但结合基于内积的Attention来用时，内积之后位置会自动作差，（想象两个向量做内积只跟夹角有关，一个意思）从而实现了相对位置编码的效果。可同一大小的向量可以作差，不同大小的向量怎么作差呢？这就是多模态位置编码的困难所在。

不少工作选择“逃避”这个困难，直接Flatten所有模态然后使用RoPE-1D，这不失为一种解决办法，但终究显得不够优雅。此外，强行Flatten也可能会降低模型性能的天花板，因为[《VisionLLaMA: A Unified LLaMA Backbone for Vision Tasks》](https://papers.cool/arxiv/2403.00522)等工作已经表明，RoPE-2D的引入有助于提升模型效果尤其是变分辨率输入的效果。
![image]({{ "images/RxRzbjNHuooidzx93ZAcftCVnrf.png" | relURL }})
---
## CLS token
Reference:

https://arxiv.org/pdf/1810.04805

https://h2o.ai/wiki/classify-token/
### basic concept
Classify token ([CLS]) is a special token used in NLP and ML models, particularly those based on the Transformer architecture. It is a token that represents the entire input sequence or sentence and is placed at the beginning of the input.CLS = Classification Token，是BERT等Transformer模型中的一个特殊标记，专门用于获取整个序列的聚合表示。
### how it works
Classify token ([CLS]) serves as an input representation for the classification tasks in NLP and ML. It encapsulates the information from the entire input sequence and carries it through the model's layers for further processing. The model then uses this representation to make predictions or classify the input into predefined categories.

Classify token ([CLS]) plays a crucial role in NLP and ML tasks as it enables the model to perform classification on textual data. By incorporating the entire input sequence into a single representation, the model can capture important context and semantic information that aids in accurate classification. It helps the model understand the relationship between different words and their impact on the overall meaning of the text.
### Use Cases
- Sentiment analysis: Determining the sentiment (positive, negative, or neutral) of a given text.

- Text categorization: Classifying documents or articles into predefined categories.

- Intent recognition: Identifying the intent or purpose behind a user's input in conversational AI systems.

- Named entity recognition: Identifying and classifying named entities such as names, organizations, locations, etc., in text.

### 与普通token的区别
以：The cat is running 为例
#### CLS Token的特殊身份
想象一个会议室里的讨论场景：

普通token（如"cat", "running"） = 会议中的普通与会者

CLS token = 会议的主持人/记录员
#### CLS Token与普通Token的根本区别
##### 位置特殊性
```plaintext
普通Token: 在句子中有具体的语义位置
输入: "The cat is running"
- "The" 在位置1，表示限定词
- "cat" 在位置2，表示主语  
- "is" 在位置3，表示谓语

CLS Token: 永远在位置0，不表示任何具体语义
输入: "[CLS] The cat is running"
- "[CLS]" 在位置0，是个"空容器"，等待装入信息
```
##### 初始状态差异
```plaintext
普通Token的初始状态:
- "cat" 的embedding包含关于"猫"的语义信息
- "running" 的embedding包含关于"跑步"的动作信息

CLS Token的初始状态:
- 只是一个随机初始化的向量
- 不包含任何预定的语义信息
- 像一张白纸，等待被写入内容
```
#### 注意力机制中的不同角色
##### 第一层的注意力模式
普通token "cat" 的注意力:
```plaintext
"cat" 作为query时关注:
- "The" (30%) - 了解这是特指的猫
- "cat" (40%) - 保持自身信息  
- "is" (20%) - 理解动作关系
- "running" (10%) - 知道在做什么

目的: 丰富自己的语义理解
```
CLS token 的注意力:
```plaintext
"[CLS]" 作为query时关注:
- "The" (20%) - 收集限定信息
- "cat" (30%) - 收集主语信息
- "is" (20%) - 收集谓语信息  
- "running" (30%) - 收集动作信息

目的: 平等地收集所有信息，不偏向任何特定词汇
```
##### 深层的注意力演进
到第6层时:

普通token "cat":
```plaintext
现在的"cat"已经知道:
- 自己是主语
- 正在执行"running"动作
- 在一个完整的句子中

注意力更加精准:
- 主要关注与自己语法相关的词 (60%)
- 适度关注其他内容 (40%)
```
CLS token:
```plaintext
现在的"[CLS]"已经理解:
- 整个句子的语法结构
- 主要的语义关系
- 句子的整体含义

注意力变得有选择性:
- 重点关注核心内容词 "cat"(40%) + "running"(40%)  
- 较少关注功能词 "The"(10%) + "is"(10%)
```
#### 信息聚合过程的直观对比
##### 信息流向的差异
普通token的信息更新:
```plaintext
第1层: "cat" = 原始"cat"语义 + 少量上下文
第6层: "cat" = 丰富的"cat"语义 + 大量上下文
第12层: "cat" = 完整的上下文化"cat"表示

特点: 始终以"cat"的语义为核心，向外扩展
```
CLS token的信息更新:
```plaintext
第1层: "[CLS]" = 空白 + 一点点各种信息
第6层: "[CLS]" = 句子结构 + 主要语义关系  
第12层: "[CLS]" = 完整的句子级表示

特点: 从空白开始，逐步装入整个句子的精华
```
#### 为什么CLS Token能代表整个句子
##### 信息无损聚合的原理
普通token的局限:
```plaintext
每个普通token都有"自我中心"的倾向：
- "dog" 主要关心与狗相关的信息
- "playing" 主要关心动作相关的信息  
- "garden" 主要关心地点相关的信息

如果让"dog"代表整个句子 → 会偏向动物信息
如果让"playing"代表整个句子 → 会偏向动作信息
```
CLS token的优势:
```plaintext
CLS token没有预设的语义偏好：
- 不会偏向任何特定类型的信息
- 可以平等地聚合所有类型的信息
- 专门训练来承担"全局总结"的职责

就像一个专业的会议记录员，不会因为个人喜好
而偏重记录某些内容，而是客观全面地记录
```
##### 梯度更新的特殊性
普通token的更新目标:
```plaintext
"dog" 的梯度目标：
- 更好地表示"狗"这个概念
- 更好地理解自己在句子中的作用
- 保持与"狗"相关的语义特征

但这些都有"自我中心"的倾向
```
CLS token的更新目标:
```plaintext
"[CLS]" 的梯度目标：
- 更好地预测句子的整体标签（如情感、类别）
- 更好地表示句子的整体语义
- 学会如何从各个token提取最重要的信息

没有"自我中心"，完全为了整体效果而优化
```
---
## BERT
Reference:

https://zhuanlan.zhihu.com/p/360343071

http://arxiv.org/pdf/1810.04805



Transformer (2017)

    ├── Text Domain

    │   └── LLM (GPT, BERT, T5...)

    │

    ├── Vision Domain    │   └── ViT (2020) → Vision Transformers

    │

    ├── Generation Domain

    │   └── DiT (2022) → Diffusion + Transformer

    │

    └── Multimodal Domain

        ├── VLM → Vision + Language

        └── VLA → Vision + Language + Action
###  (Masked Language Model, MLM)
```python
# 核心思想：随机掩盖输入中的部分token，预测被掩盖的内容
原始句子: "The cat sat on the mat"
掩码处理: "The [MASK] sat on the [MASK]"
预测目标: 预测第2个位置是"cat"，第6个位置是"mat"

# 具体实现
def create_masked_lm_predictions(tokens, masked_lm_prob=0.15):
    """
    masked_lm_prob: 掩盖15%的token
    """
    output_tokens = list(tokens)
    masked_lm_positions = []
    masked_lm_labels = []
    
    for i, token in enumerate(tokens):
        if random.random() < masked_lm_prob:
            masked_lm_positions.append(i)
            masked_lm_labels.append(token)
            
            # BERT的巧妙设计：
            if random.random() < 0.8:
                output_tokens[i] = "[MASK]"     # 80%: 替换为[MASK]
            elif random.random() < 0.5:
                output_tokens[i] = random_token  # 10%: 替换为随机token  
            # else: 保持原token不变              # 10%: 保持不变
    
    return output_tokens, masked_lm_positions, masked_lm_labels
```
为什么这样设计？
```python
# 问题：如果总是用[MASK]替换，微调时会遇到分布偏移
# 训练时: "The [MASK] is running"
# 微调时: "The dog is running"  # 没有[MASK] token

# 解决方案：10%保持不变 + 10%随机替换
# 让模型学会在没有明确掩码信号时也能理解上下文
```
### (Next Sentence Prediction, NSP)
```python
# 目标：理解句子间的关系
def create_nsp_data(documents):
    examples = []
    
    for doc in documents:
        for i in range(len(doc) - 1):
            # 50%的正样本：连续的两个句子
            if random.random() < 0.5:
                sentence_a = doc[i]
                sentence_b = doc[i + 1]
                label = 1  # IsNext
            
            # 50%的负样本：来自不同文档的句子
            else:
                sentence_a = doc[i]
                sentence_b = random.choice(random.choice(documents))
                label = 0  # NotNext
            
            examples.append((sentence_a, sentence_b, label))
    
    return examples

# 输入格式
input_format = "[CLS] sentence_A [SEP] sentence_B [SEP]"
# CLS token的输出用于NSP分类
```
### Pre-trained Objective Function
```python
# 联合训练两个任务
def bert_pretraining_loss(model_output, masked_positions, masked_labels, nsp_labels):
    # MLM损失：只计算被掩盖位置的损失
    mlm_logits = model_output.prediction_logits[masked_positions]
    mlm_loss = cross_entropy(mlm_logits, masked_labels)
    
    # NSP损失：基于CLS token的输出
    nsp_logits = model_output.seq_relationship_logits
    nsp_loss = cross_entropy(nsp_logits, nsp_labels)
    
    # 总损失
    total_loss = mlm_loss + nsp_loss
    return total_loss
```
### Dual-direction encoding implementation
```python
# 与单向模型的对比
# GPT (单向): 只能看到左边的context
attention_mask = [
    [1, 0, 0, 0],  # token1只能看到自己
    [1, 1, 0, 0],  # token2能看到token1和自己  
    [1, 1, 1, 0],  # token3能看到前面所有
    [1, 1, 1, 1]   # token4能看到前面所有
]

# BERT (双向): 可以看到所有位置的context
attention_mask = [
    [1, 1, 1, 1],  # 每个token都能看到所有其他token
    [1, 1, 1, 1],
    [1, 1, 1, 1], 
    [1, 1, 1, 1]
]
```
### Encoder vs Decoder
1. 信息约束：Encoder无因果约束，Decoder有严格因果约束

1. 优化目标：Encoder优化理解质量，Decoder优化生成概率

1. 计算模式：Encoder并行处理，Decoder顺序处理

1. 设计哲学：Encoder以理解为中心，Decoder以生成为中心

非mask注意力和mask注意力机制


---
## VLM
Reference：

https://zhuanlan.zhihu.com/p/701039113


---
## POPE
Reference:

https://zhuanlan.zhihu.com/p/699623105

https://arxiv.org/pdf/2305.10355
![image]({{ "images/XUZubPm4KoCUj9xRvbZcvnBfnqe.png" | relURL }})
 a more suitable method for the stable, fair and flexible object hallucination evaluation of LVLMs, namely pollingbased object probing evaluation (POPE). Specifically, POPE formulates the evaluation of object hallucination as a binary classification task that prompts LVLMs to output “Yes” or “No”, e.g., “Is there a chair in the image?”. In this way, by sampling objects that LVLMs are prone to hallucinate, we can construct a set of hard questions to poll LVLMs. As standard answers to these questions are just “Yes” or “No”, we can easily identify them without complex parsing rules, and avoid the influence of instruction designs and caption length, thus guaranteeing stability, fairness and flexibility
## KV cache
## GGML
Reference:

https://huggingface.co/blog/introduction-to-ggml

ggml 默认计算方式：调用 `ggml_mul_mat()`

在 `llama.cpp` 中的前向传播过程中，比如线性层（FullyConnected Layer）会构造：
```cpp
ggml_mul_mat(A, B)
```
这构建了一个“图节点”，之后由：
```cpp
llama_graph_compute(graph)
```
## Gumbel-Softmax
---


## Low-rank Attention

Nadaraya–Watson Regression（纳达拉亚–沃森回归）是一种非常经典的非参数回归方法，本质上是加权平均法，用于拟合数据的平滑曲线，尤其常见于**核回归（kernel regression）**和一些深度学习中的可解释性模块。
---
### Maths explanation
假设我们有一组训练样本 (xi,yi)(x_i, y_i)，现在想预测一个新点 xx 的函数值 y(x)y(x)，Nadaraya–Watson 回归形式为：

$$
y^(x)=∑i=1nK(x,xi)⋅yi∑i=1nK(x,xi)\hat{y}(x) = \frac{\sum_{i=1}^n K(x, x_i) \cdot y_i}{\sum_{i=1}^n K(x, x_i)}
$$


其中：
- K(x,xi)K(x, x_i) 是一个核函数（kernel function），常用高斯核：

- K(x,xi)=exp⁡(−(x−xi)22h2)K(x, x_i) = \exp\left(-\frac{(x - x_i)^2}{2h^2}\right)

- hh：称为带宽参数（bandwidth），控制“邻近范围”大小。

---
**Nadaraya–Watson 回归 = 用邻近样本的加权平均来平滑预测结果，是核方法的基础形式，类似软注意力机制。**

---
## Cover's theme
https://kexue.fm/archives/7546
---
## GPU storage
https://zhuanlan.zhihu.com/p/29264672961

https://zhuanlan.zhihu.com/p/462191421

https://github.com/CalvinXKY/BasicCUDA/tree/master/memory_opt
---
## pip install
在使用 `pip install` 命令时，“后缀 option（-...）”实际上是指 命令行选项（command-line options），它们以 `-` 或 `--` 开头，用于控制 `pip install` 的行为。这些选项可以改变安装方式、指定源、启用特定功能等。
### Options

| 选项                             | 含义                                                                                      |
| ------------------------------ | --------------------------------------------------------------------------------------- |
| `-r, --requirement <file>`     | 从文件中读取要安装的包列表（通常是 `requirements.txt`）<br>例：`pip install -r requirements.txt`            |
| `-e, --editable <path/url>`    | 以“可编辑模式”安装包（开发模式），修改源码立即生效<br>例：`pip install -e .`                                      |
| `-i, --index-url <url>`        | 指定包的索引源（PyPI 镜像地址）<br>例：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy` |
| `--extra-index-url <url>`      | 添加额外的包索引源                                                                               |
| `--no-index`                   | 不使用任何索引，只从本地目录或 `--find-links` 安装                                                       |
| `-f, --find-links <url/path>`  | 指定查找包的额外路径（本地或网络）<br>例：`pip install -f ./packages/ mypackage`                           |
| `--no-deps, --no-dependencies` | 安装包但不安装其依赖项<br>例：`pip install --no-deps requests`                                       |
| `--force-reinstall`            | 重新安装包，即使它已经存在                                                                           |
| `--upgrade, -U`                | 升级包到最新版本<br>例：`pip install --upgrade package_name`                                      |
| `-t, --target <dir>`           | 将包安装到指定目录，而不是当前环境                                                                       |
| `-c, --constraint <file>`      | 指定版本约束文件，限制版本升级范围                                                                       |
| `--user`                       | 安装到用户目录（避免使用 `sudo`）<br>例：`pip install --user package_name`                             |
| `--proxy <proxy>`              | 使用代理访问网络                                                                                |
| `--timeout <sec>`              | 设置连接超时时间                                                                                |
| `-v, --verbose`                | 增加输出详细程度（可多次使用 `-vvv`）                                                                  |
| `-q, --quiet`                  | 减少输出信息                                                                                  |
| `--pre`                        | 允许安装预发布版本（如 alpha、beta、rc）<br>例：`pip install --pre package_name`                        |
| `--no-cache-dir`               | 禁用缓存，强制重新下载包                                                                            |

### further use
Reference：https://zhuanlan.zhihu.com/p/673336277
---
## Normalization
Reference: https://www.cnblogs.com/wuliytTaotao/p/10837533.html

Reference: https://0809zheng.github.io/2020/03/03/regularization.html
---
## MLE
最大似然估计
---
## Alignment of Pytorch, CUDA, Python
### Supported newest CUDA version
```plaintext
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
### [torch cuda](https://pytorch.org/get-started/previous-versions/) alignment
### flash-attn alignment
首先检查你的cuda版本，通过nvcc -V查看环境是否含有cuda以及版本是否在11.6及以上，如果没有需要自己安装，下载地址在这里：[cuda-toolkit](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/cuda-toolkit-archive)

https://github.com/Dao-AILab/flash-attention/releases在这个里面找到对应的版本下载即可
## Safetensor, Config, Download to local
```python
cd ../etc
source network_turbo
```
```python
pip install -U huggingface_hub
apt install aria2c
# sudo apt install aria2c
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh model_name --tool aria2c -x 10
```
```python
scp -P xxxxx -r user@xxxx:/path/to/your/file ./path/
# 不推荐传大文件，小的可以，从本地传到服务器远端
scp -v #detail info
scp -q # quiet
```
```python
wget # Web get
aria2c # 多线程
# 优先用后者，get address的时候注意不要cp了web的地址，而要raw file的地址
aria2c https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json
# detailed 
https://www.cnblogs.com/TangQF/articles/18714419
```
## Dataset
[MME](https://huggingface.co/datasets/lmms-lab/MME) / [POPE](https://huggingface.co/datasets/lmms-lab/POPE) / [textvqa](https://huggingface.co/datasets/lmms-lab/textvqa) ...

https://huggingface.co/collections/lmms-lab/lmms-eval-661d51f70a9d678b6f43f272

在这里找多模态模型的所有数据集去下载，别在github里下通常不是放真正数据集的仓库
## Vim Use
**全选（高亮显示**）：按esc后，然后ggvG或者ggVG

**全部复制：**按esc后，然后ggyG

**全部删除：**按esc后，然后dG

解析：

**gg：**是让光标移到首行，在**vim**才有效，vi中无效 

**v ：** 是进入Visual(可视）模式 

**G ：**光标移到最后一行 

**选**中内容以后就可以其他的操作了，比如： 
**d**  删除**选**中内容 
**y**  复制**选**中内容到0号寄存器 
**"+y**  复制**选**中内容到＋寄存器，也就是系统的剪贴板，供其他程序用
## cp/mv PATH
```markdown
cp 命令

  # 情况1：复制目录内容
  cp -r source_dir/ dest_dir/
  # 结果：source_dir里的所有文件复制到dest_dir里

  # 情况2：复制整个目录
  cp -r source_dir dest_dir/
  # 结果：在dest_dir里创建source_dir文件夹
mv 命令
移动到目录内
  mv file.txt dir1/        # file.txt移动到dir1目录里
  mv file.txt dir1/dir2/   # file.txt移动到dir1/dir2目录里

重命名
  mv file.txt newname.txt  # 重命名文件
  mv dir1 newdir          # 重命名目录
```
## Ridge regression  &&  Square loss
1. 岭回归是：假设参数w有高斯先验（prior），从而改变MAP最大后验，在求解的时候约束参数分布

1. `square loss`是：如果假设误差服从 **高斯分布**，那么用 **最大似然估计**`**(MLE)**` 来推参数，就自然等价于最小化 **平方损失**。一般都会默认噪声是gaussian分布的，这样得到的普通最小二乘回归；但缺点也很明显就是平方损失对 outlier 太敏感。实际应用里，经常需要根据数据特点选择更合适的噪声模型，从而得到更鲁棒的损失函数。

1. 似然更像是从数据推参数：概率是已知参数下看数据的可能性；似然是已知数据下看参数的可能性。

似然：数据给参数的证据。

先验：数据之前你对参数的信念。

后验：结合两者，数据之后你对参数的新信念。




## for circle
首先看当前循环头部和循环头部的差值，就是i+=？；

再看循环几次，就是i max（！）=n*？；

再看次要条件，可以考虑将？和！都缩小一半；然后循环内主索引扩一倍。







$$
d_{L2}(x,y) = \sqrt{\sum_{i=1}^{d}(x_i - y_i)^2}
$$


$$
d_{cosine}(x,y) = 1 - \frac{x \cdot y}{||x|| \cdot ||y||}
$$



















































