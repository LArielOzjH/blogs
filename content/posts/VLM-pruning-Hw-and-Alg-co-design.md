## **VisionZip: Longer is Better but Not Necessary in Vision Language Models**
---
### Background

### Algorithm

此外也是采用了merge，整篇文章没什么突出的算法核心点，但是工程、实验做的很扎实
## **[CLS] Attention is All You Need for Training-Free Visual Token Pruning: Make VLM Inference Faster**
![image]({{ "images/R2UWbLa34oFCItxsmIAcBwchnTc.png" | relURL }})
---
### **Background Analysis：**_**attention shift **_**and **_**attention dispersion**_
1. _**Attention shift: **_** a tendency for textual attention to focus more on later parts of the visual token sequence, which is not desirable for preserving valuable visual information.**


1. _**Attention dispersion:  **_**refers to the less concentrated attention distribution within the LLM compared to the visual encoder.**


### **Methods**
![image]({{ "images/DgQMb6qIco9naKxhiWzc6TAHnWh.png" | relURL }})
核心的想法是用CLS attention 来决定prune掉的token（patch），经过 Visual Encoder 之后，取CLS attention，后R%的被prune掉（提到一个动态阈值的公式，没啥用），认为这些部分的patch对于整体的语义贡献很低。整体就是简单的思路采用CLS token的attention score去处理。

这里的问题就是到底是否能真正以CLS token的attention值来说明真正的importance然后做prune，这样做基本不需要硬件的额外支持
### **Scores**
![image]({{ "images/VOZFbT9N3o9iaBxCQUicNL47nKe.png" | relURL }})
从ablation来看整个论文的思路基本都是在考虑：
1. prune的位置：LLM浅层 OR LLM之前（visual encoder之后）

1. 判断prune token的方式：[CLS] token attention OR random OR patch attention

没有考虑的地方：
1. head attention ：加入head的充分考虑，比如HeatViT那种加入head score的or so

1. prune的只能是token-wise的么？有论文会在d维度做适当的prune，或者head维度做prune，也都是一些常见的思路，另外结构化稀疏性肯定是要优先考虑的，比如可以细粒度结构化之类的，参考sanger非常经典，但是做的也比较多

1. 大多论文都采用prune+merge的模式，能不能改变这个模式，能把prune掉的token information利用起来（如果是token-wise的话）或者充分利用prune掉的其他信息，比如用可学习的小型网络等来学习一些模式，把prune掉的用比较efficient的方式恢复出来。

1. 有没有可能摆脱importance做prune的选择，比如#20DiVprune是一个很取巧的方式

1. text-agnostic肯定是一种更为accuracy-friendly的算法，那么能不能有一种方法，先prune掉一部分token（if token-wise)然后再想办法根据text token去恢复或者更好的提取non-informative token或者进一步做更合适的prune，是一个2-stage的方法

1. 可以调查一下对于VLM计算复杂度最大的地方，二次方的attention肯定是要解决的，可是如果能有办法优化后续的大规模的FFN那肯定是更efficient的

## **SPARSEVLM: VISUAL TOKEN SPARSIFICATION FOR  EFFICIENT VISION-LANGUAGE MODEL INFERENCE_VVVI**
---
### Background & Target
we propose an efficient training-free token optimization mechanism dubbed SparseVLM without extra parameters or fine-tuning costs. Concretely, given that visual tokens complement text tokens in VLMs for linguistic reasoning, we select visual-relevant text tokens to rate the significance of vision tokens within the self-attention matrix extracted from the VLMs. Then we progressively prune irrelevant tokens.

SparseVLM 是一个训练免（training-free）、文本引导（text-aware）的视觉 token 稀疏化框架：
 先挑出“和图像强相关”的文本 token 当评审（raters），用它们在解码器里的跨模态注意力给每个视觉 token 打分；然后逐层按打分和一个秩（rank）自适应规则删掉不重要的视觉 token；被删的 token 不直接丢，而是回收—聚类—重构成少量“压缩 token”再放回，尽量不丢信息。这样能显著减 FLOPs/显存/时延，同时保持高精度。
### Algorithm
非常好的一个text-agnostic算法，大概率我最后会follow这个做一些工作
#### Raters 
text-aware的核心是根据text tokens有针对性的对visual tokens做pruning，本文的一个重要算法是首先在进入LLM之前进行一次raters的选择，raters是一部分text tokens，叫做“评委”，通常来说一句text中真正对图像token选取有指导意义的并没有几个，比如大量的冠词，介词等都是不重要的可以忽略的，因此要选择raters，先从整句文本里评估每个词和视觉的关联度$$
R
$$
，只保留高于平均值 $$
m=Mean(R)
$$
 的候选词当评委，减少无关词对打分的噪声；评委集合用策略 $$
S
$$
 确定，整步只在进入解码器之前做一次，开销 $$
O(Lt\ · Lv\  · D)
$$

#### Rate the visual tokens with scores
对每个视觉 token j，把评委（文本）对它的注意力分数按行求平均，得到第 j 个视觉 token 的综合价值分，然后根据这个得分进行prune，具体prune掉多少个呢，用到了一个rank-based的方法，比如如果P矩阵是接近满秩的话，那么就说明都是线性无关的，就基本不需要prune，保留基本所有visual tokens。如果P是低秩的话，那么就多prune掉一些，根据注意力分数。 $$
N=α⋅(Lv−Rank(P))
$$

#### Merge
也是按照分数merge，做一次k 近邻密度峰聚类：


## **MADTP: Multimodal Alignment-Guided Dynamic Token Pruning for  Accelerating Vision-Language Transformer**
---
![image]({{ "images/YRIEbmTgqorENvx2tmDcVrIbnff.png" | relURL }})
### **MAG**


### **DTP**
**事实证明，单模态压缩中的动态令牌修剪比静态令牌修剪更有效，因为它可以根据输入实例的复杂程度自适应调整模型的压缩率。**
![image]({{ "images/GZqrbdyPSoDvgAx8octcFCM0nvE.png" | relURL }})
### **TIS**
$$
TIS = (Scls + Sself + Stoken)/3                      



$$


**比如拿 visual 模态来举例：**


## **VScan: Rethinking Visual Token Reduction for Efficient Large Vision-Language Models**
![image]({{ "images/PzhNbrL9SocCajxN5CncqrLDnJe.png" | relURL }})
---
### **Observations**
1. **In the visual encoding stage, the visual encoder attends to locally significant tokens in the shallow layers, focusing on fine-grained local details, while at deeper layers, it gradually shift its focus to a highly condensed set of tokens that encapsulate broader global context;（visual encoder里面随着层数增多，注意力呈现的变化趋势,下面两个图我自己可视化了一下，可以看到attention focus由广泛到集中，但问题是集中的部分并不完全是我们人眼所普遍认为的focus）**


1. **In the LLM decoding stage, early layers exhibit strong positional bias toward visual tokens appearing later in the sequence, neglecting their semantic relevance; as the layers deepen, cross-modal interactions begin to emerge, and output token probabilities typically converge in the mid-to-late layers where visual information is more effectively integrated into the language stream.（和FsterVLM里面提到的attention dispersion是一个东西）**

### **Current methods limitations**
**可以看到这里用了这样一张图片和query来disable之前的几种主流方法**

![image]({{ "images/ZtWWbX6aHo8m1Cx9xQdcZulHnEe.png" | relURL }})

**然后做了3个study：各种可视化去探究LLM对于textual和visual信息的处理随着层数变化的改变情况**
![image]({{ "images/OQvkb3nnuopOgrxweM2cPr0sn5I.png" | relURL }})
**左边的图其实就是attention shift的可视化（位置编码的影响），随着LLM layers增多，这种现象逐渐diminish**

**右边的图：We observe that the middle LLM layers are primarily responsible for interacting with the visual tokens, whereas the early and deep layers focus predominantly on processing textual information.就是LLM的中间的那些layers会更倾向于结合visual信息聚合处理，然而shallow/deep layer都会focus更多在textual信息上**
![image]({{ "images/IO9vbikGqoUEXwxNisucXMZHnG0.png" | relURL }})
**We observe that in more challenging open-ended tasks like GQA, the next-token predictions stabilize around LLM layer 20, whereas in simpler yes/no tasks such as POPE, the predictions converge earlier, around LLM layer 16.**

**在这一部分得出的结论就是：LLM的early layers并不是最适合pruning的层数位置，因为**
1. ** positional bias；**

1. ** limited engagement of visual content**

**而middle layers才是最好的，因为：**
1. **better preserves critical cross-modal interactions**

1. **minimizes disruption to model predictions（太深层起不到太好的pruning效果而且可能会有disrupt）**

### **Methods**
![image]({{ "images/MtSRbnAfHoWKSexJlXGctCgDn4e.png" | relURL }})
#### **Reducing Visual Redundancy via Complementary Global and Local Scans**
##### **Global Scan**
**因为visual encoder的最后一层（或者倒数第二层）是global content，所以采用跟之前工作类似的方法CLS attention的方式来选择 global tokens （g）**
##### **Local Scan**
**因为前面已经分析过了直接使用CLS attention获取全局的信息会使得一些重要的细节信息遗漏掉，所以还要在visual encoder的浅层来获取一些finer的细节信息，具体选择方式是按照windows去做选择（l），g和l的数量被控制为一样**
##### **Token Merging**
**把未被选择的那些token与被选择的token做内积，对每个unselected token选择最相似的selected token并且做average merge得到最终的merged representation（感觉这个地方在arch层面可以考虑适当的优化）**
#### **Reducing Textual Irrelevance via Middle Layer Pruning**
**Further refine the token set based on their relevance to the text query**

### **Experiment Scores**


## **Skip-Vision: Efficient and Scalable Acceleration of Vision-Language Models via Adaptive Token Skipping**
![image]({{ "images/IjGpbXnSroa5uoxmbiAcdZzIn9g.png" | relURL }})
---

## **Accelerating Pre-training of Multimodal LLMs via Chain-of-Sight**
![image]({{ "images/DI3hbanZloB0KCxy4iccxpavnEe.png" | relURL }})
---
### **Brief Introduction **
**用更少的visual tokens训练通常意味着perfomance的下降，那么有没有一种方式能够解决这个问题，用更少的visual tokens去包含更多的信息，且能不受input resolution的影响，从而实现更efficient的pre-training？**

**CoS(chain of sight)就是这样一个方式，这是一个vision-language bridge的模块，整体的思路有点类似之前的Perceiver抑或是Q-former，但还有一个特点是对预训练和微调部分使用的token有很大差别，后者使用更多更fine的tokens，以获取更finer的vision信息，从而弥补perfomance可能的掉点**
![image]({{ "images/HSSVbssrXodcvmxT2z4ciYR5n8d.png" | relURL }})
```plaintext
The core mechanism is our multi-scale visual resampler, which produces visual tokens 
of multiple visual scales. Inspired by the classical concept of multi-scale feature hierarchy in visual 
understanding [105, 41, 32, 106, 82, 52], we partition the visual features produced by the visual 
backbone using windows of multiple sizes. For each window size, a visual resampler is implemented 
to produce a specified number of visual tokens per window. Subsequently, the visual tokens from 
various window sizes are gathered and linked in a global-to-local manner, forming a chain of reasoning 
steps from coarse views gradually to fine-grained perspectives.
On top of this, we propose a post-pretrain token scaling strategy, which compounds the elements of 
input resolution and window size manipulation to enable a significant escalation in the token count 
for our Chain-of-Sight, reaching up to 16× increase during fine-tuning. Such adaptability allows for 
the fine-tuning of the model with a flexible granularity or complexity as required, without the the 
necessity for an additional pre-training phase.


图像 → ViT视觉编码器 → Multi-Scale Visual Resampler → 少量 token 输入语言模型 → 预训练  
                                               ↓  
                      （微调时再放大 token 数量） → 多 token 输入语言模型 → 下游任务微调

```
### **Methods Delineate**
#### **Multi-scale visual Resamplers**


#### **Post-Pretrain Token Scaling**
![image]({{ "images/BPBibeDZZokNk6xe0LycsqwXnuc.png" | relURL }})
1. **在预训练中只用少量视觉 token（如 32/80），大幅加速训练；**

1. **微调阶段再将 token 数扩大（通过调整输入分辨率 + 减小窗口 size）；**

1. **提出 compound scaling：resolution scaling × window scaling，最多可将 token 数扩大 16 倍（如 32 → 512）；**

```plaintext
Step 1: 预训练阶段
    图像分辨率低（224×224）
    window size 粗（16、8）
    每个 window 分配少量 query
    → 得到少量（如 32、80）视觉 token
    → 快速预训练

Step 2: 微调阶段
    提高分辨率（如 448×448）
    引入更小的 window size（8、4、2）
    每个 window 分配更多 query
    → 得到更多（如 336、528、1296）视觉 token
    → 细粒度理解图像，提高任务性能
```
## **HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers**
---
### Background & Target
_**HeatViT**__: _While vision transformers (ViTs) have continuously achieved new milestones in the field of computer vision, their sophisticated network architectures with high computation and memory costs have impeded their deployment on resource-limited edge devices

we propose a hardware-efficient image-adaptive token pruning framework called HeatViT for efficient yet accurate ViT acceleration on embedded FPGAs.

_**SPViT: **_high computation and memory cost

a dynamic attention-based multi-head token selector, which is a lightweight module for adaptive instance-wise token selection. We further introduce a soft pruning technique, which integrates the less informative tokens chosen by the selector module into a package token rather than discarding them completely

project: https://github.com/PeiyanFlying/SPViT
### Algorithm
#### _Head-Evaluation Multi-Head __Token Classifier_
_**SPViT:**_

整体呢就是采用一个可学习的网络（主要就是几层很小的MLP，实际计算量不足ViT的1%），用这个网络去学习keep/prune的规则、模式，算法思路上就是每个head关注的特征和部分是不一样的，也就是每个head本身对于所有token是自带一定注意力的，可以学习到head score那么自然就会想到在最后concat的时候采用加权平均的方式，也就是用 $$
head\ score \times token \ score \ for each \ head
$$
来表示
![image]({{ "images/UeyTbGK3WooyitxO2oMcFFf1nCd.png" | relURL }})
1. $$
MLP_1:LayerNorm \rightarrow Linear(d,d/2) \rightarrow GELU
$$

1. $$
MLP_2:Linear(d,d/2) \rightarrow GELU \rightarrow Linear(d/2,d/4) \rightarrow GELU \rightarrow Linear(d/4,2)
$$

1. $$
MLP_3:Linear(H,H/2) \rightarrow GELU \rightarrow Linear(H/2,H) \rightarrow Sigmoid
$$

1. $$
f_i^{local}=MLP_1(x_i) \in \mathbb{R}^{N \times d/2} 
$$

1. $$
f_i^global=AvgPool(MLP_1(x_i),D) \in \mathbb{R}^{1 \times d/2} 
$$

1. $$
f_i=[f_i^local,f_i^global] \in \mathbb{R}^{N \times d}
$$

1. $$
t_i=Softmax(MLP_2(f_i)) \in \mathbb{R}^{N \times 2}
$$

1. $$
\bar{X}=AvgPool(X) \in \mathbb{R}^{N \times H}
$$

1. $$
A=MLP_3(\bar{X})
$$

1. $$
\tilde{T} = \frac{\sum_{i=1}^{H} t_i \ast a_i}{\sum_{i=1}^{H} a_i} \in \mathbb{R}^{N \times 2}
$$

1. $$
D=GumbelSoftmax( \tilde{T}) \in \{0,1\}^N
$$

1-3是MLP的declaration，4-7是token score branch，8-10是head score branch

MLP都用降维，因为不是要提取更详细更复杂的特征信息，目标是生成一个简单、清晰、低维的策略 score，而不是保留原始语义。head score最后activation用了Sigmoid，为了便于加权

token score的部分，为什么保留到2维？这两维分别用于表示keep/prune的概率，这也是这个网络最核心期望学习到的特征，另一方面后续会用到Gumbel Softmax，为了反向传播学习，而不是直接用argmax等来决定是prune还是keep，Gumbel Softmax 是解决“可微分离散选择”的标准做法，SPViT 通过Gumbel Softmax实现了 token pruning 的 end-to-end 训练和部署闭环。

_**HeatViT:**_

#### Token Packaging Technique
做这部分packager的原因很简单：1. 直接prune的话仍然是会丢掉不少信息，所以想聚合一下，不直接丢掉信息；2. 另外在transformer里，earlier block受到这种info的丢失影响会更显著；3. 此外在“Instance localization for self-supervised detection pre-training”这篇文章提到了，background信息的过多剔除会导致self-attention提取关键特征的能力降低；4. 结合这些被prune掉的tokens的信息到一个统一的package token里，保有一定的语义信息


### Hardware 
硬件主要做了几部分优化：1. 控制流的设计去尽可能多的复用ViT已有的backbone部件；2. 并行化的优化，对于GEMM支持多头的并行处理；3. 优化非线性运算操作；4. LayerNorm是在CPU上做的
![image]({{ "images/SY4EbGvNooVYiHxdTXNcVt8LnVd.png" | relURL }})

### Limitations / Weakness / Further research 
HeatViT是用可学习的小型网络来选择prune/keep的token，但缺点就是对于特定的图片，在inference的时候整个网络的focus是固定的，但在VLM中实际的focus必然离不开prompt的语言token部分


## **ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention**
---
### Background & Target
we propose a first-of-its-kind algorithm-hardware co-designed framework, dubbed VITALITY, for boosting the inference efficiency of ViTs. Unlike sparsity-based Transformer accelerators for NLP, VITALITY unifies both low-rank and sparse components of the attention in ViTs.
### Algorithm
#### Mean-centered K


#### Taylor-softmax attention
ViTality想用线性注意力，选择采用taylor来近似exp()，理由很简单，进行了mean-center后的注意力分数大多集中在[-1,1]之间，而在0附近采用Taylor来做近似exp()是不错的选择，所以核心的思路就是用taylor展开的一阶项来近似注意力分数处于[-1,1]之间的weak connection部分，那自然会想到用高阶项来近似strong connection部分，但问题又出现了，高阶的计算及其复杂，很可能就offset掉linear attention带来的优势了，所以需要找方法替代高阶/strong connection的计算，ViTality选择的方法是用Sanger（调整threshold）来以Sparse attention的方式近似处理strong connection。所以一句话总结ViTality算法层面就是：用unified的Linear Taylor Attention以及Sparse Attention来替代传统的Softmax Attention来减小计算复杂度 $$
O(n^2) \rightarrow O(n)
$$
。
![image]({{ "images/FxezbG1ZAoCDCwxGO0DcsceKn3c.png" | relURL }})
另外注意：ViTality在训练阶段采用的是Linear+Sparse的形式做训练，后者能起到正则化的作用。但是在推理的时候仅仅考虑Linear的部分，而忽视掉了Sparse部分，定然会有些许的掉点但是考虑到其他因素可以接受。
### Hardware
micro-architecture
![image]({{ "images/N6ZsbmjUFoLOPLxe5BgcY1s3nqg.png" | relURL }})
####  多块式设计（Chunk-based Design）
- 不用一个可重构处理阵列去跑所有操作（这样开销大），而是分成多块：
	- 大阵列（SA-General）：负责大规模矩阵乘法，比如 `QG`、`K̂^T V`。
		- 小阵列（SA-Diag）：负责对角矩阵乘法（如 `Q k̂_sum^T`），乘法量小得多，只用一列PE。
		- 累加器阵列（Accumulator Array）：列方向求和，用于 `k̂_sum` 和 `v_sum` 等。
		- 除法阵列（Divider Array）：两种模式——单除数（均值化） & 多除数（Taylor 分子分母相除）。
		- 加法阵列（Adder Array）：元素级加/减（如均值中心化、Taylor 分子分母加法）。
	
优势：
- 专用单元避免大阵列跑小任务的浪费。

- 小阵列、轻运算单元功耗低、面积占用小。

- 不同块可并行执行，方便流水线。

---
####  四级存储层次（Memory Hierarchy）
- DRAM → SRAM → NoC → 寄存器（Regs）
	- DRAM：大容量存储
		- SRAM：片上缓存，减少DRAM访问
		- NoC：片内传输
		- Regs：每个计算单元局部寄存器，配合 systolic array 数据复用
	
- 数据复用优化：
	- 矩阵乘法时，让权重或中间结果在PE内驻留（stationary）以减少访存。
		- `V` 在计算 `K̂^T V` 时驻留，`G` 在计算 `QG` 时驻留。
	
---
#### 流水线创新（Intra-Layer Pipeline）


---
#### 数据流创新（Down-Forward Accumulation Dataflow）
![image]({{ "images/JSx0b7dyYoGy5Mx5eBycHMj2nPg.png" | relURL }})
- 常见两种数据流：
	Output Stationary：输出留在PE内（内累加）	Input Stationary：输入权重留在PE内（行/列移动，向下累加）
- ViTALiTy 的选择：
	- 全部矩阵乘法统一用 Input Stationary + Down-forward accumulation。
		- 好处：
		- 不用为不同矩阵乘法切换累加模式 → 简化PE设计。
			- 降低 systolic array 功耗（实验表明 systolic array 的能耗占总能耗大头）。
			- 代价：
		- `G` 不驻留 → 访存量增加，但总能耗反而下降（因为矩阵乘法能耗减少更多）。
		
### Limitations / Weakness / Further research
在处理弱连接的时候认为[-1,1]是weak，并用一阶来近似，有一个问题是注意力分布不见得都是大部分为弱连接的，可能随着输入patch或者是不同head等等的变化，甚至会出现大部分为强连接，那这时的掉点可以预想会很严重。------static 判定的不足。

稀疏部分在训练微调有用，但是在推理的时候用不到：说明sparse attention很可能只起到了正则化作用，而缺乏提供足够语义信息的能力---可能在一些强局部相关的任务会表现比较差




## **ViTCoD: Vision Transformer Acceleration via Dedicated Algorithm and Accelerator Co-Design**
---
### Background & Target
ViTs have a relatively fixed number of input tokens, whose attention maps can be pruned by up to 90% even with fixed sparse patterns, without severely hurting the model accuracy
### Algorithm
ViTCoD的核心算法是split and conquer & auto encoder
#### _ViTCoD’s Split and Conquer Algorithm / hardware_
ViTCoD采用fixed mask的稀疏格式，扔进去所有数据，算出每一个注意力分数矩阵$$
A
$$
并且求平均，根据这个平均值注意力分数矩阵$$
\bar{A}
$$
来构建一个fixed mask，并且在推理的时候直接用这个mask固定化稀疏模式


#### _ViTCoD Learnable Auto-encoder Module_


### Hardware
#### Why
- ViTCoD 的 S&C 让注意力变成固定且结构化的稀疏，再配合 AE 压缩 Q/K，给硬件提供了两类新机会：
 ① 固定稀疏图不再需要在线预测、控制更简单；② Q/K 可被压缩减少数据搬运。然而视觉里的稀疏注意力常沿对角线集中，会造成数据重用差、PE 利用率低且带宽受限，因此需要“算法+加速器”协同设计（文中 roofline 分析明确：仅稀疏还会更受带宽限制，必须再降通信）。

#### Dataflow：S-stationary vs K-stationary
- 论文比较了两种做 SDDMM(Q·Kᵀ) 的数据流：
S-stationary：把注意力得分“空间映射”到 PE 阵列，每个 PE 算一个分数——这对稀疏不友好：PE 利用率低、控制/重构开销大、还要在 PE 寄存器里存大量中间部分和做 intra-PE 累加。代表作 Sanger 用的就是它。
K-stationary：按列生成注意力分数，K 向量重用充分、中间缓冲小、且只按稀疏索引配对 Q/K 做乘法，天生更适合稀疏。但缺点是Q 访问更频繁——论文说这个缺点由 AE 压缩来缓解（少搬 Q）。结论：选 K-stationary。

#### Two-pronged
- 加速器由两套独立计算引擎组成：
Denser Engine 负责 SDDMM 的“取样致密”部分和后续 SpMM 的 S·V；
Sparser Engine 负责剩余的高度稀疏部分；
 两路有独立输出缓冲并行写回；芯片内还集成 Encoder/Decoder 引擎去配合 AE，先压再搬、到阵列前再解码。

#### Denser Engine
- 动态 PE 资源划分：不同层/头的全局 token 数不同，利用已知的固定 mask预估工作量，把 PE/MAC 按比例在 Denser/Sparser 之间分配。

- 并行与切片：各注意力头并行，但单行 PE 线不足以一拍完成 Q·Kᵀ，因此做细粒度切片并设计时空映射：
	- SDDMM(Q·Kᵀ)：采用 K-stationary，在特征维对 Q/K 切片并空间映射到 PEs，时间上让同一个 K依次与相关的 Q 相乘，并在 inter-PE 方向做部分和累加（按列生成注意力）。
		- SpMM(S·V)：转为 output-stationary，在token 维切片并空间映射，时间上沿特征维累加intra-PE 部分和，减少对注意力图的反复加载、显著降低 on-chip 缓冲压力。两种模式之间需要在 PE 线级别切换“inter-PE ↔ intra-PE”累加。
	
#### Sparser Engine
- 稀疏度可达 >90%，采用 CSC 索引格式预存非零位置，按列（契合 K-stationary 的“按列产出”）取索引，仅加载所需 Q/K；计算只遍历非零。

- Query-based Q forwarding：两路并行时，Sparser 侧需要的某些 Q 很可能 Denser 侧正在用，因此先查 Denser 的 Q 缓冲再决定是否从 off-chip 取，按需查询降低带宽。其余时空映射策略与 Denser 一致。

- 两路都内置 SoftMax 单元（计算完成单个分数后做 exp）和激活单元（ReLU 用门控，其他用 LUT）。

#### Encoder/Decoder （AE on-chip）
- 为配合 AE，芯片上实现了独立的 Encoder/Decoder 引擎（权重很小，如 6×3，可常驻片上）。Encoder 在 Q/K 线性投影之后立即启用，把 Q/K 压缩后再写回 off-chip；Decoder 在加载入阵列前恢复维度。两者都能和数据搬运全流水重叠，空闲时其 PE 线可复用于其他计算。

#### on-chip buffer and control module
- 两路引擎都配置了专用缓冲：输出(OBuf)、权重(WBuf)、K/S 缓冲、索引缓冲(IdxBuf)、Q/V 缓冲(Q/V Buf)，多端口并行读写以增强复用；矩阵乘法控制器可在致密/稀疏两类负载间切换；并带 SoftMax/Activation 功能单元。

#### 编译与可重构（Algorithm-Hardware Interface）
- 给定经过 ViTCoD 训练后的稀疏 ViT 层，网络解析器抽取“全局 token 数、缓冲大小、数据流”等配置，交给硬件编译器生成指令，指导加速器在 Denser/Sparser 之间重分配 on-chip 内存和 PE/MAC，并在 Q·K 与 S·V 两阶段切换 inter-PE ↔ intra-PE 累加模式。一次编译、多次复用摊薄重构成本。

#### 端到端数据流（推理时，一次注意力的“落地版”）
1. 线性投影→Encoder 压缩（Q/K），回写 off-chip；2) 取下一步需要的 Q/K，Decoder 解压进入片上；3) 依据 reorder 的顺序与固定 mask，把工作拆到两路：

- Denser：按 K-stationary 做 Q·Kᵀ（inter-PE 累加），再以 output-stationary 做 S·V（intra-PE 累加）；

- Sparser：用 CSC 索引只算非零；需要 Q 时先查询 Denser 的 Q 缓冲再决定外取；

1. 两路各自写入独立输出缓冲并行回写/合并。

#### key
- PE 累加模式切换：SDDMM 用 inter-PE（跨 PE 聚合列方向部分和），SpMM 用 intra-PE（每个 PE 内聚合输出），两阶段在同一 PE 线重配置，这是论文强调的“从 K-stationary（Q·K）切到 output-stationary（S·V）”。

- 为什么 K-stationary 能跑得好：K 被充分重用、中间缓冲更小、且天然匹配“按列产出 + 稀疏按列索引”的实现；其“Q 访问更频繁”的缺点由 AE 抵消。

- Sparser 的索引与转发：用 CSC 预存非零列索引（配合按列产出），并用 Query-based Q forwarding 在两路间共享 Q，减少 off-chip 访问。

> 硬件平台参数（论文实现）：面积约 **3 mm²**，DDR4-2400 带宽 **76.8 GB/s**，功耗 **323.9 mW@500 MHz**，片上 **320 KB SRAM**。
### Limitations / Weakness / Further research
---
1. **局限性分析**

(1) 方法层面（S&C + AE）
- **依赖固定 Mask**
	- SC（Split & Conquer）的 reorder + prune 依赖一个在 fine-tuning 阶段学到的固定稀疏模式。
		- 对于输入分布变化大或 domain shift 明显的任务，固定 Mask 可能导致性能退化。
		- 例如场景变化剧烈（不同类别/布局的图片）时，固定稀疏模式可能错过关键信息。
	
- **无法动态适配注意力模式**
	- 一旦 mask 固定，推理阶段不再根据输入图片动态生成稀疏模式，这在一些需要局部自适应注意力的任务（如物体检测、多目标跟踪）可能限制性能。
	
- **AE 压缩只针对 Q/K，不覆盖 V 和中间结果**
	- 带宽瓶颈可能在中间阶段转移到 V 或 S·V 阶段，而 AE 主要压缩了 Q/K。
	
- **Reorder 对多层 ViT 的全局一致性影响未深挖**
	- 不同层的注意力模式差异很大，文中似乎是对单层 mask 进行优化，但对跨层 mask 复用的影响分析不够深入。
	
---
(2) 硬件实现层面
- **双引擎（Dense/Sparse PE）资源利用率问题**
	- 在某些稀疏度分布下，sparse engine 的利用率可能下降，而 dense engine 可能闲置或饱和，这导致硬件资源不均衡。
		- 适合固定比例 dense/sparse 的任务，但若稀疏比例波动，性能可能不稳定。
	
- **CSC 稀疏格式存储开销**
	- CSC 对稀疏度高的情况非常好，但如果后续模型稀疏度下降，索引存储开销相对增加。
	
- **缺乏多任务并行调度机制**
	- 当前 pipeline 面向单路 attention，VLM/多模态任务往往有多路 cross-attention，需要调度多个 attention kernel 并行工作。
	
---
(3) 任务/应用层面
- **对下游任务泛化未验证**
	- 主要验证是分类任务（ImageNet），在检测、分割、视频理解等任务上的表现未深入研究。
		- 稀疏模式在需要保留空间结构信息的任务（比如 dense prediction）可能要重新设计。
	
- **对 Token 语义的敏感性不足**
	- Reorder 过程是基于 attention map 的排序，而不是直接考虑 token 的语义（比如物体边界、文本区域等）。
	
---
1. **未来可能 Follow 的方向**

方法改进
1. **动态可调稀疏模式**
	- 在推理时根据输入图片的低成本特征（如低分辨率 attention map）调整 sparse block 的 mask。
		- 可引入轻量的 Gumbel-Softmax / Top-K 筛选器实现动态更新。
	
1. **跨层稀疏模式协同优化**
	- 考虑多层 attention pattern 的相关性，在训练阶段优化一个跨层共享的压缩/稀疏策略，减少 mask 存储。
	
1. **多分辨率/分块重排**
	- 在 reorder 过程中融合多尺度 token 信息，让 global block 更好地覆盖多尺寸目标。
	
1. **全路径压缩**
	- AE 不仅压缩 Q/K，还压缩 V 及中间 S 矩阵（可以用低秩分解），进一步降低带宽。
	
---
硬件优化
1. **弹性双引擎调度**
	- 根据实际稀疏比例动态调整 dense/sparse engine 的分配比例，提升资源利用率。
	
1. **多路 Attention 并行化**
	- 针对 VLM 中的 cross-attention、image-text attention，设计多路并行的 sparse/dense 计算通道。
	
1. **新稀疏存储格式**
	- 针对固定 mask，可预编译成 PE-friendly 的压缩布局，减少索引访问延迟。
	
---
1. **在 VLM 时代的适配/改进点**

在 Vision-Language Models 中，ViTCoD 这类结构化稀疏 + 压缩技术仍然有用，但要解决以下问题：
1. **Cross-attention 的稀疏模式不同**
	- VLM 中 image-to-text attention 与 image self-attention 的稀疏分布差异很大。
		- 需要对不同类型的 attention 分别设计 mask，或者做多模态联合 mask 学习。
	
1. **Token 长度更长**
	- VLM 往往输入长文本 + 高分辨率图像，token 数量可达数千甚至上万，稀疏化带来的收益会更明显，但 mask 存储/调度更复杂。
	
1. **多模态 token 排序问题**
	- 现有 reorder 针对视觉 token，如果混合了文本 token，需要保持跨模态 token 对齐，否则 cross-attention 信息可能受损。
	
1. **对齐/语义保持**
	- VLM 强调图文对齐，reorder 如果破坏视觉 token 与文本 token 的语义对应，会降低模型性能，需要加入对齐约束。
	
---
## **ViT-slice: End-to-end Vision Transformer Accelerator with Bit-slice Algorithm**
---
## **FNM-Trans: Efficient FPGA-based Transformer Architecture with Full N:M Sparsity**
---
## **HG-PIPE: Vision Transformer Acceleration with Hybrid-Grained Pipeline**
---
## **FAS-Trans: Fully Exploiting FFN and Attention Sparsity for Transformer on FPGA**
---
## **FACT: FFN-Attention Co-optimized Transformer Architecture with Eager Correlation Prediction**
![image]({{ "images/TpsdbnDStoCzBSxZFhhcYIMFnsc.png" | relURL }})
---
### Background & Target
---
While the attention computation, focused by most previous works, only has decent power share when dealing with extremely long inputs. FACT, an efficient algorithm-hardware co-design optimizing all three modules of Transformer
---
### Algorithm
---
#### EP -- eager prediction
If there exists a few large probabilities in the 𝑆, the rest are very small and can be safely skipped since they have little impact on the output.

Generating the QK matrices causes much more computation and power than the _𝑄 _· _𝐾 𝑇 _, leading to suboptimal improvement

EP with cross-stage log-based inner-product estimation, which can reduce not only attention score computation but also the _𝑄𝐾𝑉  _linear projection.

#### Attention-distribution-aware QKV Generation
在 $$
Eager\ Generation
$$
的时候，对于预测注意力分数矩阵$$
\tilde{A}~
$$
,每一行会取 $$
top-k
$$
做filter，然后 $$
non-top-k
$$
呢就直接skip掉，那么由于这样形成的一个跟 $$
\tilde{A}
$$
维度相同的mask矩阵自然会形成一定的sparsity，比如 $$
\tilde{A}~
$$
的某一列全部都被skip掉的话，那么对应反推，在 $$
Q\ K\ V\ generation
$$
的时候，利用 $$
\tilde{A}~
$$
产生的sparsity mask就可以忽略掉一部分的精确计算，从而实现 $$
Q K V\ generation / FFN\ layers
$$
更好的加速。

$$
KV\ sparsity
$$
: It derives directly from the top-k result of the predicted _𝐴_ˆ matrix. If a column in the _𝐴_ˆ matrix has no values selected by the top-k, the key tensor related to this column is no longer required and can be safely pruned. Similarly, since the V matrix is multiplied by the attention matrix, the row with the same index in V matrix has no effect on the output, either, and can be safely removed.

$$
Q\ sparsity
$$
: Hence, when the _𝐴_ˆ matrix is obtained from EP, the difference between the 1 _𝑠𝑡 _and 2 _𝑛𝑑 _value of each row is compared to a threshold (we choose 3 based on experiments). If the former is larger, EP regards this row as being dominated by the largest token and directly uses a one-hot tensor as the _softmax _result where the largest token is assigned with 1.0 probability. In this way, the QK generation and attention computation related to this row can be fully skipped, and all that is needed is to copy the corresponding V tensor as the output.
#### Token-wise Mixed Precision FFN computation

---
### Hardware
---
#### EP unit with LOD circuit

#### KV-differential order (a scheduler to better match with the EP algorithm) 

#### Diagonal Storage Pattern for Mixed Precision FFN

---
### Limitations / Weakness / Further research 
---
The design concept of FACT and EP is via predicting the redundancy before computation, thus skipping unnecessary computation. FACT’s EP is an output dynamic sparsity method. Further, EP prediction is a unique cross-stage method for Transformer

感觉token importance的评估方法可以有所改进，尤其是FFN的这个加速方法是不是应该有更合理的判断方法？不知道是否是真的有效果

另外QKV的sparse部分感觉做的很不错，可以有参考，但是是按照token-wise进行的，直接舍弃掉完整token对应的Q/KV的generation，那么如果带入到多模态里面的话，直接进行这样成token的prune会不会掉点比较严重。
## **Breaking the Low-Rank Dilemma of Linear Attention**
---
## **FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers**
---
## **Qwen2.5-VL Technical Report**
---
### window attention
#### computation complexity
1. self-attention

$$
\Omega(MSA) = 4HW\times d_h\times d_i + 2(HW)^2 \times d_i\\
$$

1. window-self-attention

$$
 \Omega(\text{1-MSA}) = 4M^2\times d_h\times d_i + 2({M^2})^2 \times d_i\\
$$


$$
\Omega(\text{W-MSA}) = 4HW\times d_h\times d_i + 2M^2HW\times d_i= HW \times(4d_hd_i+2M^2d_i)\\
$$


window attention不等价于对MSA做tiling，WMSA是算法级别的注意力掩码（mask），它硬性屏蔽了跨窗口的 token 交互。即便你用相同的 M×M 大小切 patch，如果是全局 Attention，就还是能跨 patch 互相注意；Window Attention 则不行。

能否有纯硬件调度来支持attention的计算复杂度的降低？可能类似于window MSA这种么，Flashattention就是纯HBM➕算法调度的，但是flashattention基本上做烂了
## BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
---
## DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models
---
非常取巧的一篇文章，没有按照常规的比如判断importance的方法，而是换了一种prune的思路，采用最大化最小距离，换句话说就是用贪心算法取Visual token的特定元素数量的subset，使得最终该subset内的元素之间的距离最大化（最多样性化）
## ToDRE: Visual Token Pruning via Diversity and Task Awareness for Efficient Large Vision-Language Models
作者不是很distinguished
### Background & Target
同样是考虑diversity的一个prune思路，(cite了上面的DivPrune），额外还有token-task relevance的一个维度，根据两个维度进行prune
### Algorithm

## **FCoT-VL:Advancing Text-oriented Large Vision-Language Models with Efficient Visual Token Compression**
---





## **ATP-LLaVA: Adaptive Token Pruning for Large Vision Language Models**
---
## **Treat Visual Tokens as Text? But Your MLLM Only Needs Fewer Efforts to See**
---
## **Window Token Concatenation for Efficient Visual Large Language Models**
---
## **STAR: Stage-Wise Attention-Guided Token Reduction for Efficient Large Vision-Language Models Inference**
---
## **Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs**
---
## **Sanger: A Co-Design Framework for Enabling Sparse Attention using Reconfigurable Architecture**
---



