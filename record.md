### 数据集设定
- 外部工具
    
    <details><summary>总共包含 7 种去除工具</summary>

    + 去雨 rain：
    + 去雾 haze：
    + 去噪 noise：
    + 去压缩 jpeg：
    + 去运动模糊 motion blur：
    + 去失焦模糊 defocus blur：
    + 超分辨率 super resolution：
    
    </details>

- Ps 内置工具
    <details><summary>总共包含 14 种内置工具（均为数值型工具）</summary>

    + 对比度 contrast：数值范围∈[-100,100]
    + 曝光度 exposure：数值范围∈[-5,5]（假定输入为：图片轻微欠曝，需要让模型感知到要轻微增加曝光度）
    + 阴影 shadow：数值范围∈[-100,100]
    + 黑色 black：数值范围∈[-100,100]
    + 白色 white：数值范围∈[-100,100]
    + 色温 temperature：数值范围∈[-100,100]
    + 色调 tint：数值范围∈[-100,100]
    + 饱和度 saturation：数值范围∈[-100,100]
    + 自然饱和度 vibrance：数值范围∈[-100,100]
    + 纹理 texture：数值范围∈[-100,100]
    + 清晰度 clarity：数值范围∈[-100,100]
    + 去除薄雾 dehaze：数值范围∈[-100,100]
    + 晕影 vignette：数值范围∈[-100,100]
    + 颗粒 grain：数值范围∈[-100,100]
    
    </details>


### 1. 指令集合成
#### 1.1. 合成方式
- 人工合成
- LLM 合成
- 代码合成
#### 1.2. 合成规则
##### 1.2.1. 单步指令（仅包含一个工具）
- 外部工具
    - 随机选取一种退化类型并获取全部外部工具，根据该退化类型生成一个对应的指令（e.g. 去雾工具 RIDCP 就生成去雾的指令）

- Ps 内置工具
    - 考虑采用 `Fuzzy Logic` 的思想（Mamdani 模糊推理系统（Mamdani Fuzzy Inference System, MFIS））
        1. 首先限定一个安全数值范围（如最值范围的 50%，即[-50,50]）
        2. 然后将数值范围划分为 n 个语言词 并赋予一个归一化在[-1,1]之间的`隶属函数`（Membership Function, MF）\
            **Tri(a,b,c)** 表示三角形顶点 b，底边在 a 和 c
            - NL（Negative Large） 
            - NS（Negative Small） 
            - Z（Zero / Neutral）  
            - PS（Positive Small）
            - PL（Positive Large）\
            让 LLM 自己去总结`模糊规则`（Fuzzy Rules, FR）\
            - IF NL THEN ++
            - IF NS THEN +
            - IF Z THEN 0
            - IF PS THEN -
            - IF PL THEN -- \
            设定好`解模糊方法`（Defuzzification Method, DM）\
            - 质心法（Centroid Method）
            - 最大隶属度法（Maximum Membership Method）
        3. 进行多轮，看 LLM 是否已经准确总结到了`模糊规则`
    - 目标：让 LLM 学会根据这种文字的`模糊逻辑`学习到`模糊规则`
##### 1.2.2. 多步指令（包含多个工具）
- 外部工具
    - 随机选取多种退化类型并获取全部外部工具以及已经知道的工具优先级，根据该退化类型生成一个对应的指令（e.g. 去雾工具 RIDCP + jpeg 就生成去雾 + jpeg 的指令）
- Ps 内置工具
    - 依旧采用 `Fuzzy Logic` 的思想，但多了一个排列组合的问题，需要让 LLM 学会总结出更加复杂的`模糊逻辑`（Fuzzy Logic）（e.g. 此时一个输入的属性可能对应多个`隶属函数`且可能包含正向和负向的混合）
    - 目标：让 LLM 学会根据这种文字的`模糊逻辑`学习到更加复杂的`模糊规则`

### 2. 向量数据库的构建
- 利用 [AgentNet](https://github.com/zoe-yyx/AgentNet/tree/main) 中使用到的经验池 RAG 模块的基础上搭建（使用到的 embedding model 是 `BAAI/bge-large-en-v1.5`）
- 利用 [LangChain](https://python.langchain.com/docs/introduction/) 中使用到的向量数据库模块的基础上搭建

### 3. stage1：基础知识学习阶段
- 外部工具
    1. 随机从`extra_tools_instruction.json`文件中选取一个单步指令
    2. 跳过 task evolution 阶段
    3. 进入 solution evolution 阶段，将 `instruction` 和 `input` 作为提示词的一部分，提供对应 `tool` 的经验池信息（总共运行的次数 + 模糊规则作为的经验条例 + 当前退化类型中工具的排序信息）以及 `tool` 的基本信息，要求模型提供 3 个 `solution` 且这 3 个 `solution` 有一个排序从高到低的置信度，置信度最高的记为 `solution_1`，其次为 `solution_2`，最后为 `solution_3`
    4. 进入 feedback 阶段，计算模糊规则下 3 个 `solution` 实际在各种IQA指标上的评估，作为 `reward` 反馈给下一阶段
    5. 进入 experience refinement 阶段，将整个执行过程以及 `reward` 进行一个学习总结，横向对比预测的 ranking 和实际的 ranking 差异性，总结出可以存放在数据库中的经验条例，执行的详细信息可以作为一个统计学样本
    6. 更新向量数据库
    7. 重复 1-6，直到模型可以比较准确地掌握某一个工具的使用
- Ps 内置工具
    1. 随机从`ps_tools_instruction.json`文件中选取一个单步指令
    2. 跳过 task evolution 阶段
    3. 进入 solution evolution 阶段，将 `instruction` 和 `input` 作为提示词的一部分，提供对应 `tool` 的经验池信息（总共运行的次数 + 模糊规则作为的经验条例）以及 `tool` 的基本信息，要求模型提供 3 个 `solution` 且当且仅当对某一个方案特别有信心的时候才可以两个以上完全相同的 `solution`，否则必须提供 3 个不同的 `solution`
    4. 进入 feedback 阶段，计算模糊规则下 3 个 `solution` 输出的数值与真实数值的差距，作为 `reward` 反馈给下一阶段
    5. 进入 experience refinement 阶段，将整个执行过程以及 `reward` 进行一个学习总结，横向对比不同方案的差异性，总结出可以存放在数据库中的经验条例，执行的详细信息可以作为一个统计学样本
    6. 更新向量数据库
    7. 重复 1-6，直到模型可以比较准确地掌握某一个工具的使用

### 4. stage2：组合知识学习阶段
1. 在线构成多步指令，提供 `EXP pool1（Ps 工具池）和 EXP pool2（外部工具池）中的经验条例`作为提示词的一部分，让模型结合经验条例和自身对图像复原及图像增强的认识，充分思考现实当中可能存在的多种退化类型的组合，生成一个多步指令
2. 进入 task evolution 阶段，将 `instruction` 和 `input` 并提供 `EXP pool1 和 EXP pool2 中的经验条例`作为提示词的一部分，告诉模型工具库当中有`哪些类别`的工具可以使用（不涉及具体名称），让模型判断需要使用到哪些类别的工具进行处理（what to use）
3. 进入 solution evolution 阶段，将 `instruction` 和 `input` 并提供 `EXP pool3 中的经验条例`作为提示词的一部分，以及工具的具体名称，要求模型按照正确的去除退化执行顺序提供三个 `solution`，且这三个 `solution` 必须是不同的（how to use）
4. 进入 feedback 阶段，计算模糊规则下 3 个 `solution` 实际在各种IQA指标上的评估，作为 `reward` 反馈给下一阶段
5. 进入 debate 阶段，提供 3. 4. 部分的输出，让模型从方案设定到最终方案效果做一个辩论，辨别为什么某一个比较好，为什么某一个比较差，选定一个最优方案（refinement）
6. 进入 experience refinement 阶段，将整个执行过程以及 `reward` 进行一个学习总结，总结出多步指令中正确的单步指令执行顺序，执行的详细信息可以作为一个统计学样本（conclusion）
7. 更新向量数据库
8. 重复 1-7，直到模型可以比较准确地掌握多步指令的使用

### 5. stage3：动态知识学习阶段
1. 在线构成多步指令，提供已经存放过在 EXP1 pools，EXP pool2 和 EXP pool3 中的退化类型作为提示词的一部分，让模型结合经验条例和自身对图像复原及图像增强的认识，思考这个新的退化类型可能和之前存放过的退化类型的异同，生成一个在原本经验池当中没有出现过的多步指令，并指示模型可以考虑替换某一个退化类型（e.g. 之前没有见过去雨 + jpeg + 去雾 的组合，但是见过去雨 + jpeg 和 jpeg + 去雾 的组合，那么模型可以考虑生成去雨 + jpeg + 去雾 的组合）
2. 进入 task evolution 阶段，将 `instruction` 和 `input` 并提供 `EXP pool1 和 EXP pool2 中的经验条例`作为提示词的一部分，告诉模型工具库当中有`哪些类别`的工具可以使用（不涉及具体名称），让模型判断需要使用到哪些类别的工具进行处理（what to use）
3. 进入 solution evolution 阶段，将 `instruction` 和 `input` 并提供 `EXP pool3 中的经验条例`作为提示词的一部分，以及工具的具体名称，要求模型按照正确的去除退化执行顺序提供三个 `solution` 其中有一个必须是这个新的工具名称，且这三个 `solution` 必须是不同的（how to use）
4. 进入 feedback 阶段，计算模糊规则下 3 个 `solution` 实际在各种IQA指标上的评估，作为 `reward` 反馈给下一阶段
5. 进入 debate 阶段，提供 3. 4. 部分的输出，让模型从方案设定到最终方案效果做一个辩论，辨别为什么某一个比较好，为什么某一个比较差，选定一个最优方案（refinement）
6. 进入 experience refinement 阶段，将整个执行过程以及 `reward` 进行一个学习总结，总结出多步指令中正确的单步指令执行顺序，执行的详细信息可以作为一个统计学样本（conclusion）
7. 更新向量数据库
8. 重复 1-7，直到模型可以比较准确地掌握多步指令的使用


### Questions & Answers
1. 是否使用 VLM 来辅助理解图像内容而不仅仅是 LLM 纯文字思路
2. 经验池中量化的经验条例如何设计，如果分隔开感知端，后续使用中如何确保检索到需要的经验条例
3. 是否需要搭建完整的 RAG 框架，包括文本分割、向量化、检索、生成等 or 只使用 json + embedding model 进行向量化和检索