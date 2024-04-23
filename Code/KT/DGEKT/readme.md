# DGEKT A DUAL GRAPH ENSEMBLE LEARNING METHOD FOR KNOWLEDGE TRACING

期刊及时间：ACM Transactions on Information Systems, 2024

论文地址：https://dl.acm.org/doi/abs/10.1145/3638350
包括：集成学习，知识跟踪，知识蒸馏

# 所用数据集
ASSIST09, ASSIST17, and EdNet.

# 创新点

1. 提出了一种新的学生学习交互的双图结构，用于知识追踪，在此基础上，DGEKT分别通过超图建模和有向图建模来捕获异构的习题与概念关联和交互转换。
2. 利用在线知识蒸馏，自适应地结合对偶图模型，形成一个集成教师模型，该模型进而提供了对所有练习的预测，作为更高的建模能力的额外监督。

# 简述

论文提出了一种新的知识跟踪对偶双图集成学习方法（DGEKT），该方法分别通过超图建模和有向图建模，建立了学生学习交互的双图结构来捕获异构习题与概念间的关联。接着引入了在线知识蒸馏的技术，由于尽管知识追踪模型将预测学生的反应练习有关不同的概念，它是优化仅仅对预测精度在一个练习在每一步。通过在线知识精馏，将对偶图模型自适应组合，形成更强的教师模型，从而提供对所有练习的预测，作为额外的监督，以获得更好的建模能力。    


# 代码链接
https://github.com/yumo216/dgekt
可使用自己的数据集 https://github.com/yumo216/dgekt#try-using-your-own-dataset

# 代码修改
建议可以如下调整代码的目录结构，以便更明晰
```
run.py
log
model
KnowledgeTracing
Dataset
```

