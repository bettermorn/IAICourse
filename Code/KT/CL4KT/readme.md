# Contrastive Learning for Knowledge Tracing

会议及时间：Proceedings of the ACM Web Conference 2022

论文地址：https://dl.acm.org/doi/abs/10.1145/3485447.3512105
包括：对比学习，数据增强，知识跟踪

# 所用数据集
KDD Cup 2010 EDM Challenge  https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp

# 创新点
解决了由于学生与问题之间的稀疏交互，隐藏的表示很容易过拟合，往往不能准确地捕捉学生的知识状态的问题。第一次介绍了一个用于知识追踪的对比学习框架，它揭示了语义上相似或不同的学习历史的例子，并刺激人们学习他们之间的关系。

# 论文摘要

论文提出了一个针对KT的对比学习（CL）框架，名为CL4KT。CL4KT的主要思想是通过将相似的学习历史拉在一起，并在表征空间中将不同的学习历史分开来学习有效的表征。为了对学生的学习历史进
行编码，我们使用了多个transfomer编码器即问题编码器和交互编码器来进行学习历史答题记录，以及一个知识检索器来预测对以下问题的回答。在预测未来的响应时，我们使用单向编码器来防止未来
的信息泄漏。另一方面，在学习对比表征时，我们利用双向自注意编码器从两个方向总结学习历史的整个上下文。此外，本文还设计了特定于领域的数据增强方法，以反映每个学习历史的语义。由于KT使
用由两个相互依赖的标记（问题和回答）组成的学习历史，所以本文使用四种数据增强方法来揭示语义上相似和不同的学习历史，从而用对比损失刺激学习它们之间的关系。

# 代码链接
https://github.com/UpstageAI/cl4kt
数据集   KDD Cup 2010 EDM Challenge. https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp

# 代码修改
train.py Line 190 改为
```
    logs_df = logs_df._append(
```


