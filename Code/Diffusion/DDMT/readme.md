# DDMT
DDMT: Denoising Diffusion Mask Transformer Models for Multivariate Time Series Anomaly Detection
Chaocheng Yang, Tingyin Wang, Xuanhui Yan
https://arxiv.org/abs/2310.08800

## 论文摘要

摘要：多变量时间序列中的异常检测已成为时间序列研究中的一个重要挑战，在欺诈检测、故障诊断和系统状态估计等多个领域具有重要的研究意义。近年来，基于重构的模型在检测时间序列数据异常方面
显示出了巨大的潜力。然而，由于数据规模和维度的快速增长，时间序列重构过程中的噪声和弱身份映射（WIM）问题日益突出。为解决这一问题，我们引入了一种新的自适应动态邻域掩码（ADNM）机制，
并将其与变换器和去噪扩散模型相结合，创建了一种新的多变量时间序列异常检测框架，命名为动态掩码和去噪扩散变换器（DMDDT）。引入 ADNM 模块是为了减少数据重构过程中输入和输出特征之间的信
息泄漏，从而减轻重构过程中的 WIM 问题。去噪扩散变换器（DDT）采用变换器作为去噪扩散模型的内部神经网络结构。它通过学习时间序列数据的逐步生成过程来模拟数据的概率分布，捕捉正常的数据
模式，并通过去除噪声逐步恢复时间序列数据，从而清晰地恢复异常数据。据我们所知，这是第一个将去噪扩散模型和变换器结合起来用于多元时间序列异常检测的模型。我们在五个公开的多元时间序列异
常检测数据集上进行了实验评估。结果表明，该模型能有效识别时间序列数据中的异常，在异常检测方面达到了最先进的性能。

通过DeepL.com（免费版）翻译

## 关键词 

多变量时间序列 异常检测 动态掩码 变换器 去噪扩散模型

# 数据集
参考
https://github.com/yangchaocheng/DMDDT#datasets


# 代码链接
https://github.com/yangchaocheng/DMDDT

# 代码修改
准备

main.py line 10 根据实际机型修改
```
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
```
