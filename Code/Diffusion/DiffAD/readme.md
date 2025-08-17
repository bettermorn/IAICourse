# DiffAD
基于插值的时序异常检测与条件权重增量扩散模型，KDD 2023

* 论文地址：https://dl.acm.org/doi/10.1145/3580305.3599391
* 关键词：时间序列，扩散模型，状态空间模型，数据插补
* 相关网站


## 创新点

## 论文摘要
现有的时间序列异常检测模型主要基于正常数据点占主导地位的数据集进行训练，当异常数据点在特定时间段内密集出现时，这些模型会失去有效性。为解决这一问题，我们提出了一种新的方法，称为DiffAD，从时间序列插值的角度出发。与之前的基于预测和重建的方法不同，这些方法采用部分或全部数据作为估计的观测值，DiffAD采用基于密度比的策略灵活选择正常观测值，能够轻松适应异常集中场景。为缓解异常集中场景下的模型偏置问题，我们设计了一种基于去噪扩散的插值方法，通过条件权重递增扩散提升缺失值插值性能，既能保留观测值信息，又能显著提升数据生成质量，从而实现稳定的异常检测。此外，我们定制了多尺度状态空间模型，以捕捉不同异常模式下跨时段的长期依赖关系。在真实世界数据集上的广泛实验结果表明，DiffAD的性能优于现有最先进基准方法。

# 所用数据集
SMD
https://www.kaggle.com/datasets/mgusat/smd-onmiad


# 代码

## 参考代码
https://github.com/ChunjingXiao/DiffAD

## 代码修改
prepare_time_data.py
```
df = df.append(pd.Series(), ignore_index=True)
```
to
```
df = df._append(pd.Series(), ignore_index=True)
```

## 代码演示
[【实验演示】DiffAD基于插值的时序异常检测与条件权重增量扩散模型-测试](https://www.bilibili.com/video/BV1adYezUE5X)
[【实验演示】DiffAD基于插值的时序异常检测与条件权重增量扩散模型-训练](https://www.bilibili.com/video/BV1o4Yvz4EHg)


# 相关参考资料

