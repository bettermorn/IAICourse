# SAINTplus
在
https://paperswithcode.com/task/knowledge-tracing，对EdNet数据集， SAINT+是best model



## 代码来源
https://github.com/shivanandmn/SAINT_plus-Knowledge-Tracing-
这里有SAINTplus模型的介绍

## 运行环境
torch 2.2.2
cuda 12.1

## 数据集
EdNet

## 程序修改
### 1 lightning 相关
需要根据lightning 2.x的语法修改原项目仓库的代码
可参考 https://lightning.ai/docs/pytorch/stable/common/lightning_module.html 选择与自己环境匹配的版本说明。
### 2 config.py
`torch.set_float32_matmul_precision('high')`
### 3 train.py
`trainer = pl.Trainer(strategy = DDPStrategy(find_unused_parameters=True),max_epochs=5)`

## 程序运行
`python train.py`
