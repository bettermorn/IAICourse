# DDPM

* 论文地址：[https://arxiv.org/abs/2006.11239](https://arxiv.org/pdf/2006.11239v2) 
* 关键词：生成式模型 自回归解码的一般化 渐进式有损解压缩方案 扩散模型
* 相关网站 https://hojonathanho.github.io/diffusion/


## 创新点

## 论文摘要
扩散概率模型是一类潜变量模型，其灵感来自非平衡热力学。我们的最佳结果是在根据扩散概率模型和去噪分数匹配与朗格文动力学之间的新联系设计的加权变分约束上进行训练而获得的，
我们的模型自然采用了渐进式有损解压缩方案，该方案可解释为自回归解码的一般化。在无条件的 CIFAR10 数据集上，我们获得了 9.46 分的入门分数和 3.17 分的先进 FID 分数。
在 256x256 LSUN 上，我们获得了与 ProgressiveGAN 类似的样本质量。

# 所用数据集

hugging faces

https://github.com/lucidrains/denoising-diffusion-pytorch

# 代码链接
https://www.kaggle.com/code/b07202024/hw6-diffusion-model

## 参考代码
https://github.com/lucidrains/denoising-diffusion-pytorch

## 代码解释
https://www.bilibili.com/video/BV1TD4y137mP


# 相关参考资料
https://nn.labml.ai/diffusion/ddpm/index.html


# 代码修改
Inference 部分改为
```
ckpt = str(trainer.results_folder / f'model-{trainer.step // trainer.save_and_sample_every}.pt')
```
