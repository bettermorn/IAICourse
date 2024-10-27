# Source
https://arxiv.org/abs/2009.14193  Uncertainty Sets for Image Classifiers using Conformal Prediction by
Anastasios Angelopoulos, Stephen Bates, Jitendra Malik, Michael I. Jordan.
代码仓库:https://github.com/aangelopoulos/conformal_classification

# 环境信息
* python 3.11
* torch              2.5.0
* torchaudio     2.5.0
* torchvision     0.20.0
* CUDA Version: 12.3

# 代码修改
## example.py
'''
import torchvision.models as models
from torchvision.models import ResNet152_Weights


model = torchvision.models.resnet152(pretrained=True,progress=True).cuda()
改为
model = models.resnet152(weights=ResNet152_Weights.DEFAULT,progress=True).cuda()
'''

## figure2.py
'''
df.append
改为
df._append
'''
实验1：IMAGENET上的覆盖率VS集合大小.在这个实验中，我们计算了两种不同α值下每个过程的覆盖率和平均集合大小。在100多次试验中，我们随机抽取了Imagenet-Val的两个子集：一个是20K大小的保角校准子集，一个是20K大小的评估子集。表1列出了覆盖率和集合大小的中位数。图2显示了朴素、APS和RAPS的性能；RAPS的集合比朴素和APS的集合小得多，但也能达到覆盖率。我们还报告了保形固定k程序的结果，该程序可以找到在保留集上实现覆盖的最小固定集大小k∗，然后在新样本上预测大小为k∗-1或k∗的集，以实现精确覆盖；参见附录E中的算法4。
