# 来源
https://github.com/rmunro/bicycle_detection Human-in-the-Loop Machine Learning book
打开一个 HTML 窗口，让你注释给定图片中是否有一辆自行车。是否包含自行车。我们的目标是标注足够多的数据，以训练出一个比目前最先进的 的系统（F-score 约为 0.85）。当注释的图像足够多，可以开始构建模型时，窗口中将显示当前的精确度，系统也将开始使用不确定性采样和基于模型的异常值采样，以获取最有可能帮助改进模型的图像。系统还将开始使用不确定性采样和基于模型的异常值来采样最有可能帮助提高模型整体准确性的图像。

# 架构
参考 Eel https://github.com/bettermorn/IAICourse/wiki/Eel%E7%9A%84%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95


# main function
用于实现一个主动学习框架，以检测图像中的自行车。它利用PyTorch torchvision库中预先训练的深度学习模型从图像中提取特征，并训练一个简单的分类器来区分包含自行车和不包含自行车的图像。

* 数据加载和预处理：从Open Images数据集中加载有标签和无标签的图像数据，处理注释，并使用SQLite管理特征存储。
* 特征提取：使用预先训练好的模型（ResNeXt50和Faster R-CNN）从图像中提取特征。
* 模型训练：使用提取的特征和注释数据训练一个简单的线性分类器（SimpleClassifier）。
* 主动学习：根据不确定性和异常检测，实施选择图像进行标注的策略。
* 评估：使用验证和评估数据集评估分类器的性能。
* 用户界面：使用eel库创建基于网络的图像注释界面。


# JavaScript

异步函数，定时检查函数，侦听函数

|Name|Function|Input|Output|Comment|
|:--|--|---|---|--:|
|function data_loading_completed()|数据加载成功后给出信息|无|更新页面|需要eel.expose|
|document.addEventListener('DOMContentLoaded', async function ()|页面加载完成后通知后端开始加载数据|调用后台异步函数start_data_loading()|无|更新加载状态|使用Promise处理异步操作|
|function add_annotation(is_bicycle)|完成标注过程|is bicycle|调用后台异步函数add_annotation|Promise|
|function focus_first_image()|Focus first Image|无|聚焦|Promise|
|function remove_first_image()|Remove first Image|无|移除标注好的图片并聚焦新的图片|Promise|
|document.addEventListener("keypress", function (event)|根据key处理标注过程|key|无|b，n，z|
|function training_loaded() |加载训练数据|无|调用后台异步函数training_loaded()，返回成功结果或者false|Promise|
|function validation_loaded()|加载验证数据|无|调用后台异步函数validation_loaded()，返回成功结果或者false|Promise|
|setInterval(function ()|定时检查用于标注的新图片|无|聚焦于第一张图片|处理竞争条件|
|setInterval(function ()|定时获取当前的准确率|调用异步函数get_current_accuracies|更新界面的准确率|使用Promise|
|setInterval(function ()|估算处理时间|调用异步函数estimate_processing_time|显示处理时间|使用Promise以及显示错误信息|
|function showError(message)|显示错误信息|message|页面显示错误信息|无|

# Python file
* Note：处理race condition 异步处理，需要用到 lock，或者也可以使用条件变量。
* 加载数据需要一段时间，尤其是第一次，因此我们并行处理 # eel.spawn(load_data)。单独线程逐步下载并提取COCO和ImageNet表示eel.spawn(add_pending_annotations)， 不断重新训练模型，并对未标记的项目进行预测 eel.spawn(continually_retrain)

## 1 load_data
1.1 loading val  call load_annotations
1.2 loading existing annotations
1.3 loading eval
1.4 loading train
数据加载完成以后，通知前端。
1.5 load most recent model

## 2 add pending annotation


## 3 train model

# 主要方法
特征向量创建
## def make_feature_vector(image_id, url, label=“”):
    # 创建或检索图像特征向量的函数
用途：将两个模型提取的特征相结合，为图像创建一个全面的特征向量。
功能：
* 检查URL是否缺失或图像是否损坏。
* 尝试从特征存储（SQLite数据库）中检索特征。
* 如果没有，则下载图像并使用两个模型提取特征。
* 将特征存储在数据库中以备将来使用。

## def load_annotations(annotation_filepath, image_filepath, load_all=False):
从Open Images数据集中加载注释和图像URL
用途：解析包含图像ID、URL和标签的CSV文件。使用封存和压缩技术缓存数据结构，以便更快地访问。

def train_model(batch_size=20, num_epochs=40, num_labels=2, num_inputs=2058, model=None):
    # 使用标注数据训练SimpleClassifier模型
目的：为分类器实现训练循环。
功能：
* 确保有足够的标记样本开始训练。
* 平衡正负样本数量相等的数据集。
* 训练模型，并评估验证数据上的性能。
* 如果性能提高，则保存模型。

## def get_random_prediction(model=None):
功能：获取对未标记数据的预测，以选择样本进行注释
目的：预测未标记图像的标签，以找出不确定性或异常值较高的图像。




