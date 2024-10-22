# main function
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
|document.addEventListener('DOMContentLoaded', async function ()||调用后台异步函数start_data_loading()|无|更新加载状态|使用Promise处理异步操作|
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

## 1 load_data

## 2 add pending annotation

## 3 train model
