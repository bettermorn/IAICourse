ActiveLearning Human in the loop
# bicycle_detection.py 
source: https://github.com/rmunro/bicycle_detection/blob/master/bicycle_detection.py, 
updated for runnable purpose.
该代码执行以下主要任务：
* 数据加载和预处理：从Open Images数据集中加载有标签和无标签的图像数据，处理注释，并使用SQLite管理特征存储。
* 特征提取：使用预先训练好的模型（ResNeXt50和Faster R-CNN）从图像中提取特征。
* 模型训练：使用提取的特征和注释数据训练一个简单的线性分类器（SimpleClassifier）。
* 主动学习：根据不确定性和异常检测，实施选择图像进行注释的策略。
* 评估：使用验证和评估数据集评估分类器的性能。
* 用户界面：使用eel库创建基于网络的图像注释界面。


