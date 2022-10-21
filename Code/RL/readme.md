Mac OSX and Windows version for simple game by openai

建议在Linux、MacOS上运行，不建议使用Windows，
试用云服务器信息可参考
https://help.aliyun.com/document_detail/312253.htm?spm=5176.8789780.J_5975631400.1.358945b5sEg3MW
和 https://ecs-buy.aliyun.com/ecs/?spm=a2c4g.11186623.0.0.22a263dbg0G3sx#/trial

# 注释X window相关代码

Mac OSX 和 Windows 都不支持X window，所以注释virtual display
```
#virtual_display = Display(visible=0, size=(1400, 900))
#virtual_display.start()
```

# Windows 运行程序需注意
参考 https://github.com/udacity/deep-reinforcement-learning/issues/54
需安装swig，box2d，box2d-py
```
pip install box2d
pip install box2d-py
```
# 忽略代码中的google.colab

