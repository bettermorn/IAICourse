Mac OSX and Windows version for simple game by openai

建议在Linux、MacOS上运行，不建议使用Windows

# Mac OSX
Mac OSX不支持X window，所以注释virtual display
#virtual_display = Display(visible=0, size=(1400, 900))
#virtual_display.start()

# Windows
参考 https://github.com/udacity/deep-reinforcement-learning/issues/54
需安装swig，box2d，box2d-py
```
pip install box2d
pip install box2d-py
```

