# GraphKT


## 程序运行环境
```
torch 2.2.2
cuda 12.1
```

## 运行程序需要安装依赖包
The libraries required for this program are torch-geometric, torch and its various accompanying libraries
`pip install  --no-index torch-sparse -f https://data.pyg.org/whl/torch-2.2.2+cu121.html`
详情参考 https://github.com/pyg-team/pytorch_geometric/issues/8523

## 程序修改
以下代码已修改
    # exclude the  groupings
    seqs = data.groupby(['user_id']).apply(lambda x: x[features].values.tolist(),include_groups=False)

## 程序运行
ipython GraphKT.ipynb

    
