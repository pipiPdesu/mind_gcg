## 实验

### 单模型单behavior

如果只是试用的话，请直接运行demo.ipynb或者
```
python SingleModelSingleBehavior/single.py
```
如果想运行大规模实验，请
```
python scripts/exp_single.py
```

### 单模型多behavior

实验是
```
python SingleModelMultiPrompts/multi_prompt.py
```
在log文件中可以找到一个最终的后缀，评估这个后缀使用
```
python scripts/suffix_asr.py
```

### 多模型多behavior