import os
import pandas as pd
from datetime import datetime
import shlex

#创建一个只有列的表格 列为Completion Trigger和is_success
df = pd.DataFrame(columns=['goal','is_success','completion' ,'trigger'])
now = datetime.now()
filename = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = f'../result/SINGLE_{filename}.csv'
df.to_csv(filename,index=False)

test = pd.read_csv('../data/harmful_behaviors.csv')
offset = 0

for i in range(offset, offset+10):
    os.system(f'python ../SingleModelSingleBehavior/single.py   --user_prompt {shlex.quote(test["goal"][i])} --target {shlex.quote(test["target"][i])} --save_result {filename}')