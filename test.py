

import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('data/translation_en2zh/train-00000-of-00001.parquet')

# 查看数据
print(df.head())
pass