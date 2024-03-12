## 预测集的目标变量列要提前删除，等生成处理好的excel时，再加上目标编码列。
import os
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 指定 Excel 文件的路径
file_path = "D:\电脑桌面\华数B题\第一问的求解\第一问的数据.xlsx"
# 读取数据（假设数据在第一个工作表中）
data = pd.read_excel(file_path, sheet_name=0)

# 统计每个特征的缺失值数量和占比
missing_count = data.isnull().sum()
missing_percent = missing_count / len(data) * 100

# 输出每个特征的缺失值情况
for feature in data.columns:
    count = missing_count[feature]
    percent = missing_percent[feature]
    print(f'特征"{feature}"的缺失值数量为{count}，占比{percent:.2f}%。')

# 使用 missingno 库的 bar() 函数生成缺失值条形图
fig, ax = plt.subplots(figsize=(10, 5)) # 创建图形窗口并绘制缺失值条形图
msno.bar(data, ax=ax)
plt.xticks(fontsize=12) # 设置 x 轴刻度标签的字体大小为 12
plt.yticks(fontsize=12) # 设置 y 轴刻度标签的字体大小为 12
plt.subplots_adjust(bottom=0.4) # 调整 Y 轴的最小值，使图形窗口的底部向上平移
plt.show()

# 使用 KNN 方法对数据进行缺失值填充（假设 K=5）
knn_imputer = KNNImputer(n_neighbors=5)
data_imputed = knn_imputer.fit_transform(data)

# 将缺失值填充后的数据保存到 Excel 文件中
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
file_path_out = os.path.join('D:\电脑桌面', 'KNN缺失值处理.xlsx')
data_imputed.to_excel(file_path_out, index=False)

print(f'已将处理后的数据保存至文件"{file_path_out}"中。')
