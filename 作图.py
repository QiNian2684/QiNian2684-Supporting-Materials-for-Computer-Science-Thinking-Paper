import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from Excel file
file_path = 'D:\OneDrive\桌面\第二问求解\优劣解距离算法计算结果.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Scatter Plot for D+ vs D-
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='正理想解距离(D+)', y='负理想距离(D-)', hue='索引', palette='viridis')
plt.title('正理想解距离(D+) 与 负理想距离(D-) 关系图')
plt.xlabel('正理想解距离(D+)')
plt.ylabel('负理想距离(D-)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()

# Bar Plot for Comprehensive Score Index
plt.figure(figsize=(12, 10))
sns.barplot(data=data, x='综合得分指数', y='索引', palette='viridis')
plt.title('各地区综合得分指数')
plt.xlabel('综合得分指数')
plt.ylabel('地区')
plt.tight_layout()

# Line Plot for Ranking
plt.figure(figsize=(12, 8))
sns.lineplot(data=data, x='索引', y='排序', marker='o')
plt.title('各地区排序')
plt.xlabel('地区')
plt.ylabel('排序')
plt.xticks(rotation=45)
plt.tight_layout()

# Show all plots
plt.show()
