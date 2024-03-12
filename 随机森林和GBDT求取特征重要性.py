# 导入所需的库
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为黑体
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 指定Excel文件的路径
file_path = '转换后的数据.xlsx'

# 使用NumPy数组表示数据集
data = pd.read_excel(file_path)

# 划分特征和目标变量
X = data.iloc[:, 1:]  # 特征
y = data.iloc[:, 0]   # 目标变量

# 将目标变量转换为离散类型
kmeans = KMeans(n_clusters=10, random_state=0).fit(y.values.reshape(-1, 1))
y_discrete = kmeans.predict(y.values.reshape(-1, 1))

# 使用随机森林计算特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X, y_discrete)
importances_rf = rf.feature_importances_

# 使用梯度提升树计算特征重要性
gb = GradientBoostingClassifier(n_estimators=100, random_state=0)
gb.fit(X, y_discrete)
importances_gb = gb.feature_importances_

# 将两种特征重要性求平均值
importances_avg = (importances_rf + importances_gb) / 2

# 将两种特征重要性求平均值
importances_avg = (importances_rf + importances_gb) / 2

# 对特征重要性进行排序
indices = np.argsort(importances_avg)[::-1]

# 取前十个重要性最高的特征
top_features = X.columns.values[indices][:10]

# 获取前十个特征的特征重要性值
top_importances = importances_avg[indices][:10]

# 生成特征重要性数据集
importance_data = pd.DataFrame(top_importances.reshape(-1, 1), index=top_features, columns=["Importance"])

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(importance_data.iloc[:10], cmap="YlOrRd", annot=True, fmt='.2f', linewidths=0.5, annot_kws={"fontsize":14})
plt.title("前10个特征的重要性", fontsize=16)
plt.xlabel("重要性", fontsize=14)
plt.ylabel("特证名", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# 将特征重要性进行排序，并输出前20个最重要的特征
indices_rf = importances_rf.argsort()[::-1]
indices_gb = importances_gb.argsort()[::-1]

# 输出随机森林模型的特征重要性排序
print("随机森林模型特征重要性排序：")
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices_rf[f]], importances_rf[indices_rf[f]]))
    if f == 19:
        break
print('\n')

# 输出梯度提升树模型的特征重要性排序
print("梯度提升树模型特征重要性排序：")
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices_gb[f]], importances_gb[indices_gb[f]]))
    if f == 19:
        break

# 将特征重要性可视化，并添加图例
colors = list(mcolors.TABLEAU_COLORS.values())

plt.figure(figsize=(8, 6))
bars = plt.bar(range(X.shape[1]), importances_rf[indices_rf], color=colors)
plt.xticks(range(X.shape[1]), X.columns[indices_rf], rotation=90)
plt.title('随机森林模型特征重要性')

# 设置图例
legend_labels = [X.columns[indices_rf[i]] for i in range(len(indices_rf))]
plt.legend(bars, legend_labels, loc='best', prop={'size': 12})

plt.show()

plt.figure(figsize=(8, 6))
bars = plt.bar(range(X.shape[1]), importances_gb[indices_gb], color=colors)
plt.xticks(range(X.shape[1]), X.columns[indices_gb], rotation=90)
plt.title('梯度提升树模型特征重要性')

# 设置图例
legend_labels = [X.columns[indices_gb[i]] for i in range(len(indices_gb))]
plt.legend(bars, legend_labels, loc='best', prop={'size': 12})

plt.show()