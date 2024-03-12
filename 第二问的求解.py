# 更新假设的数据
平均日照时长_新疆 = 8  # 每天小时
平均太阳辐射强度_新疆 = 6  # kWh/㎡/日
光伏板成本_新疆 = 0.4  # 每瓦美元
安装成本_新疆 = 0.3  # 每瓦美元
其他成本_新疆 = 500000  # 美元
年运维成本百分比_新疆 = 0.015  # 总投资的百分比
光伏板效率_新疆 = 0.20  # 20%
电价_新疆 = 0.08  # 每kWh美元
项目寿命_新疆 = 30  # 年
总功率_新疆 = 10_000_000  # 10兆瓦（MW），以瓦为单位
折现率 = 0.05  # 5%

# 重新计算总投资成本
总投资成本_新疆 = 总功率_新疆 * (光伏板成本_新疆 + 安装成本_新疆) + 其他成本_新疆

# 重新计算年度电力产出
年度电力产出_新疆 = 总功率_新疆 * 平均日照时长_新疆 * 平均太阳辐射强度_新疆 * 光伏板效率_新疆 * 365  # kWh

# 重新计算年度收益
年度收益_新疆 = 年度电力产出_新疆 * 电价_新疆  # 美元

# 重新计算年运维成本
年运维成本_新疆 = 总投资成本_新疆 * 年运维成本百分比_新疆  # 美元

# 重新计算回报期
回报期_新疆 = 总投资成本_新疆 / 年度收益_新疆  # 年

# 重新计算净现值（NPV）
NPV_新疆 = sum([(年度收益_新疆 - 年运维成本_新疆) / ((1 + 折现率)**t) for t in range(1, 项目寿命_新疆 + 1)]) - 总投资成本_新疆

print(总投资成本_新疆, 年度电力产出_新疆, 年度收益_新疆, 年运维成本_新疆, 回报期_新疆, NPV_新疆)

import matplotlib.pyplot as plt
import numpy as np

# 定义参数
年份 = np.arange(1, 项目寿命_新疆 + 1)
年度净收益 = np.array([(年度收益_新疆 - 年运维成本_新疆) for _ in 年份])
累计净收益 = np.cumsum(年度净收益)
折现后的净收益 = np.array([(年度收益_新疆 - 年运维成本_新疆) / ((1 + 折现率)**t) for t in 年份])
累计折现后的净收益 = np.cumsum(折现后的净收益)

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制年度净收益和累计净收益
plt.plot(年份, 年度净收益, label='Annual Net Profit', color='blue', marker='o')
plt.plot(年份, 累计净收益, label='Cumulative Net Profit', color='green', marker='x')

# 绘制累计折现后的净收益
plt.plot(年份, 累计折现后的净收益, label='Cumulative Discounted Net Profit', color='red', linestyle='--')

# 添加标题和标签
plt.title('Financial Analysis of 10 MW Solar Power Plant in Xinjiang')
plt.xlabel('Years')
plt.ylabel('Profit (in billion dollars)')
plt.legend()

# 显示图表
plt.grid(True)
plt.show()
