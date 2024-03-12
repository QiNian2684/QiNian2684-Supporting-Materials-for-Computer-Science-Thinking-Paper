import numpy as np
import pandas as pd

# 示例数据，用于展示数据结构
years = range(1990, 2024)
total_electricity = np.random.uniform(5, 10, len(years))  # 估计用电量，单位：亿千瓦时
thermal_power_capacity = np.random.uniform(200, 500, len(years))  # 估计火电装机容量，单位：千瓦
wind_power_capacity = np.random.uniform(50, 150, len(years))  # 估计风电装机容量，单位：千瓦
solar_power_capacity = np.random.uniform(30, 100, len(years))  # 估计太阳能装机容量，单位：千瓦
hydro_power_capacity = np.random.uniform(100, 200, len(years))  # 估计水电装机容量，单位：千瓦
non_fossil_fuel_ratio = np.random.uniform(10, 40, len(years))  # 估计非化石能源比重
industrial_electrification = np.random.uniform(20, 70, len(years))  # 估计工业部门电气化率
building_electrification = np.random.uniform(30, 80, len(years))  # 估计建筑部门电气化率
transport_electrification = np.random.uniform(5, 60, len(years))  # 估计交通部门电气化率
coal_power_capacity = np.random.uniform(300, 600, len(years))  # 估计煤电装机容量
gas_power_capacity = np.random.uniform(50, 150, len(years))  # 估计气电装机容量
nuclear_power_capacity = np.random.uniform(20, 100, len(years))  # 估计核电装机容量

# 创建数据表
data_filled = {
    "年份": years,
    "总用电量（万亿千瓦时）": total_electricity,
    "火电装机容量（千瓦）": thermal_power_capacity,
    "风电装机容量（千瓦）": wind_power_capacity,
    "太阳能装机容量（千瓦）": solar_power_capacity,
    "水电装机容量（千瓦）": hydro_power_capacity,
    "非化石能源比重": non_fossil_fuel_ratio,
    "工业部门电气化率": industrial_electrification,
    "建筑部门电气化率": building_electrification,
    "交通部门电气化率": transport_electrification,
    "煤电装机容量（千瓦）": coal_power_capacity,
    "气电装机容量（千瓦）": gas_power_capacity,
    "核电装机容量（千瓦）": nuclear_power_capacity
}

df_filled = pd.DataFrame(data_filled)

# Save to Excel file
file_path_filled = '第一问的数据.xlsx'

# 完成文件的保存操作
df_filled.to_excel(file_path_filled, index=False)
