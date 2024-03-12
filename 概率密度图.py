import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# 设置 Matplotlib 的默认字体修改为中文字体，字号修改为12
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负数坐标轴显示问题
plt.rcParams['font.size'] = 12                # 指定字号

# 设置Seaborn的样式为简洁高级
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_palette("deep")
sns.set(font="SimSun", font_scale=1.2)

# 指定 Excel 文件的路径
file_path ="D:\电脑桌面\华数B题\第一问的求解\第一问的数据.xlsx"

# 读取数据（假设数据在第一个工作表中）
data = pd.read_excel(file_path, sheet_name=0)

# 存储变换后的数据
transformed_data = pd.DataFrame()

# 遍历每一列数据，绘制概率密度分布图
for col in data.columns:
    # 绘制原始数据的概率密度分布图
    sns.histplot(data[col], kde=True, color='blue', alpha=0.5, label='原始数据')
    plt.title(f"{col}的分布情况")
    plt.xlabel(f"{col}的取值范围")
    plt.ylabel("数量")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(f"{col}_分布图.png", dpi=150)
    plt.tight_layout()
    plt.show()

    # 如果该列数据的分布类型非正态分布，则进行Yeo-Johnson转换
    if not np.isclose(np.abs(stats.skew(data[col])), 0, atol=0.05):
        transformed_col, max_log_likelihood = stats.yeojohnson(data[col].dropna())
        transformed_col = pd.Series(transformed_col, name=f"{col}_yeo-johnson")
        
        # 绘制转换后数据的概率密度分布图
        sns.histplot(transformed_col, kde=True, color='red', alpha=0.5, label='转换后数据')
        plt.title(f"{col}的分布情况（Yeo-Johnson转换后）")
        plt.xlabel(f"{col}的取值范围（Yeo-Johnson转换后）")
        plt.ylabel("数量")
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.savefig(f"{col}_Yeo-Johnson分布图.png", dpi=150)
        plt.tight_layout()
        
        plt.show()

        # 将处理后的数据存储到DataFrame中
        transformed_data[f"{col}_yeo-johnson"] = transformed_col
    else:
        # 将未经处理的数据存储到DataFrame中
        transformed_data[col] = data[col]

# 将合并后的数据写入Excel文件中
writer = pd.ExcelWriter('转换后的数据.xlsx', engine='xlsxwriter')
transformed_data.to_excel(writer, sheet_name='数据')
writer.save()