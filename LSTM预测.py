import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import tensorflow as tf

# 设置matplotlib的风格和参数
style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'  # 设置图表背景色为白色
plt.rcParams['axes.facecolor'] = 'white'   # 设置坐标轴背景色为白色
plt.rc('axes', titlesize=20)     # 设置轴标题的大小
plt.rc('axes', labelsize=20)     # 设置轴标签的大小
plt.rc('legend', fontsize=25)    # 设置图例的大小
plt.rc('xtick', labelsize=20)    # 设置x轴刻度标签的大小
plt.rc('ytick', labelsize=20)    # 设置y轴刻度标签的大小
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定中文显示字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负数坐标轴显示问题
plt.rcParams['font.size'] = 18                # 设置字体大小

# 读取数据
file_path = '第一问的数据.xlsx'  # 更改为您的文件路径
data = pd.read_excel(file_path)
data.set_index('年份', inplace=True)

# 选择总用电量作为目标变量
target_variable = '总用电量（万亿千瓦时）'
values = data[[target_variable]].values

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

# 创建数据集
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1
X_all, Y_all = create_dataset(scaled_values, look_back)
X_all = np.reshape(X_all, (X_all.shape[0], 1, X_all.shape[1]))

#prepare train and test sets
train_size = int(len(X_all) * 0.8)
test_size = len(X_all) - train_size
X_train, X_test = X_all[0:train_size, :], X_all[train_size:len(X_all), :]
Y_train, Y_test = Y_all[0:train_size], Y_all[train_size:len(Y_all)]

# 构建LSTM模型
model = Sequential()
model.add(LSTM(17, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_all, Y_all, epochs=100, batch_size=1, verbose=2)

# 预测
all_predict = model.predict(X_all)
all_predict = scaler.inverse_transform(all_predict)
Y_all_inv = scaler.inverse_transform([Y_all])

# 绘制模型的预测结果与原始数据
plt.figure(figsize=(12, 6))
plt.plot(data.index, values, label='Original Data', color='green', linestyle='--', linewidth=2)
plt.plot(data.index[look_back:], all_predict[:,0], label='Model Predictions', color='red', linestyle='-', linewidth=2)
plt.grid(True, linestyle='--', color='black', linewidth=0.5, alpha=0.7)
plt.title('LSTM Model Predictions vs Original Data', fontsize=20, fontweight='bold', fontname='Times New Roman')
plt.xlabel('Year', fontsize=20, fontname='Times New Roman')
plt.ylabel('Total Power Consumption (Trillion kWh)', fontsize=20, fontname='Times New Roman')  # 修改y轴标签为英文
plt.legend()
plt.tight_layout()
plt.show()


# 计算评价指标
mse = mean_squared_error(Y_all_inv[0], all_predict[:,0])
rmse = sqrt(mse)
mae = mean_absolute_error(Y_all_inv[0], all_predict[:,0])
r2 = r2_score(Y_all_inv[0], all_predict[:,0])
print(f"All Data - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}")

# 绘制训练历史
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='black', linewidth=2)
plt.grid(True, linestyle='--', color='black', linewidth=0.5, alpha=0.7)
plt.title('Model Training History', fontsize=20, fontweight='bold', fontname='Times New Roman')
plt.xlabel('Epoch', fontsize=20, fontname='Times New Roman')
plt.ylabel('Loss', fontsize=20, fontname='Times New Roman')
plt.legend()
plt.tight_layout()
plt.show()

# 预测未来的电力消耗
def predict_future(last_value, model, scaler, steps=36):
    prediction_list = last_value
    for _ in range(steps):
        x = prediction_list[-look_back:]
        x = x.reshape((1, 1, look_back))
        next_prediction = model.predict(x, verbose=0)
        prediction_list = np.append(prediction_list, next_prediction)
    prediction_list = prediction_list[look_back-1:]
    return scaler.inverse_transform(prediction_list.reshape(-1, 1))

# 使用最后一个已知值开始预测
last_known_value = scaled_values[-1]
future_predictions = predict_future(last_known_value, model, scaler, steps=36)

# 将未来年份添加到预测中
future_years = np.arange(2024, 2061)
future_df = pd.DataFrame(future_predictions, index=future_years, columns=[target_variable])

# 绘制未来年份的预测图
plt.figure(figsize=(12, 6))
plt.plot(future_df.index, future_df[target_variable], label='Future Predictions', color='#FF0000', linestyle='-', linewidth=3)
plt.grid(True, linestyle='--', color='black', linewidth=0.5, alpha=0.7)
plt.title('Clean Energy Generation Prediction(2024-2060)', fontsize=20, fontweight='bold', fontname='Times New Roman')
plt.xlabel('Year', fontsize=20, fontname='Times New Roman')
plt.ylabel('Total Power Consumption (Trillion kWh)', fontsize=20, fontname='Times New Roman')
plt.legend()
plt.tight_layout()
plt.show()
