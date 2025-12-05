import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import time

start_time = time.time()
# ==========================================
# [修改 2]在此处控制是否开启时间特征
# True: 加入小时、星期等时间编码 (理论上效果更好)
# False: 仅使用原始5个特征
ENABLE_TIME_FEATURES = False
TRAIN_OR_INFERENCE = 0  # 0:训练，1：推理
# ==========================================

# 1.1 读取数据并清洗
# [建议] 建议使用英文路径或确保路径无特殊字符，这里保留你的路径
df_raw = pd.read_csv(r"E:\pycharmproject\电力交易\data\日前_实时_data_2024(含价差).csv")

# 1.2 确定输入特征和目标
INPUT = ['直调负荷', '联络线', '风电', '光伏', '竞价空间', '价差']
OUTPUT = '价差'
OUTPUT_EN = 'price_diff'  # 用于保存模型名

# 对应的输出的保存路径
save_dir = os.path.join(fr"E:\pycharmproject\pytorch_test", OUTPUT + "_训练数据only2024")
os.makedirs(save_dir, exist_ok=True)

# 模型保存路径
model_dir = os.path.join(save_dir, "model")
os.makedirs(model_dir, exist_ok=True)

# 测试图保存路径
picture_dir = os.path.join(save_dir, "模型测试图")
os.makedirs(picture_dir, exist_ok=True)

# 用于存储不同组合指标
results_list = []

# input_len_list = [1, 3, 7, 14, 30, 60]
input_len_list = [1]
# output_len_list = [1, 24]
output_len_list = [24]
if OUTPUT in INPUT:     # 预测目标历史序列作为特征
    df1 = df_raw[INPUT]
else:                   # 预测目标历史序列不作为特征
    df1 = df_raw[INPUT + [OUTPUT]]

print("原始数据缺失值：")
print(df1.isnull().sum())

df = df1.interpolate(method='linear', limit_direction='forward').copy()  # 插值处理缺失值

target_feature = df[[OUTPUT]].values  # 目标
# feature_cols = ['直调负荷', '联络线', '风电', '光伏', '实时价格']
feature_cols = INPUT

for col in feature_cols:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.replace(' ', '').astype(float)

input_features = df[feature_cols].values

print(f"当前输入特征维度: {len(feature_cols)}")


# -----------------------------------------------------------------------------
# 2. 构建数据集 (支持 step 参数)
# -----------------------------------------------------------------------------
class ElectricityDataset(Dataset):
    # [修改 4] 增加 step 参数，控制采样步长
    def __init__(self, data, target, seq_len=24 * 7, pred_len=24, step=1):
        self.X = []
        self.y = []
        # 制作样本
        # range(start, stop, step)
        for i in range(0, len(data) - seq_len - pred_len + 1, step):
            self.X.append(data[i: i + seq_len])
            self.y.append(target[i + seq_len: i + seq_len + pred_len])

        if len(self.X) > 0:
            self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
            self.y = torch.tensor(np.array(self.y), dtype=torch.float32).squeeze(-1)
        else:
            self.X = torch.empty(0)
            self.y = torch.empty(0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


for output_len in output_len_list:
    PRED_LEN = output_len
    for input_len in input_len_list:
        SEQ_LEN = 24 * input_len

        train_point = 7320  # 前305天7320条用于训练，后61天(11.1-1:00 ~ 12.31-24:00)1464条用于测试
        # train_point = 8784

        X_train_raw = input_features[:train_point]
        y_train_raw = target_feature[:train_point]

        # 测试集
        X_test_raw = input_features[train_point - SEQ_LEN + PRED_LEN:]
        y_test_raw = target_feature[train_point - SEQ_LEN + PRED_LEN:]

        # 划分完训练集和测试集后数据标准化
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        train_data_scaled = scaler_x.fit_transform(X_train_raw)
        train_target_scaled = scaler_y.fit_transform(y_train_raw)
        test_data_scaled = scaler_x.transform(X_test_raw)
        test_target_scaled = scaler_y.transform(y_test_raw)

        # 实例化 Dataset
        # 训练集：step=1 (密集采样，尽可能多地学习)
        train_dataset = ElectricityDataset(train_data_scaled, train_target_scaled, seq_len=SEQ_LEN, pred_len=PRED_LEN, step=1)

        # 测试集：step=24 (不重叠采样)
        # 这样预测出来的结果拼接起来就是一条连续的时间线，方便画图对比
        test_dataset = ElectricityDataset(test_data_scaled, test_target_scaled, seq_len=SEQ_LEN, pred_len=PRED_LEN, step=output_len)

        # 防止数据量不足导致报错
        if len(test_dataset) == 0:
            print("⚠️ 测试集数据不足以进行 step=24 的采样，退化为 step=1")
            test_dataset = ElectricityDataset(X_test_raw, y_test_raw, seq_len=SEQ_LEN, pred_len=PRED_LEN, step=1)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # 训练可以 Shuffle
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 测试不要 Shuffle

        print(f"训练集样本数: {len(train_dataset)}")
        print(f"测试集样本数: {len(test_dataset)}")

        # -----------------------------------------------------------------------------
        # 3. 定义 LSTM 模型 (自动适应输入维度)
        # -----------------------------------------------------------------------------
        class PricePredictor(nn.Module):
            def __init__(self, input_size, output_size):
                super(PricePredictor, self).__init__()
                # 四层LSTM
                self.lstm1 = nn.LSTM(input_size, hidden_size=16, batch_first=True)
                self.lstm2 = nn.LSTM(16, hidden_size=32, batch_first=True)
                self.lstm3 = nn.LSTM(32, hidden_size=64, batch_first=True)
                self.lstm4 = nn.LSTM(64, hidden_size=128, batch_first=True)
                # 三层全连接层降维
                self.fc1 = nn.Linear(128, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, output_size)
                # Dropout设置0.2
                self.dropout = nn.Dropout(p=0.2)

                self.relu = nn.ReLU()

            def forward(self, x):
                # 四层LSTM
                x, _ = self.lstm1(x)
                x = self.dropout(x)
                x, _ = self.lstm2(x)
                x = self.dropout(x)
                x, _ = self.lstm3(x)
                x = self.dropout(x)
                x, _ = self.lstm4(x)
                x = self.dropout(x)

                # 全连接层降维输出
                x = self.fc1(x[:, -1, :])
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc3(x)
                return x


        # -----------------------------------------------------------------------------
        # 4. 训练流程
        # -----------------------------------------------------------------------------
        if TRAIN_OR_INFERENCE == 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🚀 正在使用设备: {device}")

            # [修改 7] input_size 动态传入，取决于是否启用了时间特征
            model = PricePredictor(input_size=train_data_scaled.shape[1], output_size=PRED_LEN)
            model = model.to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

            epochs = 100  # 演示用50次，实际建议100+
            print("开始训练...")
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    avg_loss = epoch_loss / len(train_loader)
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

            torch.save(model.state_dict(), os.path.join(model_dir, f"LSTM({input_len}天输出下{output_len}点).pth"))
            print(f"保存模型到:{model_dir}\LSTM({input_len}天输出下{output_len}点).pth")

        # 直接读取保存模型参数
        else:
            device = torch.device("cpu")  # 预测通常不需要 GPU，CPU 足够快
            model = PricePredictor(input_size=5, output_size=PRED_LEN)

            MODEL_PATH = os.path.join(model_dir, f"LSTM({input_len}天输出下{output_len}点).pth")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)

        # -----------------------------------------------------------------------------
        # 5. [修改 8] 全测试集预测与评估
        # -----------------------------------------------------------------------------
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        model.eval()
        all_preds = []
        all_trues = []

        print("正在对测试集进行全量预测...")
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)

                # 预测
                pred = model(batch_X)

                # 收集数据 (转回CPU)
                all_preds.append(pred.cpu().numpy())
                all_trues.append(batch_y.numpy())

        # 拼接所有 Batch
        # 形状从 list of (Batch, 24) -> (Total_Samples, 24)
        np_preds = np.concatenate(all_preds, axis=0)
        np_trues = np.concatenate(all_trues, axis=0)

        # 反归一化
        real_preds = scaler_y.inverse_transform(np_preds)
        real_trues = scaler_y.inverse_transform(np_trues)

        # 拉平成一维序列以便计算指标和绘图
        flat_preds = real_preds.flatten()
        flat_trues = real_trues.flatten()


        def smape(y_true, y_pred):
            return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


        def sgn(x):
            if x > 0:
                return 1
            elif x == 0:
                return 0
            else:
                return -1


        def WSCR(y_actual, y_pred):  # 价差加权准确率
            a = 0
            b = 0
            for i in range(len(y_actual)):
                if sgn(y_actual[i]) == sgn(y_pred[i]):
                    a += abs(y_actual[i])
                b += abs(y_actual[i])
            return a / b


        def SCR(y_actual, y_pred):  # 价差方向准确率
            a = 0
            for i in range(len(y_actual)):
                if sgn(y_actual[i]) == sgn(y_pred[i]):
                    a += 1
            return a / len(y_actual)


        # 计算指标
        mae = mean_absolute_error(flat_trues, flat_preds)
        rmse = np.sqrt(mean_squared_error(flat_trues, flat_preds))
        smape1 = smape(flat_trues, flat_preds)
        wscr_val = None
        scr_val = None

        print(f"\n===== 评估结果 (Time Features: {ENABLE_TIME_FEATURES}) =====")
        print(f"MAE (平均绝对误差): {mae:.4f}")
        print(f"RMSE (均方根误差): {rmse:.4f}")
        print(f"SMAPE: {smape1:.4f}")
        if OUTPUT == "价差":
            wscr_val = WSCR(flat_trues, flat_preds)
            scr_val = SCR(flat_trues, flat_preds)
            print("价差加权准确率：", wscr_val)
            print("价差方向准确率：", scr_val)

        current_result = {
            "Input_Days_len": input_len,
            "Output_Points": output_len,
            "MAE": mae,
            "RMSE": rmse,
            "SMAPE": smape1,
            "WSCR": wscr_val,  # 如果不是价差，这里是 None
            "SCR": scr_val  # 如果不是价差，这里是 None
        }
        results_list.append(current_result)

        # 绘图
        plt.figure(figsize=(15, 6))
        plt.plot(flat_trues[-360:], label=f'真实{OUTPUT}', color='blue', alpha=0.7)
        plt.plot(flat_preds[-360:], label=f'预测{OUTPUT}', color='red', alpha=0.7, linestyle='--')
        plt.title(f'({input_len}天输出下{output_len}点){OUTPUT}预测对比 (MAE: {mae:.2f}, TimeFeat: {ENABLE_TIME_FEATURES})')
        plt.xlabel('时间 (Hours)')
        plt.ylabel(OUTPUT)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(picture_dir, f'LSTM_{input_len}天输出下{output_len}点.png'))
        print(f"预测图保存到:{picture_dir}\LSTM_{input_len}天输出下{output_len}点.png")
        try:
            plt.show()
        except Exception as e:
            print("显示图形时出现错误:", e)
        # print(f"图形已保存为 '实时电价预测(lstm)_{ENABLE_TIME_FEATURES}.png',请查看该文件。")

# 循环结束后，保存所有结果到 CSV
print("\n" + "=" * 30)
print("所有组合训练结束，正在保存汇总指标...")

results_df = pd.DataFrame(results_list)

# 保存路径设置
save_csv_path = os.path.join(save_dir, "不同输入输出长度对比(只用历史数据).csv")
# 确保目录存在
os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
results_df.to_csv(save_csv_path, index=False)
print(f"指标汇总已保存至: {save_csv_path}")

end_time = time.time()
run_time = end_time - start_time
print(f"程序运行时间: {run_time:.4f}秒")
print(f"程序运行时间: {run_time / 60:.4f}分")
