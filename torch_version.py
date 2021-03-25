import os
import datetime
from sinabs.from_torch import from_model as transmodel
import wfdb
import pywt
import seaborn
import numpy as np
# import tensorflow as tf
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

root_path = "D:\\PythonFile\\ECG_CNN_Recognition"
model_path = root_path + "\\ecg_model.pkl"
snn_model_path = root_path + "\\snn_model.pkl"
# 划分测试集与训练集的比例
RATIO = 0.3


# 小波去躁预处理函数
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    # 对应5种异常类型
    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord('ecg_data/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('ecg_data/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


# 加载数据集并进行预处理
def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet)

    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 300)
    lableSet = np.array(lableSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, lableSet))
    np.random.shuffle(train_ds)

    # 数据集及其标签集
    X = train_ds[:, :300].reshape(-1, 1, 300)
    Y = train_ds[:, 300]

    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    test_length = int(RATIO * len(shuffle_index))
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test


# 使用pytorch构建CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=4,
                kernel_size=21,
                stride=1,
                padding=10
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=16,
                kernel_size=23,
                stride=1,
                padding=11
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=25,
                stride=1,
                padding=12
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=27,
                stride=1,
                padding=13
            ),
            nn.ReLU()

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 36, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)

        )
        # self.output = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        # out = out.view(out.size(0), -1)  # 展开
        out = self.fc(out)

        return out


# 定义一个我自己的dataloader 并继承dataset
class mydataloader(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)


# 混淆矩阵
def plotHeatMap(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    # 归一化
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    # 绘图
    plt.figure(figsize=(8, 8))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


def main():
    # X_train,Y_train为所有的数据集和标签集
    # X_test,Y_test为拆分的测试集和标签集
    X_train, Y_train, X_test, Y_test = loadData()
    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train)
    X_test = torch.from_numpy(X_test)
    # Y_test = torch.from_numpy(Y_test)
    print(X_train.shape)
    print(Y_train.shape)

    TrueDataSet = mydataloader(X_train, Y_train)
    data_for_train = DataLoader(TrueDataSet, batch_size=128, shuffle=True, drop_last=False, num_workers=2)
    if os.path.exists(model_path):
        # 导入训练好的模型
        model = torch.load(model_path)
    else:
        # 构建CNN模型
        model = CNN().double().cuda()  # 使用GPU加速
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        for epoch in range(30):
            tmp_loss_in_epoch = 0
            for i, (data, label) in enumerate(data_for_train):
                data = data.cuda()
                label = label.cuda()
                # GPU加速

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, label.long())
                loss.backward()
                optimizer.step()
                tmp_loss_in_epoch += loss.item()
                if i % 150 == 149:
                    print('[%d, %5d] loss: %.3f'
                          % (epoch + 1, i + 1, tmp_loss_in_epoch / 150))
                    tmp_loss_in_epoch = 0
        torch.save(model, model_path)

        # model =torchkeras.  CNN()
        # model.compile(optimizer='adam',
        #               loss='sparse_categorical_crossentropy',
        #               metrics=['accuracy'])
        # model.summary()
        # 定义TensorBoard对象
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # 训练与验证
        # model.fit(X_train, Y_train, epochs=30,
        #           batch_size=128,
        #           validation_split=RATIO,
        #           )
        # model.save(filepath=model_path)

    # 预测
    model.eval()
    with torch.no_grad():
        Y_pred = model.forward(X_test.cuda())
    Y_pred = Y_pred.data.cpu().numpy()
    Y2 = [np.argmax(Y_pred[i]) for i in range(len(Y_pred))]
    # 绘制混淆矩阵
    print(Y_pred[0:20])
    print(Y_test[0:20])
    plotHeatMap(Y_test, Y2)


def function_scnn():
    from tqdm import tqdm
    # model2 = model2.double()
    # 加载数据并进行数据格式转换
    X_train, Y_train, X_test, Y_test = loadData()
    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train)

    X_test = torch.from_numpy(X_test)
    X_test = X_test.type(torch.FloatTensor)

    # 训练snn模型
    TrueDataSet = mydataloader(X_train, Y_train)
    data_for_train = DataLoader(TrueDataSet, batch_size=128, shuffle=True, drop_last=False, num_workers=2)
    if os.path.exists(snn_model_path):
        # 导入训练好的模型
        model2 = torch.load(snn_model_path)
    else:
        # 构建CNN模型
        # model = CNN().double().cuda()  # 使用GPU加速
        # model = torch.load(model_path)
        # print(model)
        model2 = transmodel(CNN().float().cuda(), input_shape=(1, 300), add_spiking_output=True, synops=True)
        optimizer = torch.optim.Adam(model2.parameters())
        criterion = nn.CrossEntropyLoss()

        for epoch in tqdm(range(3)):
            tmp_loss_in_epoch = 0
            for i, (data, label) in tqdm(enumerate(data_for_train)):
                data = data.type(torch.FloatTensor).cuda()
                label = label.cuda()
                # GPU加速

                optimizer.zero_grad()
                output = model2(data)
                loss = criterion(output, label.long())
                loss.backward(retain_graph=True)
                optimizer.step()
                tmp_loss_in_epoch += loss.item()
                if i % 150 == 149:
                    print('[%d, %5d] loss: %.3f'
                          % (epoch + 1, i + 1, tmp_loss_in_epoch / 150))
                    tmp_loss_in_epoch = 0
        torch.save(model2, snn_model_path)

    # 固定snn模型
    model2.eval()
    print("model2:")
    print(model2)
    with torch.no_grad():
        Y_pred = model2.forward(X_test.cuda())
    Y_pred = Y_pred.data.cpu().numpy()

    Y2 = [np.argmax(Y_pred[i]) for i in range(len(Y_pred))]
    plotHeatMap(Y_test, Y2)

if __name__ == '__main__':
    function_scnn()
    # main()
