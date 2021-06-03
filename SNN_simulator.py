from torch_version import *
from spiking_model import *

root_path = "D:\\PythonFile\\ECG_CNN_Recognition"
model_path = root_path + "\\ecg_model.pkl"
snn_model_path = root_path + "\\snn_model.pkl"
snn_model_path2 = root_path + "\\snn_model2.pkl"

if __name__ == '__main__':
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
    data_for_train = DataLoader(TrueDataSet, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    if os.path.exists(snn_model_path):
        # 导入训练好的模型
        model = torch.load(snn_model_path)
    else:
        # 构建CNN模型
        model = SCNN().float().cuda()  # 使用GPU加速
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        for epoch in range(10):
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
        torch.save(model, snn_model_path)

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
    model.eval()
    print(X_test.shape)
    ACC_sum = []
    # X_test = X_test[9:].view(128, -1, 300)
    TrueTestSet = mydataloader(X_test, Y_test)
    data_for_test = DataLoader(TrueTestSet, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
    with torch.no_grad():
        for i, (data, label) in enumerate(data_for_test):
            Y_pred = model.forward(data.cuda())
            # Y_pred2.append(Y_pred.data.cpu().numpy())
            Y_pred = Y_pred.data.cpu().numpy()
            Y_label = [np.argmax(Y_pred[i]) for i in range(len(Y_pred))]
            mat_acc = [0 if Y_label[i] == label[i] else 1 for i in range(len(Y_label))]
            acc = 1 - np.sum(mat_acc) / float(len(mat_acc))
            ACC_sum.append(acc)


    print("here")
    # Y_pred2 = Y_pred2.cpu()
    print(ACC_sum)
    print(np.average(ACC_sum))
    # Y_predT = Y_predT.cpu()
    # Y2 = [np.argmax(Y_pred2[i]) for i in range(len(Y_pred2))]
    # # 绘制混淆矩阵
    # print(Y_pred2[0:20])
    # print(Y_test[0:20])
    # plotHeatMap(Y_test[:-9], Y2)
