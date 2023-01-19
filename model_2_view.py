import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as da
from timm.loss import LabelSmoothingCrossEntropy
import argparse
from GTFENet import the_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data_dir', type=str, default= "data\\5HP", help='the directory of the data')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=256, help='batchsize of the training process')
    parser.add_argument('--step_len', type=list, default=range(210, 430, 10), help='the weight decay')
    parser.add_argument('--sample_len', type=int, default=420, help='the learning rate schedule')
    parser.add_argument('--rate', type=list, default=[0.7, 0.15, 0.15], help='')
    parser.add_argument('--acces', type=list, default=[], help='initialization list')
    parser.add_argument('--epochs', type=int, default=80, help='max number of epoch')
    parser.add_argument('--losses', type=list, default=[], help='initialization list')
    args = parser.parse_args()
    return args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dataset(da.Dataset):
    def __init__(self, X, y):
        self.Data = X
        self.Label = y
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)

# STEP1 加载数据
def load_data(args):
    # path = r'data\5HP'
    # path = args.data_dir
    # rate = args.rate
    source_data = np.load('H:\GTFENet\cwrudata_data.npy')
    source_label = np.load('H:\GTFENet\cwrudata_label.npy').argmax(axis=-1)
    source_data = StandardScaler().fit_transform(source_data.T).T
    source_data = np.expand_dims(source_data, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(source_data, source_label, test_size=0.3, random_state=0, stratify=source_label)
    x_train, x_validate, y_train,  y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=0, stratify=y_train)
    # 切片
    # sample = tf.data.Dataset.from_tensor_slices((x_train, y_train))   # 按照样本数进行切片得到每一片的表述（2048+10，1）
    # sample = sample.shuffle(1000).batch(10)  # 打乱分批量(10,400,2)
    # sample_validate = tf.data.Dataset.from_tensor_slices((x_validate, y_validate))
    # sample_validate = sample_validate.shuffle(1000).batch(10)  # 打乱分批量(10,400,2)
    # sample_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # sample_test = sample_test.shuffle(1000).batch(10)  # 打乱分批量(10,400,2)
    Train = Dataset(x_train, y_train)
    Validate = Dataset(x_validate, y_validate)
    Test = Dataset(x_test, y_test)
    train_loader = da.DataLoader(Train, batch_size=args.batch_size, shuffle=True)
    validate_loader = da.DataLoader(Validate, batch_size=args.batch_size, shuffle=True)
    test_loader = da.DataLoader(Test, batch_size=args.batch_size, shuffle=False)
    return train_loader, validate_loader, test_loader


# STEP2 设计网络结构，建立网络容器
# def create_model():
#     Con_net = keras.Sequential([  # 网络容器
#         layers.Conv1D(filters=32, kernel_size=20, strides=1, padding='same', activation='relu'),  # 添加卷积层
#         layers.BatchNormalization(),  # 添加正则化层
#         layers.MaxPooling1D(pool_size=2, strides=2, padding='same'),  # 池化层
#         layers.Conv1D(filters=32, kernel_size=20, strides=1, padding='same', activation='relu'),  # 添加卷积层
#         layers.BatchNormalization(),  # 添加正则化层
#         layers.MaxPooling1D(pool_size=2, strides=2, padding='same'),  # 池化层
#         layers.Flatten(),  # 打平层，方便全连接层使用
#         layers.Dense(100, activation='relu'),  # 全连接层，120个节点
#         layers.Dense(10, activation='softmax'),  # 全连接层，10个类别节点
#     ])
#     return Con_net
# class Con_net(nn.Module):
#     def __init__(self):
#         super(Con_net, self).__init__()
#         self.p1_1 = nn.Sequential(nn.Conv1d(2, 32, kernel_size=20, stride=1, padding='same'),
#                                   nn.BatchNorm1d(32),
#                                   nn.ReLU(inplace=True))
#         self.p1_2 = nn.MaxPool1d(2, 2)
#         self.p2_1 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=20, stride=1, padding='same'),
#                                   nn.BatchNorm1d(32),
#                                   nn.ReLU(inplace=True))
#         self.p2_2 = nn.MaxPool1d(2, 2)
#         self.p3_1 = nn.Sequential(nn.Linear(32*105, 100),     #需要根据输出修改前置神经元个数
#                                   nn.ReLU(inplace=True))   #全连接层之后还需要加激活函数
#         self.p3_2 = nn.Sequential(nn.Linear(100, 10))
#
#     def forward(self, x):
#         x = self.p1_2(self.p1_1(x))
#         x = self.p2_2(self.p2_1(x))
#         x = x.reshape(-1, x.size()[1]*x.size()[2])
#         x = self.p3_2(self.p3_1(x))
#         return x

model = the_model().to(device)

def train(args, train_loader, validate_loader, test_loader):
    res = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = LabelSmoothingCrossEntropy()
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for epoch in range(args.epochs):
        for step, (img, label) in enumerate(train_loader):
            img = img.float()
            img = img.to(device)
            label = label.to(device)
            label = label.long()
            out = model(img)
            out = torch.squeeze(out).float()
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc
            if step % 37 == 0:
                print(epoch, step, 'loss:', float(loss))
            # print(epoch, step, 'loss:', float(loss))
            res.append(test(test_loader))
        result.append(res)
    return test(test_loader)
        # torch.save(model.state_dict(), './cnn_save_weights_400.pt')

# def train(sample1, sample1_validate, sample1_test, sample_len):
#     res = []
#     Con_net = create_model()  # 建立网络模型
#     Con_net.build(input_shape=(10, sample_len, 2))  # 构建一个卷积网络，输入的尺寸  ----------------------
#     optimizer = optimizers.Adam(lr=1e-4)  # 设置优化器
#     variables = Con_net.trainable_variables
#     for epoch in range(epochs):  # 外循环，遍历多少次训练数据集
#         for step, (x, y) in enumerate(sample1):  # 遍历一次训练集的所有样例
#             with tf.GradientTape() as tape:  # 构建梯度环境 # [b, 32, 32, 3] => [b, 1, 1, 512]
#                 out = Con_net(x)  # flatten, => [b, 512]
#                 loss = tf.losses.categorical_crossentropy(y, out)  # compute loss
#                 loss = tf.reduce_mean(loss)  # 求损失的平均值
#             grads = tape.gradient(loss, variables)
#             optimizer.apply_gradients(zip(grads, variables))
#             if step % 1000 == 0:
#                 print(epoch, step, 'loss:', float(loss))
#         # print("验证集正确率")
#         # test(Con_net, sample1_validate)
#         # print("测试集正确率")
#         res.append(test(Con_net, sample1_test))
#     result.append(res)
#     # Con_net.save_weights('./cnn_save_weights_400')

def test(sample_data):

    test_acc = 0.
    model.eval()
    for img, label in sample_data:
        # torch.load('./cnn_save_weights_400.pt')
        img = img.float()
        img = img.to(device)
        label = label.to(device)
        label = label.long()
        out = model(img)
        out = torch.squeeze(out).float()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        test_acc += acc
    acc = test_acc / len(sample_data)
    # print('acc:', acc)
    return acc
#
#
# def test(Con_net, sample_data):
#     total_num = 0
#     total_correct = 0
#     for x, y in sample_data:
#         # Con_net = create_model()  # 建立网络模型
#         # Con_net.load_weights('./cnn_save_weights_400')
#         out = Con_net(x)  # 前向计算
#         predict = tf.argmax(out, axis=-1)  # axis=-1, 倒数第一维, 返回每行的最大值坐标
#         # print("predict", predict)
#         y = tf.cast(y, tf.int64)
#         # print("y", y)
#         m = predict == y
#         m = tf.cast(m, dtype=tf.int64)   # tensor张量类型
#         total_correct += int(tf.reduce_sum(m))
#         total_num += x.shape[0]
#         if total_num < total_correct:
#             print("error---------------------------")
#             print("正确",total_correct,"总数",total_num)
#     acc = total_correct / total_num
#     # print('acc:', acc)
#     return acc

def run_step(args):  # epoch=10
    # step_len = list(range(210, 430, 10))
    #step_len = [420]
    # step_len = [210]
    sample1, sample1_validate, sample1_test = load_data(args)
    acc = train(args, sample1, sample1_validate, sample1_test)
    print(acc)


# def run_sample():
#     sample_len = list(range(1,7))
#     # sample_len = [1]
#     for i in sample_len:
#         sample1, sample1_validate, sample1_test = load_data(step=210, sample_len=420*i)
#         train(sample1, sample1_validate, sample1_test, sample_len=420*i)


# 当epoch=10时，随着步长的变化，实验结果的变化
if __name__ == '__main__':
    args = parse_args()
    result = []
    run_step(args)
    # run_sample()
    print(result)


