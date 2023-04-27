import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
n_epochs = 3            # 循环整个训练数据集的次数
batch_size_train = 64   # 每个训练batch有多少个样本 比如有1000个样本，把这些样本分为10批，就是10个batch
batch_size_test = 1000  # 每个测试训练batch有多少个样本
learning_rate = 0.01    # 学习率：learning_rate = 0.01
momentum = 0.5          
# momentum：我们把这个人从平地上放到了一个斜坡上, 只要他往下坡的方向走一点点, 由于向下的惯性, 他不自觉地就一直往下走,
# 走的弯路也变少了. 这就是 Momentum 参数更新
log_interval = 10       # 日志输出间隔，多少次print一下
random_seed = 1         # 随机种子
torch.manual_seed(random_seed)
 
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(r'E:\handwriting_recognize\DATASET/', train=True, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
 
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(r'E:\handwriting_recognize\DATASET/', train=False, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)
 
# torch.utils.data.DataLoader
# 用来把训练数据分成多个小组，此函数每次抛出一组数据。直至把所有的数据都抛出。最终的输出结果是一整块内容，内容内部是一批一批的
# 就是做一个数据的初始化。
# 批训练，把数据变成一小批一小批数据进行训练。
 
# torchvision：视觉工具包
# torchvision.datasets.MNIST：在pytorch下可以直接调用torchvision.datasets里面的MNIST数据集
 
# transforms：提供常用的数据预处理操作
# transforms.Compose:将多个变换方式结合在一起
# 参数：一个list数组，数组里是多个'Transform'对象，即[transforms, transforms...]
# 操作：遍历list数组，对img依次执行每个transforms操作，并返回transforms后的img
 
 
# 我们一般在pytorch或者python中处理的图像无非这几种格式：
# PIL：使用python自带图像处理库读取出来的图片格式
# numpy：使用python-opencv库读取出来的图片格式
# tensor：pytorch中训练时所采取的向量格式（当然也可以说图片）
 
# transforms.ToTensor:将(范围在0-255) 转成(范围为0.0-1.0)  返回Tensor类型的图片
# 一方面将图片数据shape成【channel, w, h】,另一方面将图片array转化为tensor
# Tensor有不同的数据类型，每种类型又有CPU和GPU两种版本，默认的tensor类型是FloatTensor
# 有点像numpy中的array
 
# transforms.Normalize(mean,std):标准化（归一化）——data减去它的均值，再除以它的标准差
# 最终呈现均值为0方差为1的数据分布
# mean (squence):每个通道的均值
# std (sequence):每个通道的标准差
# 还需要保持train_set、val_set和test_set标准化系数的一致性。
# 标准化系数就是计算要用到的均值和标准差，在本例中是((0.1307,), (0.3081,))，均值是0.1307，标准差是0.3081
# 这些系数都是数据集提供方计算好的数据
# 因为mnist数据值都是灰度图，所以图像的通道数只有一个，因此均值和标准差各一个
 
# batch_size(int, optional): 每个batch有多少个样本
# shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序
# 先打乱，再取batch
# Shuffle是一种训练的技巧，因为机器学习其假设和对数据的要求就是要满足独立同分布。
# 所以任何样本的出现都需要满足“随机性”。所以在数据有较强的“人为”次序特征的情况下，Shuffle显得至关重要。
# 1、Shuffle可以防止训练过程中的模型抖动，有利于模型的健壮性
# 假设训练数据分为两类，在未经过Shuffle的训练时，首先模型的参数会去拟合第一类数据，当大量的连续数据（第一类）输入训练时，会造成模型在第一类数据上的过拟合。
# 当第一类数据学习结束后模型又开始对大量的第二类数据进行学习，这样会使模型尽力去逼近第二类数据，造成新的过拟合现象。
# 这样反复的训练模型会在两种过拟合之间徘徊，造成模型的抖动，也不利于模型的收敛和训练的快速收敛
# 2、Shuffle可以防止过拟合，并且使得模型学到更加正确的特征
# NN网络的学习能力很强，如果数据未经过打乱，则模型反复依次序学习数据的特征，很快就会达到过拟合状态，并且有可能学会的只是数据的次序特征。模型的缺乏泛化能力。
# 如：100条数据中前50条为A类剩余50条为B类，模型在很短的学习过程中就学会了50位分界点，且前半部分为A后半部分为B。则并没有学会真正的类别特征。
 
examples = enumerate(test_loader)
# enumerate()：遍历一个集合对象，它在遍历的同时还可以得到当前元素的索引位置。
# 遍历了列表的所有元素，并通过增加从零开始的计数器变量来为每个元素生成索引
batch_idx, (example_data, example_targets) = next(examples)
# next()：指针指向下一条记录，有记录（有值）返回true并把记录内容存入到对应的对象中
 
# print(example_targets)
# 输出：tensor([3, 9, 4, 9, 9, 0, 8,....
# example_targets是图片实际对应的数字标签：
 
# print(example_data.shape)
# 输出：torch.Size([1000, 1, 28, 28])  一批测试数据是一个形状张量
# 这意味着我们有1000个例子的28x28像素的灰度
 
class Net(nn.Module):
    # nn.Module其实是PyTorch体系下所有神经网络模块的基类，这里其实是继承
    # 在Pytorch的nn模块中，它是不需要你手动定义网络层的权重和偏置的
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 两个卷积层
        # 输入通道数，输出通道数，卷积核大小（5*5）
        self.conv2_drop = nn.Dropout2d()
        # torch.nn.Dropout2d是对每个通道按照概率0.5置为0
        # 防止模型过拟合
        # 神经网络的输入单元是否归零服从伯努利分布，并以概率p随机地将神经网络的输入单元归零。
        # 一般用在全连接层
        # 也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
        # 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # 两个全连接层
        # 处理后后会输出一个N维向量，N是该程序必须选择的分类数量。
        # 如果你想得到一个数字分类程序，如果有10个数字，N就等于10
        # in_features: int, out_features: int, bias: bool = True
 
    def forward(self, x):
        # forward()传递定义了使用给定的层和函数计算输出的方式。
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # kernel_size=2  kernel_size可以看做是一个滑动窗口 2*2
        # max_pool2d()  把4个合并成1个
        # 最大池化层
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # relu 激活运算
        # 没有激活函数的每层都相当于矩阵相乘。
        # 给一个在卷积层中刚经过线性计算操作的系统引入非线性特征
        # 线性方程的复杂度有限，从数据中学习复杂函数映射的能力很小。
        # 对输入内容的所有值都应用了函数 f(x) = max(0, x)
        # 会增加模型乃至整个神经网络的非线性特征，而且不会影响卷积层的感受野
        # 激活函数可以把当前特征空间通过一定的线性映射转换到另一个空间，让数据能够更好的被分类
        x = x.view(-1, 320)
        # 参数中的-1就代表这个位置由其他位置的数字来推断
        # 比如atensor的数据个数是6个，如果view（1，-1），我们就可以根据tensor的元素个数推断出-1代表6
        # 将提取出的特征图进行铺平，将特征图转换为一维向量
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        # log_softmax()
        # 先softmax,再作log, 其实就是复合函数
        # softmax激活函数的计算方式是对输入的每个元素值x求以自然常数e为底的指数，然后再分别除以他们的和
        # logsoftmax其实就是对softmax求出来的值再求一次log值
        # 加快运算速度，提高数据稳定性
 
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
# optim.SGD()
# 这个对象能保存当前的参数状态并且基于计算梯度更新参数
# 学习率：learning_rate = 0.01
# 学习率以 0.01 ~ 0.001 为宜。
# 学习率过大时，学习速率很快，但是易损失值爆炸，易震荡
# 学习率过小时，速度很慢，容易过拟合，收敛速度也慢
# momentum：我们把这个人从平地上放到了一个斜坡上, 只要他往下坡的方向走一点点, 由于向下的惯性, 他不自觉地就一直往下走,
# 走的弯路也变少了. 这就是 Momentum 参数更新
 
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
 
def train(epoch):
    network.train()
    # pytorch可以给我们提供两种方式来切换训练和评估(推断)的模式。分别是：model.train()和model.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()   # 将梯度归零  梯度就表示从该点出发，函数值增长最为迅猛的方向
        output = network(data)
        loss = F.nll_loss(output, target)   # 损失函数计算损失值
        loss.backward()     # 反向传播计算得到每个参数的梯度值
        optimizer.step()    # 通过梯度下降执行一步参数更新，会更新所有的参数
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')
 
def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():   # 所有计算得出的tensor的requires_grad都自动设置为False
        # requires_grad：是Pytorch中通用数据结构Tensor的一个属性，用于说明当前量是否需要在计算中保留对应的梯度信息
        # 我的理解是因为这是测试，不需要再对参数进行修改了
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # reduction：对计算结果采取的操作，通常我们用sum（对N个误差结果求和）, mean（对N个误差结果取平均），默认是对所有样本求loss均值
            pred = output.data.max(1, keepdim=True)[1]
            # .data.max用于找概率最大的下标
            correct += pred.eq(target.data.view_as(pred)).sum()
            # 正确的个数
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # Test set: Avg.loss: 0.0921, Accuracy: 9709 / 10000(97 %)
 
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
    print("*"*60)