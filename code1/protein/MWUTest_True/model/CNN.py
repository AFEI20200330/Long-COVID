import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

# 设置随机种子以确保结果可复现
torch.manual_seed(0)

# 假设数据已加载并准备好，X是特征，y是标签
# 这里使用假数据作为示例
X = np.random.rand(100, 1, 28, 28)  # 100个样本，1个颜色通道，28x28像素
y = np.random.randint(0, 3, 100)  # 3个类别的标签
y = label_binarize(y, classes=[0, 1, 2])  # 用于ROC计算的二进制化标签

# 转换为torch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


# 初始化模型、损失函数和优化器
model = CNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)  # 每25个epoch学习率减小为原来的0.1


# 早停技术实现
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# 训练模型
def train_model(model, train_loader, criterion, optimizer, scheduler, n_epochs=50):
    early_stopping = EarlyStopping(patience=10, min_delta=0.01)
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # 打印统计信息
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, n_epochs, running_loss / len(train_loader)))

        # 早停检查
        early_stopping(running_loss / len(train_loader))
        if early_stopping.early_stop:
            print("早停！")
            break


# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    all_targets, all_outputs = [], []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            all_targets.extend(target.numpy())
            all_outputs.extend(torch.sigmoid(output).numpy())
    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)

    # 计算评估指标
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):  # 计算每个类的ROC
        fpr[i], tpr[i], _ = roc_curve(all_targets[:, i], all_outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制ROC曲线
    for i in range(3):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    print('Accuracy:', accuracy_score(all_targets.argmax(axis=1), all_outputs.argmax(axis=1)))
    print('Precision:', precision_score(all_targets.argmax(axis=1), all_outputs.argmax(axis=1), average='macro'))
    print('Recall:', recall_score(all_targets.argmax(axis=1), all_outputs.argmax(axis=1), average='macro'))
    print('F1 Score:', f1_score(all_targets.argmax(axis=1), all_outputs.argmax(axis=1), average='macro'))
    print('Confusion Matrix:\n', confusion_matrix(all_targets.argmax(axis=1), all_outputs.argmax(axis=1)))


# 运行训练和评估
train_model(model, train_loader, criterion, optimizer, scheduler, n_epochs=100)
evaluate_model(model, test_loader)
