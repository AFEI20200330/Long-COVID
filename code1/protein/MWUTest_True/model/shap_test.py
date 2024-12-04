# @FirstAuthor: Fay
# @Coding_Time: 2024/12/4/004 下午 5:42
# @Description：
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn import metrics
import copy

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载和预处理
data_path = '../filtered_significant_proteins.xlsx'  # 更新为实际路径
data = pd.read_excel(data_path)
X = data.drop(['Target_new', 'Blood_Sample_ID', 'Cluster'], axis=1).values
y = data['Cluster'].values

# 划分数据集并进行特征缩放
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义带自注意力的MLP模型
class SelfAttentionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SelfAttentionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        attention_scores = F.softmax(self.attention(x), dim=1)
        attended = attention_scores * x
        x = self.fc2(attended)
        return x


# 模型实例化
model = SelfAttentionMLP(input_dim=X_train.shape[1], hidden_dim=128, output_dim=len(np.unique(y))).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


# 早停技术设置
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.val_loss_min = float('inf')  # 初始化为无限大

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''保存模型时更新最小验证损失'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss  # 更新最小验证损失

early_stopper = EarlyStopping(patience=10, verbose=True)


# 训练过程
def train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs=50):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        scheduler.step(val_loss)
        early_stopper(val_loss, model)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if early_stopper.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = corrects.double() / len(test_loader.dataset)
    return epoch_loss, epoch_acc.item()


# 训练和评估模型
final_model = train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs=50)
test_loss, test_accuracy = evaluate_model(final_model, test_loader, criterion)

# 输出所有评估结果
print('Test Loss: {:.4f}'.format(test_loss))
print('Test Accuracy: {:.4f}'.format(test_accuracy))

y_pred = []
y_true = []

# 收集预测和实际标签
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        y_pred.extend(preds.view(-1).tolist())
        y_true.extend(labels.view(-1).tolist())

# 计算混淆矩阵和其他指标
conf_matrix = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
roc_auc = roc_auc_score(y_true, [p[1] for p in
                                 model(torch.tensor(X_test, dtype=torch.float32).to(device)).detach().cpu().numpy()],
                        multi_class='ovo')

print('Confusion Matrix:\n', conf_matrix)
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
print('F1 Score: {:.4f}'.format(f1))
print('ROC AUC: {:.4f}'.format(roc_auc))

# 绘制ROC曲线（仅适用于二分类）
if len(np.unique(y)) == 2:
    fpr, tpr, _ = metrics.roc_curve(y_true, [p[1] for p in model(
        torch.tensor(X_test, dtype=torch.float32).to(device)).detach().cpu().numpy()])
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# 使用KernelExplainer
background = X_train_tensor[:100]  # 使用一部分训练数据作为背景数据集
explainer = shap.KernelExplainer(model, background)
shap_values = explainer.shap_values(X_test_tensor[:10])  # 可能需要减少测试样本数量来降低计算量

# 可视化第一个预测的SHAP值
shap.initjs()  # 初始化JavaScript可视化环境
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test[0])

# 可视化特征的整体重要性
shap.summary_plot(shap_values, X_test, feature_names=data.columns[3:])