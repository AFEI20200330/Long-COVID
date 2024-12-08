from math import nan

import numpy as np
import pandas as pd
import shap
from joblib import dump
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

# 加载数据
data = pd.read_excel('../../merged_data.xlsx')

# 分离特征和目标
features = data.drop(columns=['Target_new', 'Blood_Sample_ID', 'Cluster'])
target = data['Cluster']

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# 过采样处理类别不平衡
class_1_count = target[target == 1].count()
class_0_count = target[target == 0].count()
features_class_1_upsampled, target_class_1_upsampled = resample(
    features_scaled[target == 1],
    target[target == 1],
    replace=True,
    n_samples=class_0_count,
    random_state=42
)
features_resampled = np.vstack((features_scaled[target == 0], features_class_1_upsampled))
target_resampled = np.hstack((target[target == 0], target_class_1_upsampled))

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(features_resampled, target_resampled, test_size=0.2, random_state=42, stratify=target_resampled)


rf_params = {'activation': 'tanh', 'alpha': 0.01, 'batch_size': 'auto',
             'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False,
             'epsilon': 1e-08, 'hidden_layer_sizes': (50,),
             'learning_rate': 'adaptive', 'learning_rate_init': 0.001,
             'max_fun': 15000, 'max_iter': 200, 'momentum': 0.9,
             'n_iter_no_change': 10, 'nesterovs_momentum': True,
             'power_t': 0.5, 'random_state': None, 'shuffle': True,
             'solver': 'sgd', 'tol': 0.0001, 'validation_fraction': 0.1,
             'verbose': False, 'warm_start': False}
rf_model = MLPClassifier(**rf_params)
rf_model.fit(X_train, y_train)

# SHAP值计算
explainer = shap.KernelExplainer(rf_model.predict,shap.sample(X_train,50,random_state=42))
shap_values = explainer.shap_values(X_test)  # 计算测试集的SHAP值

# 可视化整个测试集的SHAP值
plt.figure(figsize=(12,8))
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=20, show=False)
plt.gcf().subplots_adjust(left=0.35)
plt.show()
plt.close()

plt.figure(figsize=(16,10))
shap.summary_plot(shap_values, X_test, plot_type="dot", max_display=20, show=False)
plt.gcf().subplots_adjust(left=0.35)
plt.show()
plt.close()


# 预测和评估
predictions = rf_model.predict(X_test)
cm = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = cm.ravel()

# 计算指标
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
# 计算负类的指标
precision_neg = tn / (tn + fn) if (tn + fn) != 0 else 0
recall_neg = tn / (tn + fp) if (tn + fp) != 0 else 0  # 特异性
f1_score_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (
                                                                                        precision_neg + recall_neg) != 0 else 0
# 计算F1 Macro
f1_macro = (f1_score + f1_score_neg) / 2

# 输出指标
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("F1 Score:", f1_score)

# ROC AUC
if hasattr(rf_model, 'predict_proba'):
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC:", roc_auc)

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# 绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# 保存模型
dump(rf_model, 'xgb_model.joblib')