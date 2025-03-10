from math import nan

import numpy as np
import pandas as pd
import shap
from joblib import dump
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
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

# 随机森林模型配置和实例化
rf_params = {'objective': 'binary:logistic', 'base_score': None,
             'booster': None, 'callbacks': None, 'colsample_bylevel': None,
             'colsample_bynode': None, 'colsample_bytree': 0.7, 'device': None,
             'early_stopping_rounds': None, 'enable_categorical': False,
             'eval_metric': None, 'feature_types': None, 'gamma': 0.5,
             'grow_policy': None, 'importance_type': None, 'interaction_constraints': None,
             'learning_rate': 0.01, 'max_bin': None, 'max_cat_threshold': None,
             'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 3,
             'max_leaves': None, 'min_child_weight': None, 'missing': nan,
             'monotone_constraints': None, 'multi_strategy': None, 'n_estimators': 50,
             'n_jobs': None, 'num_parallel_tree': None, 'random_state': None,
             'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None,
             'scale_pos_weight': None, 'subsample': 0.5, 'tree_method': None,
             'validate_parameters': None, 'verbosity': None}
rf_model = XGBClassifier(**rf_params)
rf_model.fit(X_train, y_train)

# SHAP值计算
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer(X_test)

# 可视化整个测试集的SHAP值
plt.figure(figsize=(12,8))
shap.summary_plot(shap_values,plot_type='bar',feature_names=features.columns,max_display=20,show=False)
plt.gcf().subplots_adjust(left=0.35)  # 调整左边距来留出更多空间显示特征名称
plt.show()
plt.close()

plt.figure(figsize=(16,10))
shap.plots.waterfall(shap_values[0],max_display=20,show=False)
ax = plt.gca()
pos = ax.get_position()
# 调整左边距，其他参数保持不变
new_pos = [pos.x0 + 0.1, pos.y0, pos.width, pos.height]  # x0+0.1表示向右移动
ax.set_position(new_pos)
# 调整左边距和右边距
plt.gcf().subplots_adjust(left=0.3, right=0.95)  # 调整右边距以确保内容完整显示
plt.show()

plt.figure(figsize=(12,8))
shap.plots.beeswarm(shap_values,max_display=20,show=False)
plt.gcf().subplots_adjust(left=0.35)  # 调整左边距来留出更多空间显示特征名称
plt.show()
plt.close()

# 将SHAP值保存到Excel
shap_values_df = pd.DataFrame(shap_values.values, columns=features.columns)
shap_values_df['Instance'] = shap_values_df.index  # 添加索引作为实例标识
shap_values_df.to_excel('res/shap_values_1_xgb.xlsx', index=False)

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