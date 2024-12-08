import numpy as np
import pandas as pd
import shap
from joblib import dump
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_auc_score, make_scorer, confusion_matrix, auc, ConfusionMatrixDisplay, \
    roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

# 加载数据
data = pd.read_excel('../data/merged_data.xlsx')

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
print("RESAMPLE")

features_resampled = np.vstack((features_scaled[target == 0], features_class_1_upsampled))
target_resampled = np.hstack((target[target == 0], target_class_1_upsampled))

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(features_resampled, target_resampled, test_size=0.2,
                                                    random_state=42, stratify=target_resampled)

# 模型和超参数配置
models_params = {
    'XGBClassifier': {
        'model': XGBClassifier(eval_metric='logloss',random_state=42),
        'params': {
            'n_estimators': [50, 100, 150, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'gamma': [0, 0.1, 0.5, 1]
        }
    },
}

# 评估结果存储
results = []

print('start')
# 进行模型训练和评估
for name, mp in models_params.items():
    auc_score = make_scorer(roc_auc_score, needs_proba=True)
    grid = GridSearchCV(mp['model'], mp['params'], cv=10, scoring=auc_score,n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # 获取模型的所有参数
    full_params = best_model.get_params()
    # 保存模型
    dump(best_model, f'../Results/NoPCA/models/{name}_best_model.joblib')

    predictions = best_model.predict(X_test)
    y_probs = best_model.predict_proba(X_test)[:,1]

    #### 计算混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()

    #### 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f'../Results/NoPCA/Figures/{name}_confusion_matrix.jpg', dpi=600)
    plt.show()
    plt.close()

    #### 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0  # 敏感性
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    f1_score_pos = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    # 计算负类的指标
    precision_neg = tn / (tn + fn) if (tn + fn) != 0 else 0
    recall_neg = tn / (tn + fp) if (tn + fp) != 0 else 0  # 特异性
    f1_score_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (
                                                                                            precision_neg + recall_neg) != 0 else 0
    # 计算F1 Macro
    f1_macro = (f1_score_pos + f1_score_neg) / 2

    # 计算ROC_AUC并绘图
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUROC={roc_auc:.2f}', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(f'../Results/NoPCA/Figures/{name}_roc_curve.jpg', dpi=600)
    plt.show()
    plt.close()

    # 保存结果
    result = {
        'Model': name,
        'PCA': 'NO',
        'Best Params': grid.best_params_,
        'All Params': full_params,  # 存储所有参数
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'F1 Score': f1_score_pos,
        'F1 Macro': f1_macro,
        'Confusion Matrix': cm,
        'AUROC': roc_auc,
    }

    print(f'model:{name}')
    print('PCA: NO')
    print(f'Acc:{accuracy}')
    print(f'Precision:{precision}')
    print(f'Recall:{recall}')
    print(f'Specificity:{specificity}')
    print(f'F1 Score (Pos):{f1_score_pos}')
    print(f'F1 Score (Neg):{f1_score_neg}')
    print(f'F1 Macro:{f1_macro}')
    print(f'AUROC:{roc_auc}')
    print(f'Confusion Matrix:{cm}')

    results.append(result)

    # SHAP 值的计算
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_test)

    shap_values.feature_names = features.columns

    # # SHAP 总结图
    # plt.figure()
    # shap.summary_plot(shap_values, plot_type='bar', max_display=20, show=False)
    # plt.gcf().subplots_adjust(left=0.3, right=0.95)  # 调整右边距以确保内容完整显示
    # plt.savefig(f'../Results/NoPCA/Figures/{name}_shap_summary_plot.jpg', dpi=600)
    # plt.show()
    # plt.close()

    # 柱状图
    plt.figure()
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.gcf().subplots_adjust(left=0.35, right=0.95)  # 调整右边距以确保内容完整显示
    plt.savefig(f'../Results/NoPCA/Figures/{name}_shap_bar_plot.jpg', dpi=600)
    plt.show()
    plt.close()

    # SHAP 蜂群图
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.gcf().subplots_adjust(left=0.3, right=0.95)  # 调整右边距以确保内容完整显示
    plt.savefig(f'../Results/NoPCA/Figures/{name}_shap_beeswarm_plot.jpg', dpi=600)
    plt.show()
    plt.close()

    # 绘制瀑布图
    for i in range(min(3, len(X_test))):  # 对前5个测试样本绘制瀑布图
        plt.figure()
        shap.plots.waterfall(shap_values[0], max_display=20, show=False)
        ax = plt.gca()
        pos = ax.get_position()
        # 调整左边距，其他参数保持不变
        new_pos = [pos.x0 + 0.1, pos.y0, pos.width, pos.height]  # x0+0.1表示向右移动
        ax.set_position(new_pos)
        # 调整左边距和右边距
        plt.gcf().subplots_adjust(left=0.3, right=0.95)  # 调整右边距以确保内容完整显示
        plt.savefig(f'../Results/NoPCA/Figures/{name}_shap_waterfall_plot_sample{i}.jpg', dpi=600)
        plt.show()
        plt.close()

# 将结果保存到Excel
results_df = pd.DataFrame(results)
results_df.to_excel(f'../Results/NoPCA/xlsx/{name}_best_model_performance.xlsx', index=False)

print("所有模型的评估结果已保存到Excel文件。")
