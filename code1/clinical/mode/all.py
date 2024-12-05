import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, \
    roc_auc_score, make_scorer, confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.utils import resample
from joblib import dump
from xgboost_test import XGBClassifier

# 加载数据
data = pd.read_excel('../merged_data.xlsx')

# 分离特征和目标
features = data.drop(columns=['Target_new', 'Blood_Sample_ID', 'Cluster'])
target = data['Cluster']

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 应用PCA降维
pca = PCA(n_components=0.95)
features_pca = pca.fit_transform(features_scaled)
print(1)
#
# # 获取PCA的成分
# components = pca.components_
#
# # 创建ExcelWriter对象，用于保存多个sheet
# with pd.ExcelWriter('pca_weights_by_component.xlsx') as writer:
#     # 为每个主成分创建一个单独的工作表
#     for i, component in enumerate(components):
#         # 创建一个DataFrame，存储每个主成分的特征权重
#         pca_component_df = pd.DataFrame(component, index=features.columns, columns=[f'Principal Component {i + 1}'])
#
#         # 按照第二列（特征权重）降序排列
#         pca_component_df_sorted = pca_component_df.sort_values(by=f'Principal Component {i + 1}', ascending=False)
#
#         # 将排序后的主成分保存到一个独立的工作表
#         pca_component_df_sorted.to_excel(writer, sheet_name=f'PC{i + 1}')
#         # 保存解释方差比并排序
#     explained_variance_ratio_df = pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance Ratio'])
#     explained_variance_ratio_df_sorted = explained_variance_ratio_df.sort_values(by='Explained Variance Ratio', ascending=False)
#     explained_variance_ratio_df_sorted.to_excel(writer, sheet_name='Explained Variance Ratio')
#
# print("每个主成分的PCA结果已保存到独立的工作表中。")

# 过采样处理类别不平衡
class_1_count = target[target == 1].count()
class_0_count = target[target == 0].count()
features_class_1_upsampled, target_class_1_upsampled = resample(
    features_pca[target == 1],
    target[target == 1],
    replace=True,
    n_samples=class_0_count,
    random_state=42
)
print(2)

features_resampled = np.vstack((features_pca[target == 0], features_class_1_upsampled))
target_resampled = np.hstack((target[target == 0], target_class_1_upsampled))

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(features_resampled, target_resampled, test_size=0.2, random_state=42, stratify=target_resampled)

# 模型和超参数配置
models_params = {
     'LogisticRegression': {
        'model': LogisticRegression(random_state=42),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
        }
    },
    'LogisticRegression_elasticnet': {
        'model': LogisticRegression(max_iter=1000,random_state=42),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['elasticnet'],
            'solver': ['saga'],
            'l1_ratio': [0.1, 0.2, 0.5, 0.7, 0.9]
        }
    },
    'LogisticRegression_l1': {
        'model': LogisticRegression(max_iter=1000,random_state=42),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1'],
            'solver': ['saga']
        }
    },
    'SVC': {
        'model': SVC(probability=True,random_state=42),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4]  # 只对'poly'核有效
        }
    },
    'NuSVC': {
        'model': NuSVC(probability=True,random_state=42),
        'params': {
            'nu': [0.01, 0.05, 0.1, 0.2],  # 调整nu值到较小的范围
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
    },
    'ExtraTreesClassifier': {
        'model': ExtraTreesClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150, 200, 250],
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'VotingClassifier': {
        'model': VotingClassifier(estimators=[
            ('lr', LogisticRegression()),
            ('rf', RandomForestClassifier()),
            ('svc', SVC(probability=True))],
            voting='soft',random_state=42),
        'params': {
            'weights': [[1, 1, 1], [1, 2, 3], [3, 2, 1], [2, 1, 1], [1, 3, 2]]
        }
    },
    'RidgeClassifier': {
        'model': RidgeClassifier(random_state=42),
        'params': {
            'alpha': [0.01, 0.1, 0.5, 1.0, 10, 100]
        }
    },
    'SGDClassifier': {
    'model': SGDClassifier(random_state=42),
    'params': {
        'loss': ['perceptron', 'squared_error', 'modified_huber', 'huber', 'squared_epsilon_insensitive', 'hinge', 'log_loss', 'epsilon_insensitive', 'squared_hinge'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': [0.01, 0.1, 1]  # 当使用constant或adaptive学习率时有效
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10]
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150, 200, 250],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(random_state=42),
        'params': {
            'n_neighbors': [3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 30, 50, 70]
        }
    },
    'GaussianNB': {
        'model': GaussianNB(random_state=42),
        'params': {
            'var_smoothing': np.logspace(-10, -8, num=50)
        }
    },
    'AdaBoostClassifier': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150, 200, 300],
            'learning_rate': [0.001, 0.01, 0.1, 1.0]
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150, 200, 250],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
        }
    },
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
    'LGBMClassifier': {
        'model': LGBMClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.5, 0.7, 0.9],
        }
    },
    'CatBoostClassifier': {
        'model': CatBoostClassifier(verbose=0,random_state=42),
        'params': {
            'iterations': [50, 100, 150, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [3, 5, 7, 9],
            'l2_leaf_reg': [1, 3, 5, 10]
        }
    },
    'MLPClassifier': {
        'model': MLPClassifier(max_iter=1000,random_state=42),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100), (100, 100, 100)],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.001, 0.01]
        }
    }
}

# 评估结果存储
results = []

print(3)
# 进行模型训练和评估
for name, mp in models_params.items():
    precision_scorer = make_scorer(precision_score, pos_label=1)
    grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring=precision_scorer)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # 获取模型的所有参数
    full_params = best_model.get_params()
    # 保存模型
    dump(best_model, f'{name}_best_model.joblib')

    predictions = best_model.predict(X_test)
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    # 计算指标
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

    result = {
        'Model': name,
        'Best Params': grid.best_params_,
        'All Params': full_params,  # 存储所有参数
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'F1 Score': f1_score_pos,
        'F1 Macro': f1_macro,
        'Confusion Matrix': cm
    }
    # 检查模型是否支持概率输出
    if hasattr(best_model, 'predict_proba'):
        y_proba = best_model.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) == 2:
            result['ROC AUC'] = roc_auc_score(y_test, y_proba)
        else:
            result['ROC AUC'] = 'N/A'
    else:
        result['ROC AUC'] = 'N/A'  # 不适用或不可计算

    results.append(result)
    print(name)

# 将结果保存到Excel
results_df = pd.DataFrame(results)
results_df.to_excel('his_2/model_performance_new.xlsx', index=False)

print("所有模型的评估结果已保存到Excel文件。")
