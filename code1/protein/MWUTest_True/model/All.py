import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, \
    ExtraTreesClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

# 数据加载和预处理
data_path = '../filtered_significant_proteins.xlsx'  # 更新为实际路径
data = pd.read_excel(data_path)
X = data.drop(['Target_new', 'Blood_Sample_ID', 'Cluster'], axis=1)
y = data['Cluster']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型和参数
models_params = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
        }
    },
    'LogisticRegression_elasticnet': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['elasticnet'],
            'solver': ['saga'],
            'l1_ratio': [0.1, 0.2, 0.5, 0.7, 0.9]
        }
    },
    'LogisticRegression_l1': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1'],
            'solver': ['saga']
        }
    },
    'SVC': {
        'model': SVC(probability=True),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4]  # 只对'poly'核有效
        }
    },
    'NuSVC': {
        'model': NuSVC(probability=True),
        'params': {
            'nu': [0.01, 0.05, 0.1, 0.2],  # 调整nu值到较小的范围
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
    },
    'ExtraTreesClassifier': {
        'model': ExtraTreesClassifier(),
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
            voting='soft'),
        'params': {
            'weights': [[1, 1, 1], [1, 2, 3], [3, 2, 1], [2, 1, 1], [1, 3, 2]]
        }
    },
    'RidgeClassifier': {
        'model': RidgeClassifier(),
        'params': {
            'alpha': [0.01, 0.1, 0.5, 1.0, 10, 100]
        }
    },
    'SGDClassifier': {
    'model': SGDClassifier(),
    'params': {
        'loss': ['perceptron', 'squared_error', 'modified_huber', 'huber', 'squared_epsilon_insensitive', 'hinge', 'log_loss', 'epsilon_insensitive', 'squared_hinge'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': [0.01, 0.1, 1]  # 当使用constant或adaptive学习率时有效
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10]
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 150, 200, 250],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 30, 50, 70]
        }
    },
    'GaussianNB': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': np.logspace(-10, -8, num=50)
        }
    },
    'AdaBoostClassifier': {
        'model': AdaBoostClassifier(),
        'params': {
            'n_estimators': [50, 100, 150, 200, 300],
            'learning_rate': [0.001, 0.01, 0.1, 1.0]
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(),
        'params': {
            'n_estimators': [50, 100, 150, 200, 250],
            'learning_rate': [0.01, 0.1, 0.2, 0.5],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }
    },
    'XGBClassifier': {
        'model': XGBClassifier(eval_metric='logloss'),
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
        'model': LGBMClassifier(),
        'params': {
            'n_estimators': [50, 100, 150, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9]
        }
    },
    'CatBoostClassifier': {
        'model': CatBoostClassifier(verbose=0),
        'params': {
            'iterations': [50, 100, 150, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [3, 5, 7, 9],
            'l2_leaf_reg': [1, 3, 5, 10]
        }
    },
    'MLPClassifier': {
        'model': MLPClassifier(max_iter=1000),
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

results = []

# 网格搜索和评估
for model_name, mp in models_params.items():
    grid = GridSearchCV(mp['model'], mp['params'], cv=3, return_train_score=False)
    grid.fit(X_train_scaled, y_train)
    for i in range(len(grid.cv_results_['mean_test_score'])):
        result = {
            'model': model_name,
            'params': grid.cv_results_['params'][i],
            'accuracy': grid.cv_results_['mean_test_score'][i],
            'precision': precision_score(y_test, grid.predict(X_test_scaled), average='macro'),
            'recall': recall_score(y_test, grid.predict(X_test_scaled), average='macro'),
            'f1_score': f1_score(y_test, grid.predict(X_test_scaled), average='macro')
        }
        # ROC和AUC
        if hasattr(grid.best_estimator_, 'predict_proba'):
            probs = grid.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            result['roc_auc'] = roc_auc
            # 绘制ROC曲线
            # plt.figure(figsize=(5, 5))
            # plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
            # plt.plot([0, 1], [0, 1], 'r--')
            # plt.title('ROC Curve for %s' % model_name)
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.legend(loc='lower right')
            # plt.show()
        # 混淆矩阵
        result['confusion_matrix'] = confusion_matrix(y_test, grid.predict(X_test_scaled))
        results.append(result)

# 将结果保存到DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# 可以将结果保存到Excel
results_df.to_excel('res/model_evaluation_results2.xlsx',index=False)
