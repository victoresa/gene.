import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix, accuracy_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib
from scipy.stats import uniform, randint
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载数据集
def load_data():
    gse_data = pd.read_csv('GSE98793.csv')
    data_set1 = pd.read_excel('data_set1.xlsx')
    data_set2 = pd.read_excel('data_set2.xlsx')
    return gse_data, data_set1, data_set2

# 合并数据集并进行预处理
def preprocess_data():
    gse_data, data_set1, _ = load_data()
    combined_data = pd.concat([gse_data, data_set1], ignore_index=True)
    combined_data = combined_data.drop_duplicates()
    combined_data.columns = combined_data.columns.astype(str)
    numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
    combined_data[numeric_cols] = combined_data[numeric_cols].astype(np.float64)
    return combined_data, numeric_cols

# 分批次迭代插补
def batched_iterative_impute(df, imputer, batch_size=500):
    imputed_df = df.copy()
    numeric_cols = imputed_df.select_dtypes(include=[np.number]).columns
    for start in range(0, len(numeric_cols), batch_size):
        end = start + batch_size
        batch_cols = numeric_cols[start:end]
        logger.info(f"正在插补列 {start} 到 {end}...")
        imputed_df[batch_cols] = imputer.fit_transform(imputed_df[batch_cols])
    return imputed_df

# 特征选择和标准化
def feature_selection_and_scaling(X, y):
    selector = SelectKBest(score_func=f_classif, k=2000)
    X_selected = selector.fit_transform(X, y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    return X_scaled, selector, scaler

# 处理数据不平衡
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# 超参数调优
def hyperparameter_tuning(X_train, y_train, models):
    best_estimators = {}
    for name, config in models.items():
        logger.info(f"正在调优 {name}...")
        clf = RandomizedSearchCV(
            config['model'],
            config['params'],
            n_iter=50,
            scoring='roc_auc',
            cv=5,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        best_estimators[name] = clf.best_estimator_
        logger.info(f"{name} 最佳AUC: {clf.best_score_:.4f}")
        logger.info(f"{name} 最佳参数: {clf.best_params_}\n")
    return best_estimators

# 定义组合模型并评估
def evaluate_combined_models(X_train, y_train, best_estimators):
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', best_estimators['LogisticRegression']),
            ('svc', best_estimators['SVC']),
            ('mlp', best_estimators['MLPClassifier'])
        ],
        voting='soft',
        n_jobs=-1
    )

    stacking_clf = StackingClassifier(
        estimators=[
            ('lr', best_estimators['LogisticRegression']),
            ('svc', best_estimators['SVC']),
            ('mlp', best_estimators['MLPClassifier'])
        ],
        final_estimator=best_estimators['SVC'],
        n_jobs=-1
    )

    model_performance = {}
    for clf in [voting_clf, stacking_clf]:
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        model_performance[clf.__class__.__name__] = scores.mean()
        logger.info(f"{clf.__class__.__name__} 交叉验证AUC: {scores.mean():.4f}")

    best_combined_model_name = max(model_performance, key=model_performance.get)
    final_model = voting_clf if best_combined_model_name == 'VotingClassifier' else stacking_clf
    logger.info(f"最佳组合模型是: {best_combined_model_name}")

    return final_model

# 训练最终模型并进行验证
def train_and_validate_model(final_model, X_train, y_train, X_val, y_val):
    final_model.fit(X_train, y_train)
    y_val_pred_proba = final_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    logger.info(f"验证集AUC: {val_auc:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % val_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # AUPR Curve
    precision, recall, _ = precision_recall_curve(y_val, y_val_pred_proba)
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label='AUPR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    # Confusion Matrix
    y_val_pred = final_model.predict(X_val)
    cm = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Feature Importance (if applicable)
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

    # Metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    specificity = recall_score(y_val, y_val_pred, pos_label=0)
    logger.info(f"Accuracy: {accuracy:.4f}")

    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")

    joblib.dump(final_model, 'best_model.joblib')

# 处理测试集并生成预测
def process_test_set():
    _, _, data_set2 = load_data()
    selector = joblib.load('feature_selector.joblib')
    imputer = joblib.load('imputer.joblib')
    scaler = joblib.load('scaler.joblib')
    final_model = joblib.load('best_model.joblib')

    selected_features = selector.get_feature_names_out()
    data_set2_features = data_set2.drop(columns=['ID'])
    common_features = set(selected_features) & set(data_set2_features.columns)
    common_features = list(common_features)
    data_set2_features = data_set2_features[common_features]

    missing_cols = set(selected_features) - set(data_set2_features.columns)
    for col in missing_cols:
        data_set2_features[col] = 0

    data_set2_features = data_set2_features[selected_features]
    data_set2_features = imputer.transform(data_set2_features)
    data_set2_scaled = scaler.transform(data_set2_features)

    predictions = final_model.predict_proba(data_set2_scaled)[:, 1]
    submission = pd.DataFrame({'ID': data_set2['ID'], 'Prediction': predictions})
    submission.to_csv('submission.csv', index=False)

# 主程序
def main():
    combined_data, numeric_cols = preprocess_data()
    imputer = IterativeImputer(random_state=42, max_iter=100, sample_posterior=False)
    combined_data[numeric_cols] = batched_iterative_impute(combined_data[numeric_cols], imputer, batch_size=1)

    X = combined_data.drop(columns=['ID', 'sample_group'])
    y = combined_data['sample_group'].map({'case': 1, 'control': 0})

    X_scaled, selector, scaler = feature_selection_and_scaling(X, y)
    X_resampled, y_resampled = balance_data(X_scaled, y)

    joblib.dump(selector, 'feature_selector.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(imputer, 'imputer.joblib')

    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    models = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=2000, solver='liblinear'),
            'params': {
                'C': np.logspace(-4, 4, 20),
                'penalty': ['l1', 'l2']
            }
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': randint(100, 1000),
                'max_depth': randint(5, 50),
                'min_samples_split': randint(2, 20)
            }
        },
        'XGBClassifier': {
            'model': XGBClassifier(eval_metric='logloss'),
            'params': {
                'n_estimators': randint(100, 1000),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3)
            }
        },
        'SVC': {
            'model': SVC(probability=True),
            'params': {
                'C': uniform(0.1, 10),
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        'GradientBoostingClassifier': {
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': randint(100, 1000),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10)
            }
        },
        'MLPClassifier': {
            'model': MLPClassifier(max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(100,), (100, 100), (50, 50, 50)],
                'activation': ['tanh', 'relu'],
                'alpha': uniform(0.0001, 0.01)
            }
        }
    }

    best_estimators = hyperparameter_tuning(X_train, y_train, models)
    final_model = evaluate_combined_models(X_train, y_train, best_estimators)
    train_and_validate_model(final_model, X_train, y_train, X_val, y_val)
    process_test_set()

if __name__ == "__main__":
    main()