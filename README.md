# -_-
갑상선암에 대해 만듦

# Import
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# Data Load
train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/open/train.csv')
test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/open/test.csv')

# Data Preprocessing
# Data Preprocessing (빠르게)
X = train.drop(columns=['ID', 'Cancer'])
y = train['Cancer']

x_test = test.drop(columns=['ID'])

# 범주형 컬럼 추출
categorical_features = X.select_dtypes(include=['object']).columns

for col in categorical_features:
    le = LabelEncoder()
    combined = pd.concat([X[col], x_test[col]]).astype(str)  # 합쳐서 fit
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    x_test[col] = le.transform(x_test[col].astype(str))

# Train  
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
def train_and_eval(X_tr, y_tr, X_val, y_val, label):
    model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    print(f"[{label}] Validation F1-score: {f1:.4f}")
    return model, f1

# (1) SMOTE 미적용 학습
model_raw, f1_raw = train_and_eval(X_train, y_train, X_val, y_val, "RAW")

# (2) SMOTE 적용 학습
smote = SMOTE(random_state=42)  # n_jobs 제거
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
model_smote, f1_smote = train_and_eval(X_train_smote, y_train_smote, X_val, y_val, "SMOTE")

# 최종 학습 데이터 결정
if f1_smote >= f1_raw:
    print("✅ SMOTE 데이터 사용")
    X_final, y_final = smote.fit_resample(X, y)
else:
    print("✅ 원본 데이터 사용")
    X_final, y_final = X, y

# 최종 모델 학습
final_model = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    use_label_encoder=False,
    eval_metric='logloss'
)
final_model.fit(X_final, y_final)



# Predict
final_pred = final_model.predict(x_test)

# Submission
submission = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/open/sample_submission.csv')
submission['Cancer'] = final_pred
submission.to_csv('baseline_submit.csv', index=False)
from google.colab import files
files.download('baseline_submit.csv')
