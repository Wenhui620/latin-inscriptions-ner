import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import joblib
import os

train_data = pd.read_csv('data_for_training/train.csv').fillna("")
test_data = pd.read_csv('data_for_training/test.csv').fillna("")

# extract features
def extract_features(df):
    return df[['word', 'prev_word', 'next_word', 'prefix', 'suffix']].to_dict(orient="records")

train_features = extract_features(train_data)
test_features = extract_features(test_data)

# vectorization
vectorizer = DictVectorizer(sparse=False)
X_train = vectorizer.fit_transform(train_features)
X_test = vectorizer.transform(test_features)

# tag encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['BIO'])
y_test = label_encoder.transform(test_data['BIO'])

# training
base_model = LinearSVC(class_weight='balanced', max_iter=5000, verbose=0)
svm_model = CalibratedClassifierCV(base_model)

svm_model.fit(X_train, y_train)

# evaluation
y_pred = svm_model.predict(X_test)

# report
unique_labels = set(y_test) | set(y_pred)
present_labels = [
    label for label in label_encoder.classes_
    if label_encoder.transform([label])[0] in unique_labels
]

print("Model Evaluation Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=present_labels,
    zero_division=0,
    digits=4
))





