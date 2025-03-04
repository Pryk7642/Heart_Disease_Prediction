import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# โหลดข้อมูล
df = pd.read_csv("Heart_Disease_Prediction.csv")

# แปลงค่าของ "Heart Disease"
df["Heart Disease"] = df["Heart Disease"].map({"Absence": 0, "Presence": 1})

# แยก Features และ Target
X = df.drop(columns=["Heart Disease"])
y = df["Heart Disease"]

# แบ่งข้อมูลเป็น Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# สร้างโมเดล
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_knn = KNeighborsClassifier(n_neighbors=5)
model_dt = DecisionTreeClassifier(random_state=42)
model_gb = GradientBoostingClassifier(random_state=42)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_ada = AdaBoostClassifier(n_estimators=50, random_state=42)

# รวมโมเดลด้วย Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('LoR', model_lr),
        ('kNN', model_knn),
        ('DT', model_dt),
        ('GB', model_gb),
        ('RF', model_rf),
        ('AdaBoost', model_ada)
    ],
    voting='hard'
)

# Train โมเดล
voting_clf.fit(X_train, y_train)

# ทดสอบโมเดล
y_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# บันทึกโมเดล
joblib.dump(voting_clf, "heart_disease_model.pkl")
print("Model saved successfully!")
