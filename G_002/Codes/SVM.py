import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv") #Reading data from IBM HR Analytics

le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])  

df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)

for column in df.select_dtypes(include='object').columns:
    df[column] = le.fit_transform(df[column])

df["Increment"] = df["PerformanceRating"].apply(lambda x: 1.10 if x == 4 else 1.05)
df["FutureSalary_PerformanceBased"] = df["MonthlyIncome"] * df["Increment"]

X = df.drop('Attrition', axis=1)
y = df['Attrition']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred) # 1 is Yes 0 is No

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}') # Ideally it should be close to 1

f1 = f1_score(y_test, y_pred)
print("F1 score: ", f1)

print(df["FutureSalary_PerformanceBased"])

y_probs = model.predict_proba(X_test)[:, 1] # Predicted probabilities for positive class

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
print("AUC: ", roc_auc)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="red", lw=3)
plt.plot([0, 1], [0, 1], color="gray", linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC Curve for SVM")
plt.grid()
plt.show()


