import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])  # 1 = Yes, 0 = No

df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)

for column in df.select_dtypes(include='object').columns:
    df[column] = le.fit_transform(df[column])

df["Increment"] = df["PerformanceRating"].apply(lambda x: 1.10 if x == 4 else 1.05)
df["FutureSalary_PerformanceBased"] = df["MonthlyIncome"] * df["Increment"]

df_stay = df[df["Attrition"] == 0]
X_salary = df_stay.drop(["Attrition", "MonthlyIncome", "PerformanceRating", "FutureSalary_PerformanceBased"], axis=1)
y_salary = df_stay["FutureSalary_PerformanceBased"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_salary, y_salary, test_size=0.2, random_state=42)

salary_model = RandomForestRegressor(n_estimators=100, random_state=42)
salary_model.fit(X_train_s, y_train_s)

X_all_salary = df.drop(["Attrition", "MonthlyIncome", "PerformanceRating", "FutureSalary_PerformanceBased"], axis=1)
df["Predicted_Future_Salary"] = salary_model.predict(X_all_salary)

X_att = df.drop(["Attrition", "MonthlyIncome", "PerformanceRating", "FutureSalary_PerformanceBased", "Predicted_Future_Salary"], axis=1)
y_att = df["Attrition"]

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_att, y_att, test_size=0.2, random_state=42)

att_model = RandomForestClassifier(n_estimators=100, random_state=42)
att_model.fit(X_train_a, y_train_a)

df["Attrition_Prob"] = att_model.predict_proba(X_att)[:, 1]

# Expected loss
df["Expected_Loss"] = df["Attrition_Prob"] * df["Predicted_Future_Salary"]

# Total expected loss
total_expected_loss = df["Expected_Loss"].sum()

print(df["Expected_Loss"])
print("Total Expected Loss across all employees: ", total_expected_loss)
