import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)

for column in df.select_dtypes(include='object').columns:
    df[column] = le.fit_transform(df[column])

df["Increment"] = df["PerformanceRating"].apply(lambda x: 1.10 if x == 4 else 1.05)
df["FutureSalary_PerformanceBased"] = df["MonthlyIncome"] * df["Increment"]

X_attrition = df.drop(['Attrition', 'MonthlyIncome', 'PerformanceRating', 'FutureSalary_PerformanceBased', 'Increment'], axis=1)
y_attrition = df['Attrition']

mod = RandomForestClassifier(n_estimators=100, random_state=42)
mod.fit(X_attrition, y_attrition)

P_leave = mod.predict_proba(X_attrition)
P_stay = 1 - P_leave[:, 1]

df['P_stay'] = P_stay
likely_to_stay = df[df['P_stay'] > 0.6]

X_salary = likely_to_stay.drop(['Attrition', 'MonthlyIncome', 'PerformanceRating', 'FutureSalary_PerformanceBased', 'Increment', 'P_stay'], axis=1)
y_salary = likely_to_stay['FutureSalary_PerformanceBased']

X_train, X_test, y_train, y_test = train_test_split(X_salary, y_salary, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("Number of employees that are likely to stay: ", len(likely_to_stay))
print("Their salaries: \n")
print(y_pred)
