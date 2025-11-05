import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)

for column in df.select_dtypes(include='object').columns:
    df[column] = le.fit_transform(df[column])

df["Increment"] = df["PerformanceRating"].apply(lambda x: 1.10 if x == 4 else 1.05)
df["FutureSalary_PerformanceBased"] = df["MonthlyIncome"] * df["Increment"]

df_stay = df[df["Attrition"] == 0]

X = df_stay.drop(["Attrition", "MonthlyIncome", "PerformanceRating", "FutureSalary_PerformanceBased"], axis=1)
y = df_stay["FutureSalary_PerformanceBased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge(alpha = 1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)

r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("R2 Score: ", r2)
print("RMSE: ", rmse)