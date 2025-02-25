import pandas as pd



df = pd.read_csv('Loan-Approval-Prediction.csv')
# print(df.head(),"head data")    
# print(df.describe(),"df description")    
print(df.isnull().sum(),"sum of null values on our data set")    
categorical_data = ["Gender","Married", "Self_Employed", "Dependents"]

for col in categorical_data:
    df[col].fillna(df[col].mode()[0],inplace= True)

df["LoanAmount"].fillna(df["LoanAmount"].mean(),inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0],inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0],inplace=True)

df.drop_duplicates(inplace=True)

print(df.info(),"df info")    