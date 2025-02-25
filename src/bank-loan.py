import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



df = pd.read_csv('Loan-Approval-Prediction.csv')
# print(df.head(),"head data")    
print(df.describe(),"df description")    
print(df.isnull().sum(),"sum of null values on our data set")    
categorical_data = ["Gender","Married", "Self_Employed", "Dependents"]

for col in categorical_data:
    df[col].fillna(df[col].mode()[0],inplace= True)

df["LoanAmount"].fillna(df["LoanAmount"].mean(),inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0],inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0],inplace=True)
df.drop_duplicates(inplace=True)
df.to_csv("Cleaned-Loan-Approval-Prediction.csv")

df["ApplicantIncome"] = (df["ApplicantIncome"] - df["ApplicantIncome"].mean()) / df["ApplicantIncome"].std() 
df["LoanAmount"] = (df["LoanAmount"] - df["LoanAmount"].mean() ) / df["LoanAmount"].std()

# df.drop("")
print(df["LoanAmount"],"df info")    