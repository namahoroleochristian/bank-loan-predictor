import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


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

df["ApplicantIncome"] = (df["ApplicantIncome"] - df["ApplicantIncome"].mean()) / df["ApplicantIncome"].std() 
df["LoanAmount"] = (df["LoanAmount"] - df["LoanAmount"].mean() ) / df["LoanAmount"].std()


X = df.drop("Loan_Status",axis=1)
X = X.drop("Loan_ID",axis=1)


# One-hot encoding categorical features
# X = pd.get_dummies(X, drop_first=True)

#encoding string data into analysable data
encoder = LabelEncoder()
X["Education"] = encoder.fit_transform(X["Education"])
X["Married"] = encoder.fit_transform(X["Married"])
X["Property_Area"] = encoder.fit_transform(X["Property_Area"])
X["Gender"] = encoder.fit_transform(X["Gender"])
X["Self_Employed"] = encoder.fit_transform(X["Self_Employed"])
X["Dependents"] = encoder.fit_transform(X["Dependents"])
Y = encoder.fit_transform(df["Loan_Status"])
# X.to_csv("Cleaned-Loan-Approval-Prediction_Dummy_X.csv")

print(X)
print(Y)


# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state= 42)
# model = RandomForestClassifier()
# model.fit(X_train,Y_train)

# joblib.dump(model,"loan-approval-predictor.pkl")

# print(Y,"df info")    