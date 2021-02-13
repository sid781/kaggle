# %%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.validation import column_or_1d 
# %%
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
# %%
df.head()
# %%
df1 = df.drop(["Name", "Ticket", "SibSp","Parch", "Cabin", "Embarked"], axis="columns")
df1 = df1.set_index("PassengerId")
df2 = df_test.drop(["Name", "Ticket","SibSp","Parch", "Cabin", "Embarked"], axis="columns")
df2 = df2.set_index("PassengerId")
df1.head()
# %%
df1.columns[df1.isna().any()]
# %%
df1["Age"]=df1["Age"].fillna(df1["Age"].mean())
df1.head(10)

# %%
df2.columns[df2.isna().any()]
# %%
df2["Age"]=df2["Age"].fillna(df2["Age"].mean())
df2["Fare"]= df2["Fare"].fillna(method='ffill')
# %%
# ax = sns.pairplot(df1)

# %%
from sklearn.preprocessing import LabelEncoder
# %%
le = LabelEncoder()
df1["Sex"] = le.fit_transform(df1["Sex"])
df2["Sex"] = le.fit_transform(df2["Sex"])
df1.head()
# %%
cor = df1[df1.columns].corr()
sns.heatmap(cor)

# %%
X_train = df1.drop(["Survived"], axis=1)
y_train = df1["Survived"]
X_test = df2
X_train.head()
# %%
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
# %%
lr.fit(X_train, y_train)

# %%
lr.score(X_train, y_train)
# %%
y_predicted1 = lr.predict(X_test)
df2["Survived"] = y_predicted1.reshape(-1,1)
df2.head()

df2.to_csv("results.csv")
# %%
