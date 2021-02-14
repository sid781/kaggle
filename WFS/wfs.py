# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# %%
df = pd.read_excel("Waite First Securities.xls", sheet_name="Returns")
df1 = df.drop(["No.", "Yr.", "Mo."], axis=1)
df1
# %%
df1["mrp"]= df1["S&P 500"]-df1["Risk Free"]
df1["arp"]= df1["Apple Computer"]-df1["Risk Free"]
df1["irp"]= df1["Intel Corp."]-df1["Risk Free"]
df1["srp"]= df1["Safeway"]-df1["Risk Free"]
df1
# %%
y_cols= ["arp", "irp", "srp"]
# %%
X=df1["mrp"]
X.head()
# %%
fig, ax = plt.subplots(3,1, figsize=(20,40))
for i, premium in enumerate(y_cols):
    y=df1[premium]
    X1=sm.add_constant(X)
    model= sm.OLS(y,X1)
    result= model.fit()
    sns.regplot(x=X, y=y, ax=ax[i])
    print(result.summary())
    print("\n\n\n")
# %%

# %%
