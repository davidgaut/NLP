
import os 
import pandas as pd 

# %% Load Data
path_files = "/home/davidg/Codes/R_Codes/Projects/BoE/ARO/Gdoor"

df = pd.read_csv(os.path.join(path_files,'allreviews.csv'),encoding="ISO-8859-1")
df['date_review']= pd.to_datetime(df['date_review'])
df = df.set_index(keys='date_review').sort_index()
df = df[~pd.isnull(df.index)]


df.info()
df.describe()


df.firm.unique()


# %% Series
df = pd.read_csv("/home/davidg/Downloads/price", index_col=0).set_index("Date")
df.dropna(inplace=True)

df.plot()
