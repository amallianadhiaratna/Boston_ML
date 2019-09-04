import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_excel('indo_12_1.xls', skiprows=3, skipfooter=2, na_values=['-'])
df.rename(columns={'Unnamed: 0':'Provinsi'}, inplace=True)
df=df.fillna(method='bfill', axis=1)                        #kolooom, axis bisa 0 bisa 1
df
a=df[df['Provinsi']=='Aceh']
x=[]
for i in df.columns[1:]:
    x.append(i)
print(x)
b=df[1971][0]
y=[]
for j in range(len(df['Provinsi'])):
    y_=[]
    for i in range(len(x)):
        y_.append(df[x[i]][0])
    y.append(y_)
# print(y)

df_=[]
for j in range(len(df['Provinsi'])):
    df2={'tahun':x,
    'penduduk':y[j]
    }
    df_.append(df2)
# print(df_)
df_1=[]
for i in range(len(df['Provinsi'])):
    df_1a=pd.DataFrame(df_[i])
    df_1.append(df_1a)
    # print(df_1a)
# print(df_1)
model_=[]
for i in range(len(df['Provinsi'])):
    model=linear_model.LinearRegression()
    model.fit(df_1[i][['tahun']],df_1[i]['penduduk'])
    print(df['Provinsi'][i],model.coef_, model.intercept_)
    print(df['Provinsi'][i],model.predict([[2020]]))


# plt.plot(
#     df_1[0]['tahun'],df_1[0]['penduduk'],'ro',
#     df_1[0]['tahun'],model.predict(df_1[0][['tahun']]), 'g-'    
# )
# plt.xlabel('tahun')
# plt.ylabel('penduduk')

# plt.show()
