import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data={
    'luas':np.arange(100,300,20),
    'harga':[500,665,720,795,885,1200,1500,1600,1775,2000]
}
df=pd.DataFrame(data)
# print(df)

# plt.scatter(df['luas'],df['harga'])
# plt.show()

#bikin model ML metode linear regression
model=linear_model.LinearRegression()

#training model dg data yg kita punya
#tergantung metode dan data yg bagus
# model.fit(data independent, data dependent)
model.fit(df[['luas']], df['harga'])
m=model.coef_
c=model.intercept_
print(m,c)

# prediksi
print(model.predict([[100]]))
print(model.predict([[3000]]))

#plot data asli + best fit line
# print(model.predict([['luas']]))
plt.style.use('ggplot')
plt.plot(
    df['luas'],df['harga'],'ro',
    df['luas'],model.predict(df[['luas']]), 'g-'    
)
plt.grid(True)
plt.xlabel('Luas m2')
plt.ylabel('Harga (Rp 100 juta)')
plt.legend(['Data', 'Best Fit line'])
plt.show()