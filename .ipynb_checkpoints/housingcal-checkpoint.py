import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sb

df=pd.read_csv('housing.csv')
corr=df.corr()
df=df.dropna()

# print(corr)
# sb.heatmap(corr)
# plt.show()

# print(model.predict([[1000,5,1]]))

model=linear_model.LinearRegression()
model.fit(df[['households']],df['median_house_value'])
print(model.coef_, model.intercept_)
print(model.predict(df[['households']]))
plt.style.use('ggplot')
plt.subplot(2,3,1)
plt.plot(
    df['households'],df['median_house_value'],'ro',
    df['households'],model.predict(df[['households']]), 'g-'    
)
plt.xlabel('households')
plt.ylabel('median house value')

model.fit(df[['total_rooms']],df['median_house_value'])
print(model.coef_, model.intercept_)
print(model.predict(df[['total_rooms']]))
plt.style.use('ggplot')
plt.subplot(2,3,2)
plt.plot(
    df['total_rooms'],df['median_house_value'],'ro',
    df['total_rooms'],model.predict(df[['total_rooms']]), 'g-'    
)
plt.xlabel('total rooms')
plt.ylabel('median house value')

model.fit(df[['total_bedrooms']],df['median_house_value'])
print(model.coef_, model.intercept_)
print(model.predict(df[['total_bedrooms']]))
plt.style.use('ggplot')
plt.subplot(2,3,3)
plt.plot(
    df['total_bedrooms'],df['median_house_value'],'ro',
    df['total_bedrooms'],model.predict(df[['total_bedrooms']]), 'g-'    
)
plt.xlabel('total bedrooms')
plt.ylabel('median house value')

model.fit(df[['median_income']],df['median_house_value'])
print(model.coef_, model.intercept_)
print(model.predict(df[['median_income']]))
plt.style.use('ggplot')
plt.subplot(2,3,4)
plt.plot(
    df['median_income'],df['median_house_value'],'ro',
    df['median_income'],model.predict(df[['median_income']]), 'g-'    
)
plt.xlabel('median income')
plt.ylabel('median house value')

model.fit(df[['housing_median_age']],df['median_house_value'])
print(model.coef_, model.intercept_)
print(model.predict(df[['housing_median_age']]))
plt.style.use('ggplot')
plt.subplot(2,3,5)
plt.plot(
    df['housing_median_age'],df['median_house_value'],'ro',
    df['housing_median_age'],model.predict(df[['housing_median_age']]), 'g-'    
)
plt.xlabel('housing median age')
plt.ylabel('median house value')

plt.show()


# pakai matplotlib bikin heatmap
# plt.imshow(corr, cmap='hot_r')
# plt.colorbar()
# plt.show()