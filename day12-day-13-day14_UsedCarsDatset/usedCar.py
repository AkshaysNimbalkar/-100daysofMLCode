import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("E:/DataScience_Study/Aegis/Term-1/Project-Sessions/Bhavik Gandhi_ML/Ass-2/UserCarsDataset.csv", encoding='latin-1', low_memory=False)
df.head()

df.describe()

df.dtypes

df.info() # shows Not null Values

#descriptive statistics summary : First thing First Analyse dependant variable:
df['price'].describe()

corrmat = df.corr()
print(corrmat)

f, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(corrmat, cbar=True, annot=True, annot_kws={'size': 10}, square=True, cmap="YlGnBu", fmt='.2f')

total = df.isnull().sum().sort_values(ascending =False) # shows null or missing values
total

perc = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
perc

missing_data = pd.concat([total, perc], axis=1, keys=('Total', 'percent'))
missing_data.head(20)

cols = ['lastSeen', 'dateCreated', 'monthOfRegistration', 'abtest', 'name', 'dateCrawled', 'offerType', 'postalCode', 'nrOfPictures']
df1 = df.drop(cols, axis=1)
df1.head()

df1.mode()
df1.fillna(method='ffill')

df1.seller.unique()
df1.seller.mode()
df1['seller'] = df1['seller'].replace(np.nan, 'privat')
df1.isnull().sum()

df1.model.unique()
df1['model'].mode()
df1['model'] = df1['model'].replace(np.nan, 'golf')
df1.isnull().sum()

df1.fuelType.unique()
df1['fuelType'].mode()
df1['fuelType'] = df1['fuelType'].replace(np.nan, 'benzin')
df1.isnull().sum()

df1.brand.unique()
df1['brand'].mode()
df1['brand'] = df1['brand'].replace(np.nan, 'volkswagen')
df1.isnull().sum()

df1.notRepairedDamage.unique()
df1['notRepairedDamage'].mode()
df1['notRepairedDamage'] = df1['notRepairedDamage'].replace(np.nan, 'nein')
df1.isnull().sum()

df1.vehicleType.unique()
df1['vehicleType'].mode()
df1['vehicleType'] = df1['vehicleType'].replace(np.nan, 'limousine')
df1.isnull().sum()

df1.yearOfRegistration.unique()
df1['yearOfRegistration'].mode()
df1['yearOfRegistration'] = df1['yearOfRegistration'].replace(np.nan, '2000.0')
df1.isnull().sum()

df1.gearbox.unique()
df1['gearbox'].mode()
df1['gearbox'] = df1['gearbox'].replace(np.nan, 'manuell')
df1['gearbox'] = df1['gearbox'].replace('25-03-16 0:00', 'manuell')
df1.isnull().sum()


df1['price'].unique()
df1['price'].median()
df1['price'] = df1['price'].replace(np.nan, '2950.0')
df1.isnull().sum()

df1['powerPS'].unique()
df1['powerPS'].median()
df1['powerPS'] = df1['powerPS'].replace(np.nan, '105.0')
df1.isnull().sum()

df1['kilometer'].unique()
df1['kilometer'] = df1['kilometer'].replace('30-03-16 0:44', '105.0')

df1['kilometer'].median()
df1['kilometer'] = pd.to_numeric(df1['kilometer'])
df1['kilometer'] = df1['kilometer'].replace(np.nan, '105.0')
df1.isnull().sum()

## Outliers

df1['price'] = pd.to_numeric(df1['price'])
df1 = df1[(df1.price > 500) & (df1.price < 200000) ]
len(df1)


df1['yearOfRegistration'] = pd.to_numeric(df1['yearOfRegistration'])
q1 = df1['yearOfRegistration'].quantile(0.25)
q3 = df1['yearOfRegistration'].quantile(0.75)
iqr = q3 - q1

df1 = df1[((df1.yearOfRegistration) > (q1 - (1.5*iqr))) & ((df1.yearOfRegistration) < (q3 + (1.5*iqr)))]
len(df1)

df1['powerPS'] = pd.to_numeric(df1['powerPS'])
q1 = df1['powerPS'].quantile(0.25)
q3 = df1['powerPS'].quantile(0.75)
iqr = q3 - q1
df1 = df1[((df1.powerPS) > (q1 - (1.5*iqr))) & ((df1.powerPS) < (q3 + (1.5*iqr)))]
len(df1)

df1['kilometer'] = pd.to_numeric(df1['kilometer'])

#
sns.scatterplot(x='kilometer', y='price', data=df1)
sns.scatterplot(x='powerPS', y='price', data=df1)
sns.scatterplot(x='yearOfRegistration', y='price', data=df1)
sns.scatterplot(x='powerPS', y='kilometer', data=df1)


sns.boxplot('price', data=df1)
sns.boxplot('yearOfRegistration', data=df1)
sns.boxplot('powerPS', data=df1)

sns.pairplot(df1)

#Price corelation Matrix

f, ax = plt.subplots(figsize=(10, 5))
#sns.heatmap(df1.corr(), cbar=True, annot=True, annot_kws={'size': 10}, square=True, cmap="YlGnBu", fmt='.2f')
sns.heatmap(df1.corr(), cbar=True, annot=True, annot_kws={'size': 10}, square=True, cmap="viridis", fmt='.2f')

# Before Applying linear Regression
sns.regplot(x=df1['yearOfRegistration'], y=df1['price'], data=df1, scatter=True, fit_reg=True)


# Linear Regression

# the linear regression model expects a 2d array, so we add an extra dimension with reshape
# input : [1 2, 3] output : [[1], [2], [3]]
# this allows us to regress multiple variable later
# OR we can do like this: train_data = train[['yearofregistration']]

X = np.array(df1['yearOfRegistration']).reshape(-1, 1)
y = df1['price']

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predict_metrics(model, x, y):
    pred = model.predict(x)
    mse = mean_squared_error(y, pred)
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    return mse, mae, r2

from sklearn.linear_model import LinearRegression

linR = LinearRegression(normalize=True).fit(X_train, y_train)

# calculate error matrix:
mse_train, mae_train, r2_train = predict_metrics(linR, X_train, y_train)
print("train mse:", mse_train," train mae:", mae_train," R2-train", r2_train)

mse_test, mae_test, r2_test = predict_metrics(linR, X_test, y_test)
print("test mse:", mse_test," test mae:", mae_test," R2-test", r2_test)


############################### Multiple Linear regression ####################

df1.dtypes
df1 = pd.get_dummies(df1)
df1.shape

X = df1.drop('price', axis=1)
y = df1['price']


# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

linR = LinearRegression(normalize=True).fit(X_train, y_train)

# calculate error matrix:
mse_train, mae_train, r2_train = predict_metrics(linR, X_train, y_train)
print("train mse:", mse_train," train mae:", mae_train," R2-train", r2_train) # R2-train 0.7749783913024082

mse_test, mae_test, r2_test = predict_metrics(linR, X_test, y_test)
print("test mse:", mse_test," test mae:", mae_test," R2-test", r2_test)     # R2-test 0.7551244196922238


#################### Ridge Regression #######################

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

alpha = [1e-3,1e-2, 1, 5, 10, 20]
ridge = Ridge()
parameters = {'alpha': [1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X_train, y_train)

ridge_regressor.best_score_
ridge_regressor.best_params_
ridge_regressor.best_estimator_

mse_train, mae_train, r2_train = predict_metrics(ridge_regressor, X_train, y_train)
print("train mse:", mse_train," train mae:", mae_train," R2-train", r2_train)  # R2-train 0.7749165645653104

mse_test, mae_test, r2_test = predict_metrics(ridge_regressor, X_test, y_test)
print("test mse:", mse_test," test mae:", mae_test," R2-test", r2_test)        # R2-test 0.7551423652471105

#######

from scipy.stats import norm
sns.distplot(df1['price'], fit=norm)
mu, sigma = norm.fit(df1['price']) # for grtting mean and Std, deviation
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
print("Skewness: %f" % df1['price'].skew())

from scipy import stats
stats.probplot(df1['price'], plot=plt)

df1['price'] = np.log(df1['price'])

sns.distplot(df1['price'], fit=norm)
mu, sigma = norm.fit(df1['price']) # for grtting mean and Std, deviation
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
print("Skewness: %f" % df1['price'].skew())

stats.probplot(df1['price'], plot=plt)


############## After applying log transformations on price :

df1.head()

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge


alpha = [1e-3,1e-2, 1, 5, 10, 20]
ridge = Ridge()
parameters = {'alpha': [1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X_train, y_train)

ridge_regressor.best_score_
ridge_regressor.best_params_
ridge_regressor.best_estimator_

mse_train, mae_train, r2_train = predict_metrics(ridge_regressor, X_train, y_train)
print("train mse:", mse_train," train mae:", mae_train," R2-train", r2_train)  # R2-train 0.847936072111371

mse_test, mae_test, r2_test = predict_metrics(ridge_regressor, X_test, y_test)
print("test mse:", mse_test," test mae:", mae_test," R2-test", r2_test)        # R2-test 0.8477047821955875

# So, We can Clearly see that by using Ridge we got around 85 accurasy.
