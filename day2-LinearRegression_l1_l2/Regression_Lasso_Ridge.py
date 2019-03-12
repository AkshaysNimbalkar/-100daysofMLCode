import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("E:/DataScience_Study/3months/Data-Lit/week4-Regression/4.1_Regression_Mathematics/ISL-Ridge-Lasso-master/data/Advertising.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.head()

# DEA
independant_features = df.iloc[:,0:3]
#independant_features.iloc[:,1]

dependant_feature = df['sales']

for i in range(3):
    plt.subplot(2,2, i+1)
    sns.scatterplot(x=independant_features.iloc[:, i], y=dependant_feature)
    plt.ylabel("Sales ($K)")

plt.tight_layout()
plt.show()

# Simple Linear Model:
# Shows Reletioship between independant and dependant variable

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

Xs = df[['TV']]
y = df['sales'].values

lin_reg = LinearRegression()
mse = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)
mean_mse = np.mean(mse)

mean_mse # -10.794506106524935

# Multiple Linear Regression :
# Higher dimensions.
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

Xs = df.drop(['sales'], axis=1)
y = df['sales'].values

lin_reg = LinearRegression()
mse = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)
mean_mse = np.mean(mse)

mean_mse # -3.07294659710021

# Ridge Regression: will use to solve problem of overfitting if there is high multicollintiry prasent in dataset
# GridSearchCV automatically do cross validation

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(Xs, y)

ridge_regressor.best_params_   # {'alpha': 20}

ridge_regressor.best_score_    # -3.0726713383411437
# Results are slightly better than Multiple Regression


# Lasso Regression : If Features are not relevent lasso makes them zero :
# this is adv over Ridge
from sklearn.linear_model import Lasso

alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso = Lasso()

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5 )
lasso_regressor.fit(Xs, y)

lasso_regressor.best_params_   # {'alpha': 1}

lasso_regressor.best_score_   # -3.041405896751369

# Result of alpha and MSE are better than any other model.