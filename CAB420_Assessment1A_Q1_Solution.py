import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels import api as sm
from sklearn.metrics import mean_squared_error
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge


def load_question_1(root_dir=r'C:\Users\user\Downloads\CAB420_Assessment1A_Data\Data'):
    communitiesData = pd.read_csv(os.path.join(root_dir, 'Q1/communities.csv'))
    return communitiesData

data = load_question_1()

# remove index column as they are not predictive 
data = data.drop([' state ',' county ', ' community ', ' communityname string', ' fold '], axis=1)
print(data.head())

# find data that involve undefine value,for example : '?' value 
columns_to_remove = []
for column in data.columns.values:
  if np.sum(data[column] == '?' )  > 0:
    # add this column to the list that should be removed
    columns_to_remove.append(column)
print(columns_to_remove)
print(len(columns_to_remove))    

# remove those column
data = data.drop(columns_to_remove, axis=1)
print(data.shape)

# now drop any rows that contain a Nan, and deleted??
print(np.sum(data.isna(), axis=1))
print(np.sum(np.sum(data.isna(), axis=1) > 0))
nans = data.isna()
nans.to_csv('nans.csv')
data_filtered = data.dropna(axis=0)

# final dataset checking
print(data_filtered.head())
print('Final dataset shape = {}'.format(data_filtered.shape))


# split into traning,validation and testing sets
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

# create the linear model
model = sm.OLS(y_train.astype(float), x_train.astype(float))
# fit the model without any regularisation
first_model_fit = model.fit()
pred = first_model_fit.predict(x_val)
print('First Model RMSE = {}'.format(
  np.sqrt(mean_squared_error(y_val, first_model_fit.predict(x_val)))))
print(first_model_fit.summary())
print(first_model_fit.params)
fig, ax = plt.subplots(figsize=(8,6))
sm.qqplot(first_model_fit.resid, ax=ax, line='s')
plt.title('QQ Plot for Linear Regression')
plt.show()


all_variables = data.iloc[1,:]

# fit the model with L1 regularisation 
# if the L1_wt param is 1 representing L1 regularisation
# if L1_wt = 0 representing L2 regularisation
alpha = 1.0
l1_model_fit = model.fit_regularized(alpha=alpha, L1_wt=1)
pred = l1_model_fit.predict(x_val)
print('L1: alpha = {},  RMSE = {}'.format(
  alpha, np.sqrt(mean_squared_error(y_val, l1_model_fit.predict(x_val)))))

# fit the model with L2 regularisation
l2_model_fit = model.fit_regularized(alpha=alpha, L1_wt=0)
pred = l2_model_fit.predict(x_val)
print('L2: alpha = {},  RMSE = {}'.format(
  alpha, np.sqrt(mean_squared_error(y_val, l2_model_fit.predict(x_val)))))


# experimenting L1 and L2 parameters for finding best RMSE
# By making a huge number on best_rmse for being overwritten
best_rmse = 10e12
best_alpha = []
best_L1_L2 = []

# set up different ranges of alpha for L1 and L2
alpha_list = np.linspace(0.1, 5.0, 20)
# value that diterment whether we should used L1 or L2
L1_L2_list = [0, 1]

for L1_L2 in L1_L2_list:
  for alpha in alpha_list:
    model_cross_fit = model.fit_regularized(alpha=alpha, L1_wt=0)
    pred = model_cross_fit.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, model_cross_fit.predict(x_val)))
    print('L1_L2 = {},  alpha = {},  RMSE = {}'.format(L1_L2, alpha, rmse))
    # save model with lowest RMSE
    if rmse < best_rmse:
      best_rmse = rmse
      best_alpha = alpha
      best_L1_L2 = L1_L2

print('\nBest Model: L1_L2 = {}, alpha = {}, RMSE = {}'.format(
  best_L1_L2, best_alpha, best_rmse))

# create validation data
linear = LinearRegression(fit_intercept = False).fit(X = x_train.to_numpy(), y = y_train.to_numpy())
fig = plt.figure(figsize=[25, 16])
ax = fig.add_subplot(4, 1, 1)
ax.bar(range(len(linear.coef_)), linear.coef_)
ax = fig.add_subplot(4, 1, 2)
ax.plot(linear.predict(x_train), label='Predicted')
ax.plot(y_train.to_numpy(), label='Actual')
ax.set_title('Training Data')
ax.legend()
ax = fig.add_subplot(4, 1, 3)
ax.plot(linear.predict(x_val), label='Predicted')
ax.plot(y_val.to_numpy(), label='Actual')
ax.set_title('Validation Data')
ax.legend()
ax = fig.add_subplot(4, 1, 4)
ax.plot(linear.predict(x_test), label='Predicted')
ax.plot(y_test.to_numpy(), label='Actual')
ax.set_title('Testing Data')
ax.legend();

#train model with Lasso Regression
lasso_1 = Lasso(fit_intercept=False, alpha=0.01).fit(X = x_train.to_numpy(), y = y_train.to_numpy())
lasso_2 = Lasso(fit_intercept=False, alpha=0.1).fit(X = x_train.to_numpy(), y = y_train.to_numpy())
lasso_3 = Lasso(fit_intercept=False, alpha=0.5).fit(X = x_train.to_numpy(), y = y_train.to_numpy())

#plot the graph
fig = plt.figure(figsize=[25, 16])
ax = fig.add_subplot(4, 1, 1)
w = 0.2
pos = np.arange(0, len(linear.coef_), 1)
ax.bar(pos - w*2, linear.coef_, width=w, label='linear')
ax.bar(pos - w, lasso_1.coef_, width=w, label='alpha=0.01')
ax.bar(pos, lasso_2.coef_, width=w, label='alpha=0.1')
ax.bar(pos + w, lasso_3.coef_, width=w, label='alpha=0.5')
ax.legend()
ax = fig.add_subplot(4, 1, 2)
ax.plot(linear.predict(x_train), label='linear')
ax.plot(lasso_1.predict(x_train), label='alpha=0.01')
ax.plot(lasso_2.predict(x_train), label='alpha=0.1')
ax.plot(lasso_3.predict(x_train), label='alpha=0.5')
ax.plot(y_train.to_numpy(), label='Actual')
ax.set_title('Training Data')
ax.legend()
ax = fig.add_subplot(4, 1, 3)
ax.plot(linear.predict(x_val), label='linear')
ax.plot(lasso_1.predict(x_val), label='alpha=0.01')
ax.plot(lasso_2.predict(x_val), label='alpha=0.1')
ax.plot(lasso_3.predict(x_val), label='alpha=0.5')
ax.plot(y_val.to_numpy(), label='Actual')
ax.set_title('Validation Data')
ax.legend()
ax = fig.add_subplot(4, 1, 4)
ax.plot(linear.predict(x_test), label='linear')
ax.plot(lasso_1.predict(x_test), label='alpha=0.01')
ax.plot(lasso_2.predict(x_test), label='alpha=0.1')
ax.plot(lasso_3.predict(x_test), label='alpha=0.5')
ax.plot(y_test.to_numpy(), label='Actual')
ax.set_title('Testing Data')
ax.legend();


for lasso in [lasso_1,lasso_2]:
    rmse = np.sqrt(mean_squared_error(y_val, lasso.predict(x_val)))
    print('\nValudation set :{},  RMSE = {}'.format(str(lasso), rmse))
    rmse = np.sqrt(mean_squared_error(y_test, lasso.predict(x_test)))
    print('\nTesting set :{},  RMSE = {}'.format(str(lasso), rmse))


#train modle with Ridge Regression
ridge_1 = Ridge(fit_intercept=False, alpha=0.01).fit(X = x_train.to_numpy(), y = y_train.to_numpy())
ridge_2 = Ridge(fit_intercept=False, alpha=2.5).fit(X = x_train.to_numpy(), y = y_train.to_numpy())
ridge_3 = Ridge(fit_intercept=False, alpha=10).fit(X = x_train.to_numpy(), y = y_train.to_numpy())

#plot the graph
fig = plt.figure(figsize=[25, 16])
ax = fig.add_subplot(4, 1, 1)
w = 0.2
pos = np.arange(0, len(linear.coef_), 1)
ax.bar(pos - w*2, linear.coef_, width=w, label='linear')
ax.bar(pos - w, ridge_1.coef_, width=w, label='alpha=0.01')
ax.bar(pos, ridge_2.coef_, width=w, label='alpha=2.5')
ax.bar(pos + w, ridge_3.coef_, width=w, label='alpha=10')
ax.legend()
ax = fig.add_subplot(4, 1, 2)
ax.plot(linear.predict(x_train), label='linear')
ax.plot(ridge_1.predict(x_train), label='alpha=0.01')
ax.plot(ridge_2.predict(x_train), label='alpha=2.5')
ax.plot(ridge_3.predict(x_train), label='alpha=10')
ax.plot(y_train.to_numpy(), label='Actual')
ax.set_title('Training Data')
ax.legend()
ax = fig.add_subplot(4, 1, 3)
ax.plot(linear.predict(x_val), label='linear')
ax.plot(ridge_1.predict(x_val), label='alpha=0.01')
ax.plot(ridge_2.predict(x_val), label='alpha=2.5')
ax.plot(ridge_3.predict(x_val), label='alpha=10')
ax.plot(y_val.to_numpy(), label='Actual')
ax.set_title('Validation Data')
ax.legend()
ax = fig.add_subplot(4, 1, 4)
ax.plot(linear.predict(x_test), label='linear')
ax.plot(ridge_1.predict(x_test), label='alpha=0.01')
ax.plot(ridge_2.predict(x_test), label='alpha=2.5')
ax.plot(ridge_3.predict(x_test), label='alpha=10')
ax.plot(y_test.to_numpy(), label='Actual')
ax.set_title('Testing Data')
ax.legend();

for ridge in [ridge_1,ridge_2,ridge_3]:
    rmse = np.sqrt(mean_squared_error(y_val, ridge.predict(x_val)))
    print('\nValudation set :{},  RMSE = {}'.format(str(ridge), rmse))
    rmse = np.sqrt(mean_squared_error(y_test, ridge.predict(x_test)))
    print('\nTesting set :{},  RMSE = {}'.format(str(ridge), rmse))
    
