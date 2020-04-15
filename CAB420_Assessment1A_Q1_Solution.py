import numpy as np
# pandas handles dataframes (exactly the same as tables in Matlab)
import pandas as pd
# matplotlib emulates Matlabs plotting functionality
import matplotlib.pyplot as plt
# stats models is a package that is going to perform the regression analysis
from statsmodels import api as sm
from sklearn.metrics import mean_squared_error
# os allows us to manipulate variables on out local machine, such as paths and environment variables
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge


def load_question_1(root_dir=r'C:\Users\user\Downloads\CAB420_Assessment1A_Data\Data'):
    communitiesData = pd.read_csv(os.path.join(root_dir, 'Q1/communities.csv'))
    return communitiesData

data = load_question_1()

# remove 2 to 4 column as they are not predictive 
data = data.drop([' county ', ' community ', ' communityname string', ' fold '], axis=1)
print(data.head())


# find data that enought 
threshold = 300
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

# now drop any rows that contain a Nan, and deleted
print(np.sum(data.isna(), axis=1))
print(np.sum(np.sum(data.isna(), axis=1) > 0))
nans = data.isna()
nans.to_csv('nans.csv')
data_filtered = data.dropna(axis=0)
# final dataset checking
print(data_filtered.head())
print('Final dataset shape = {}'.format(data_filtered.shape))
print(data.iloc[17, :])

# split into traning,validation and testing sets
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

# create the linear model
model = sm.OLS(y_train.astype(float), X_train.astype(float))
# fit the model without any regularisation
model_1_fit = model.fit()
pred = model_1_fit.predict(X_val)
print('Model 1 RMSE = {}'.format(
  np.sqrt(mean_squared_error(y_val, model_1_fit.predict(X_val)))))
print(model_1_fit.summary())
print(model_1_fit.params)
fig, ax = plt.subplots(figsize=(8,6))
sm.qqplot(model_1_fit.resid, ax=ax, line='s')
plt.title('Q-Q Plot for Linear Regression')
plt.show()

# lets see if any variables aren't explicitly correlated with our response variable. 
all_variables = data.iloc[1,:]
# train = X_train.merge(y_train.to_frame(), left_index=True, right_index=True)
# corr_coeffs = train[all_variables].corr()
# plt.figure()
# plt.matshow(corr_coeffs)
# plt.colorbar()
# print(np.array(corr_coeffs))


# fit the model with L1 regularisation (Lasso Regression)
# have a look at the statsmodel documentation for info on how we specify
# the regularisation parameters
# basically if the L1_wt param is 1, will be Lasso
# if L1_wt = 0, will be Ridge
alpha = 1.0
model_l1_fit = model.fit_regularized(alpha=alpha, L1_wt=1)
pred = model_l1_fit.predict(X_val)
print('L1: alpha = {},  RMSE = {}'.format(
  alpha, np.sqrt(mean_squared_error(y_val, model_l1_fit.predict(X_val)))))

# now lets try L2
model_l2_fit = model.fit_regularized(alpha=alpha, L1_wt=0)
pred = model_l2_fit.predict(X_val)
print('L2: alpha = {},  RMSE = {}'.format(
  alpha, np.sqrt(mean_squared_error(y_val, model_l2_fit.predict(X_val)))))


# try different L1 and L2 params
# making a variable that will store the best RMSE
# making it super large so will definitely be overwritten
best_rmse = 10e12
best_alpha = []
best_L1_L2 = []

# lets try a bunch of different ranges of alpha for L1 and L2
alpha_list = np.linspace(0.1, 5.0, 20)
# list to say whether we used L1 or L2
L1_L2_list = [0, 1]

for L1_L2 in L1_L2_list:
  for alpha in alpha_list:
    model_cross_fit = model.fit_regularized(alpha=alpha, L1_wt=0)
    pred = model_cross_fit.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, model_cross_fit.predict(X_val)))
    print('L1_L2 = {},  alpha = {},  RMSE = {}'.format(L1_L2, alpha, rmse))
    # if this is the model with the lowest RMSE, lets save it
    if rmse < best_rmse:
      best_rmse = rmse
      best_alpha = alpha
      best_L1_L2 = L1_L2

print('\nBest Model: L1_L2 = {}, alpha = {}, RMSE = {}'.format(
  best_L1_L2, best_alpha, best_rmse))

# also creating validation data
linear = LinearRegression(fit_intercept = False).fit(X = X_train.to_numpy(), y = y_train.to_numpy())
fig = plt.figure(figsize=[25, 16])
ax = fig.add_subplot(4, 1, 1)
ax.bar(range(len(linear.coef_)), linear.coef_)
ax = fig.add_subplot(4, 1, 2)
ax.plot(linear.predict(X_train), label='Predicted')
ax.plot(y_train.to_numpy(), label='Actual')
ax.set_title('Training Data')
ax.legend()
ax = fig.add_subplot(4, 1, 3)
ax.plot(linear.predict(X_val), label='Predicted')
ax.plot(y_val.to_numpy(), label='Actual')
ax.set_title('Validation Data')
ax.legend()
ax = fig.add_subplot(4, 1, 4)
ax.plot(linear.predict(X_test), label='Predicted')
ax.plot(y_test.to_numpy(), label='Actual')
ax.set_title('Testing Data')
ax.legend();


#Lasso Regression
lasso_model_1 = Lasso(fit_intercept=False, alpha=0.01).fit(X = X_train.to_numpy(), y = y_train.to_numpy())
lasso_model_2 = Lasso(fit_intercept=False, alpha=0.1).fit(X = X_train.to_numpy(), y = y_train.to_numpy())
lasso_model_3 = Lasso(fit_intercept=False, alpha=0.5).fit(X = X_train.to_numpy(), y = y_train.to_numpy())

fig = plt.figure(figsize=[25, 16])
ax = fig.add_subplot(4, 1, 1)
w = 0.2
pos = np.arange(0, len(linear.coef_), 1)
ax.bar(pos - w*2, linear.coef_, width=w, label='linear')
ax.bar(pos - w, lasso_model_1.coef_, width=w, label='alpha=0.01')
ax.bar(pos, lasso_model_2.coef_, width=w, label='alpha=0.1')
ax.bar(pos + w, lasso_model_3.coef_, width=w, label='alpha=0.5')
ax.legend()
ax = fig.add_subplot(4, 1, 2)
ax.plot(linear.predict(X_train), label='linear')
ax.plot(lasso_model_1.predict(X_train), label='alpha=0.01')
ax.plot(lasso_model_2.predict(X_train), label='alpha=0.1')
ax.plot(lasso_model_3.predict(X_train), label='alpha=0.5')
ax.plot(y_train.to_numpy(), label='Actual')
ax.set_title('Training Data')
ax.legend()
ax = fig.add_subplot(4, 1, 3)
ax.plot(linear.predict(X_val), label='linear')
ax.plot(lasso_model_1.predict(X_val), label='alpha=0.01')
ax.plot(lasso_model_2.predict(X_val), label='alpha=0.1')
ax.plot(lasso_model_3.predict(X_val), label='alpha=0.5')
ax.plot(y_val.to_numpy(), label='Actual')
ax.set_title('Validation Data')
ax.legend()
ax = fig.add_subplot(4, 1, 4)
ax.plot(linear.predict(X_test), label='linear')
ax.plot(lasso_model_1.predict(X_test), label='alpha=0.01')
ax.plot(lasso_model_2.predict(X_test), label='alpha=0.1')
ax.plot(lasso_model_3.predict(X_test), label='alpha=0.5')
ax.plot(y_test.to_numpy(), label='Actual')
ax.set_title('Testing Data')
ax.legend();


#Ridge Regression


glenn_1 = Ridge(fit_intercept=False, alpha=0.01).fit(X = X_train.to_numpy(), y = y_train.to_numpy())
glenn_2 = Ridge(fit_intercept=False, alpha=2.5).fit(X = X_train.to_numpy(), y = y_train.to_numpy())
glenn_3 = Ridge(fit_intercept=False, alpha=10).fit(X = X_train.to_numpy(), y = y_train.to_numpy())

fig = plt.figure(figsize=[25, 16])
ax = fig.add_subplot(4, 1, 1)
w = 0.2
pos = np.arange(0, len(linear.coef_), 1)
ax.bar(pos - w*2, linear.coef_, width=w, label='linear')
ax.bar(pos - w, glenn_1.coef_, width=w, label='alpha=0.01')
ax.bar(pos, glenn_2.coef_, width=w, label='alpha=2.5')
ax.bar(pos + w, glenn_3.coef_, width=w, label='alpha=10')
ax.legend()
ax = fig.add_subplot(4, 1, 2)
ax.plot(linear.predict(X_train), label='linear')
ax.plot(glenn_1.predict(X_train), label='alpha=0.01')
ax.plot(glenn_2.predict(X_train), label='alpha=2.5')
ax.plot(glenn_3.predict(X_train), label='alpha=10')
ax.plot(y_train.to_numpy(), label='Actual')
ax.set_title('Training Data')
ax.legend()
ax = fig.add_subplot(4, 1, 3)
ax.plot(linear.predict(X_val), label='linear')
ax.plot(glenn_1.predict(X_val), label='alpha=0.01')
ax.plot(glenn_2.predict(X_val), label='alpha=2.5')
ax.plot(glenn_3.predict(X_val), label='alpha=10')
ax.plot(y_val.to_numpy(), label='Actual')
ax.set_title('Validation Data')
ax.legend()
ax = fig.add_subplot(4, 1, 4)
ax.plot(linear.predict(X_test), label='linear')
ax.plot(glenn_1.predict(X_test), label='alpha=0.01')
ax.plot(glenn_2.predict(X_test), label='alpha=2.5')
ax.plot(glenn_3.predict(X_test), label='alpha=10')
ax.plot(y_test.to_numpy(), label='Actual')
ax.set_title('Testing Data')
ax.legend();