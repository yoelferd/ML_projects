import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")



#cleaning and transforming the data:

loan_data = "LoanStats3a.csv"
reject_data = "RejectStats_updated.csv"


# length of data
num_lines = sum(1 for i in open(loan_data))
num_lines_r = 65500

#  size of our data that we want to use
size = int(num_lines / 9)
size_r = int(float(num_lines_r) / float(13.85568))

# The row indices to skip to make data smaller
skip_idx = random.sample(range(1, num_lines), num_lines - size)
skip_idx_r = random.sample(range(1, num_lines_r), num_lines_r - size_r)


# Read the data
loan_sample = pd.read_csv(loan_data, skiprows=skip_idx, usecols = ['loan_amnt', 'loan_status','total_pymnt','verification_status','dti', 'emp_length', 'funded_amnt'] )
reject_sample = pd.read_csv(reject_data, skiprows = skip_idx_r, usecols =['loan_amnt', 'Risk_Score', 'dti','emp_length', 'funded_amnt'])

#change categorical data to dummies so we can run regression
dum_loan_sample = pd.get_dummies(loan_sample,columns=['verification_status', 'loan_status','emp_length'])
dum_reject_sample = pd.get_dummies(reject_sample,columns=['emp_length'])


#independent and dependent variables

ind_var_one = dum_loan_sample[['loan_amnt','dti', 'emp_length_2 years', 'emp_length_3 years', 'emp_length_4 years', 'emp_length_5 years', 'emp_length_6 years', 'emp_length_7 years', 'emp_length_8 years', 'emp_length_9 years', 'emp_length_< 1 year']]
ind_var_two = dum_reject_sample[['loan_amnt', 'dti','emp_length_2 years', 'emp_length_3 years', 'emp_length_4 years', 'emp_length_5 years', 'emp_length_6 years', 'emp_length_7 years', 'emp_length_8 years', 'emp_length_9 years', 'emp_length_< 1 year']]

dep_var_one = dum_loan_sample[['funded_amnt']]
dep_var_two = dum_reject_sample[['funded_amnt']]

ind_var_list = [ind_var_one, ind_var_two]
dep_var_list = [dep_var_one, dep_var_two]

ind_var = pd.concat(ind_var_list)
dep_var = pd.concat(dep_var_list)

#splitting the data to train/test to prevent overfitting
X_train, X_test, y_train, y_test = train_test_split(ind_var, dep_var, test_size=0.4, random_state=0)

#fitting the model//Create linear model
reg=LinearRegression()
reg.fit(X_train,y_train)


#evaluating its performance

y_pred = reg.predict(X_test)

y_pred_2 = reg.predict([[5000,50,0,0,0,0,0,0,0,1,0]])

print 'Good Candidate for $5k should ask for' , y_pred_2

print 'R^2 score is', reg.score(X_test,y_test)
#coefficients
print('Coefficients: \n', reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
