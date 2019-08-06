from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Ridge, LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''Temperature is cyclical, not only on a 24 hour basis but also on a yearly basis. Convert the dataset into a richer format whereby the day of the year is also captured. For example the time 20150212 1605, can be converted into (43, 965) because the 12th of February is the 43rd day of the year, and 16:05 is the 965th minute of the day.'''

years = range(2011, 2017)
files = ['CRNS0101-05-%d-CA_Yosemite_Village_12_W.txt' % y for y in years]
usecols = [1, 2, 8]
data = [np.loadtxt(f, usecols=usecols) for f in files]
data = np.vstack(data)

# converts minutes from HHmm to integers
data[:, 1] = np.floor_divide(data[:, 1], 100) * 60 + np.mod(data[:, 1], 100)

df = pd.DataFrame(data)

# converts dates from strings to date-time format
df[0] = pd.to_datetime(df[0], format = ('%Y%m%d'))

#converts dates from date-time format to integers from the year start
df['DayInt'] = df[0].dt.strftime('%j').astype(int)
df['Yr'] = df[0].dt.year

#Names columns
df.rename(columns={0:'Date'}, inplace=True)
df.rename(columns={1:'Min'}, inplace=True)
df.rename(columns={2:'Temp'}, inplace=True)


df = df[df.Temp != -9999.0] #removes bad data

'''This data covers 6 years, so split the data into a training set of the first 5 years, and a testing set of the 6th year.

'''

train_data = df[df['Yr'].isin([2011,2012,2013,2014,2015])] #train set if first 5 years
test_data = df[df['Yr'].isin([2016,2017])]  # test set is sixth year

train_data = train_data.values #converts pandas dataframe to a numpy ndarray
test_data = test_data.values   #converts pandas dataframe to a numpy ndarray

#train test split
train_minutes = np.array(train_data[:,1])
train_temperatures = np.array(train_data[:,2])
train_days = np.array(train_data[:,3])

test_minutes = np.array(test_data[:,1])
test_temperatures = np.array(test_data[:,2])
test_days = np.array(test_data[:,3])

#reshape temperature data
train_temperatures = train_temperatures.reshape(-1,1)
test_temperatures = test_temperatures.reshape(-1,1)

numbers_of_rbfs_to_test=[6] #choose number of centers here
widths_to_test = [.1] #choose values for sigma here

for n_rbf in numbers_of_rbfs_to_test:
    for sigma in widths_to_test:
        center_min = []
        for i in range(0,1399,1399/(n_rbf-1)):
            center_min.append(i)
        center_min = np.asarray((center_min)).reshape(-1,1)

        center_days = []
        for i in range(0,364,364/(n_rbf-1)):
            center_days.append(i)
        center_days = np.asarray((center_days)).reshape(-1,1)

        '''Cover each input dimension with a list of radial basis functions. This turns the pair of inputs into a much richer representation, mapping (d,t) into (Ol1(d), Ol2(t)). Experiment with different numbers of radial basis functions and different widths of the radial basis function in different dimensions.'''

        rbf_min = rbf_kernel(train_minutes.reshape(-1,1), center_min, gamma=1/sigma)
        rbf_days = rbf_kernel(train_days.reshape(-1,1), center_days, gamma=1/sigma)
        both_rbf = np.concatenate((rbf_min,rbf_days),axis=1)

        test_rbf_min = rbf_kernel(test_minutes.reshape(-1,1), center_min, gamma=1/sigma)
        test_rbf_days = rbf_kernel(test_days.reshape(-1,1), center_days, gamma=1/sigma)
        test_both_rbf = np.concatenate((test_rbf_min,test_rbf_days),axis=1)

        all_minutes = df['Min'].values.reshape(-1,1)

        minutes_rbf = rbf_kernel(all_minutes,center_min,gamma=1/sigma)

        all_days = df['DayInt'].values.reshape(-1,1)

        days_rbf = rbf_kernel(all_days,center_days,gamma=1/sigma)

        X_train = np.concatenate((minutes_rbf[:525479],days_rbf[:525479]),axis=1)
        #print X_train.shape, "xtrain shape"
        X_test = np.concatenate((minutes_rbf[525479:],days_rbf[525479:]),axis=1)

        #print "X_test shape:", X_test.shape
        #print "test temperatures shape ", test_temperatures.shape
        #print both_rbf.shape

        alpha = 0.0001

        print "Sigma =", sigma, ",number of RBFs = ", n_rbf

        '''Using this new representation, build a linear parameter model that captures both seasonal variations and daily variations.'''

        #full model
        '''Train with the full model.'''
        regr = Ridge(alpha=alpha, fit_intercept=False)
        regr.fit(X_train,train_temperatures)
        full_predict = regr.predict(X_test)
        errors = []
        for i in range(len(test_temperatures)):
                       errors.append(test_temperatures[i] - full_predict[i][0])
        errors_squared = [errors[i]**2 for i in range(len(errors))]
        mse_full_model = np.mean(errors_squared)
        print "MSE on test data with full model", mse_full_model
        print "R^2 score with full model", regr.score(X_test,test_temperatures)

        #just the minute data:

        '''Using mean squared error, quantify how your model performs on the testing data if you: Train with just the daily component of the model'''
        regr_just_minutes = Ridge(alpha=alpha, fit_intercept = False)
        regr_just_minutes.fit(minutes_rbf[:525479],train_temperatures)
        just_minutes_predict = regr_just_minutes.predict(minutes_rbf[525479:])
        errors_minutes = []
        for i in range(len(test_temperatures)):
                       errors_minutes.append(test_temperatures[i] - just_minutes_predict[i][0])
        errors_minutes_squared = [errors_minutes[i]**2 for i in range(len(errors_minutes))]
        mse_minutes_model = np.mean(errors_minutes_squared)
        print "MSE on test data with minutes only model", mse_minutes_model
        print "R^2 score minutes only model", regr_just_minutes.score(minutes_rbf[525479:],test_temperatures)

        #just the days data:
        '''Train with just the yearly component of the model'''
        regr_just_d = Ridge(alpha=alpha, fit_intercept = False)
        regr_just_d.fit(days_rbf[:525479],train_temperatures)
        just_days_predict = regr_just_d.predict(days_rbf[525479:])
        errors_days = []
        for i in range(len(test_temperatures)):
                       errors_days.append(test_temperatures[i] - just_days_predict[i][0])
        errors_days_squared = [errors_days[i]**2 for i in range(len(errors_days))]
        mse_days_model = np.mean(errors_days_squared)
        print "MSE on test data with days only model", mse_days_model
        print "R^2 score days only model", regr_just_d.score(days_rbf[525479:],test_temperatures)



days = np.concatenate((train_days, test_days))
minutes = np.concatenate((train_minutes.reshape(-1,1),test_minutes.reshape(-1,1)))
temperatures = np.concatenate((train_temperatures, test_temperatures))


'''Create two plots, one showing the time-of-day contribution, and one showing the time-of-year contribution.'''
plt.scatter(days,temperatures)
plt.xlabel("Day of year")
plt.ylabel("Temperature (degrees Celsius)")
plt.show()


'''Create two plots, one showing the time-of-day contribution, and one showing the time-of-year contribution.'''
plt.scatter(minutes,temperatures)
plt.xlabel("Minute of the day")
plt.ylabel("Temperature (degrees Celsius)")
plt.show()

'''Make a 3D plot showing temperature as a function of (day, time). Make sure to label your axes!'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(days,minutes,temperatures)
ax.set_xlabel('Day of year')
ax.set_ylabel('Minute of Day')
ax.set_zlabel('Temperature (Celsius)')
plt.show()

#rain problem:

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

years = range(2011, 2017)
files = ['CRNS0101-05-%d-CA_Yosemite_Village_12_W.txt' % y for y in years]
usecols = [1, 2, 9]
data = [np.loadtxt(f, usecols=usecols) for f in files]
data = np.vstack(data)

# converts minutes from HHmm to integers
data[:, 1] = np.floor_divide(data[:, 1], 100) * 60 + np.mod(data[:, 1], 100)

df = pd.DataFrame(data)

# converts dates from strings to date-time format
df[0] = pd.to_datetime(df[0], format = ('%Y%m%d'))

#converts dates from date-time format to integers from the year start
df['DayInt'] = df[0].dt.strftime('%j').astype(int)
df['Yr'] = df[0].dt.year

#Names columns
df.rename(columns={0:'Date'}, inplace=True)
df.rename(columns={1:'Min'}, inplace=True)
df.rename(columns={2:'Rain'}, inplace=True)

X = df['DayInt'].values
y_real = df['Rain'].values
y_class=[]
for i in range(len(y_real)):
    if y_real[i]>0:
        y_class.append(1)
    else:
        y_class.append(0)
y=y_class

'''What accuracy would a classifier get if it simply predicted no rain all the time?'''

print "Accuracy for a classifier that always predicts 'no' : ", 100*float(1-float((sum(y_class))/float(len(y_real)))), "percent"

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)

'''Using logistic regression predict the probability of rain in a given day of the year.'''
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
pred = logreg.predict(X_test)

print "R^2 score for test set:", logreg.score(X_test,y_test)

print "To reproduce the results, read all instructions and see comments to see what i did in that line of code"
