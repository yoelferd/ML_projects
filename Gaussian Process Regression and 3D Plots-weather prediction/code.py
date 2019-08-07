%matplotlib inline
%config InlineBackend.figure_format = 'svg'
import GPy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from GPy.models import SparseGPRegression
import time

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

train_data = df[df['Yr'].isin([2011,2012,2013,2014,2015])] #train set if first 5 years
test_data = df[df['Yr'].isin([2016])]  # test set is sixth year

train_data = train_data.values #converts pandas dataframe to a numpy ndarray
test_data = test_data.values   #converts pandas dataframe to a numpy ndarray

#train test split
train_minutes = np.array(train_data[:,1])
train_temperatures = np.array(train_data[:,2]).reshape(-1,1)
train_days = np.array(train_data[:,3])

test_minutes = np.array(test_data[:,1])
test_temperatures = np.array(test_data[:,2]).reshape(-1,1)
test_days = np.array(test_data[:,3])

#compiles minutes and days into 1 dimension (minute of the year)
X_train = []
for i in range(len(train_minutes)):
	X_train.append(train_minutes[i]+(train_days[i]-1)*1440)
X_train_short = X_train[0::200]
X_train_short = np.array(X_train_short).reshape(-1,1)
X_train = np.array(X_train).reshape(-1,1)

X_test = []
for i in range(len(test_minutes)):
	X_test.append(test_minutes[i]+(test_days[i]-1)*1440)
X_test_short = X_test[0::200] #reduced set
X_test_short = np.array(X_test_short).reshape(-1,1) #reduced set
X_test = np.array(X_test).reshape(-1,1)

y_train = train_temperatures
y_test = test_temperatures

#reduced dataset
y_train_short = train_temperatures[0::200]
y_test_short = test_temperatures[0::200]

np.random.seed(101)
ndim=1
sigma = 1e-3
noise_var = 0.05

k1 = GPy.kern.RBF(ndim, lengthscale=200.)
#k2 = GPy.kern.src.periodic.Periodic(input_dim=1,variance=1,lengthscale=200,period=1440,n_freq=365,lower=None,upper=None,active_dims=1,name=None)
#from GPy.kern.src.kern import Periodic
#k3 = Periodic(input_dim=1,variance=2.0,lengthscale=100,period=1440,n_freq=366)
#(input_dim, variance, lengthscale, period, n_freq, lower, upper, active_dims, name)

#sparse GP:
n_inducing = 600
inducing = np.hstack(np.linspace(0,525600,n_inducing))[:,None]
t0=time.time()
m = GPy.models.SparseGPRegression(X_train_short,y_train_short,Z=inducing,kernel=k1)
t1=time.time()
m.likelihood.variance = noise_var
m.plot(plot_data=False)
print m
total_time = t1-t0

# with Optimized Covariance Parameters
m.inducing_inputs.fix()
m.optimize('bfgs')
m.plot(plot_data= False)
print m
