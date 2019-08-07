import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors.kde import KernelDensity
from sklearn.mixture import GaussianMixture
import seaborn as sns

#loads the data
#I extracted month and day into a new column in csv using Google Sheets. Was easier than using Pandas.
with open('fraud_data.csv', 'rb') as csvfile:
    data = csv.reader(csvfile)
    month=[]
    year=[]
    amount=[]
    day =[]
    for row in data:
        year.append(row[0])
        month.append(row[1])
        amount.append(row[2])
        day.append(row[5])

#initializes my lists for storage and counter
x1 = []
x2 = []
x3 = []
x4 = []
count=0

##this loop will add values to my xlists: # of transactions, day of month, transaction size, and month of year
for i in range(len(year)-1):
    x2.append([int(day[i])])
    x3.append([float(amount[i])])
    x4.append([int(month[i])])
    if month[i] == month[i+1]:
        if year[i] == year[i+1]:
            count+=1
        else:
            x1.append([count])
            count = 0
    else:
        x1.append([count])
        count = 0

#reshapes my lists
x1 = np.array([x1]).reshape(-1,1)
x2 = np.array(x2).reshape(-1,1)
x3 = np.array(x3).reshape(-1,1)
x4 = np.array(x4).reshape(-1,1)

### builds density models for: the number of transactions that occur in a single month
n_samples=len(x1)
m1 = KernelDensity()
m1.fit(x1)
samples1 = m1.sample(n_samples=n_samples)
score = m1.score_samples(samples1)

### Create plots showing the distributions that you’ve created.
plt.hist(x1, alpha = 0.5, label ='true # of transactions/month')
plt.hist(samples1, alpha = 0.5, label = 'samples', color = 'yellow')
plt.legend(loc='upper right')
plt.show()

### builds density models for: the day in the month that a transaction will occur on.
n_samples=len(x2)
m2 = KernelDensity()
m2.fit(x2)
samples2 = m2.sample(n_samples=n_samples)
score = m2.score_samples(samples2)

### Create plots showing the distributions that you’ve created.
plt.hist(x2, alpha = 0.5, label ='true days of month')
plt.hist(samples2, alpha = 0.5, label = 'samples', color = 'yellow')
plt.legend(loc='upper right')
plt.show()

### builds density models for:transaction size.
n_samples=len(x3)
m3 = KernelDensity()
m3.fit(x3)
samples3 = m3.sample(n_samples=n_samples)
score = m3.score_samples(samples3)

### Create plots showing the distributions that you’ve created.
plt.hist(x3, bins =15, alpha = 0.5, label ='true transaction amounts')
plt.hist(samples3,bins=15, alpha = 0.5, label = 'samples', color = 'yellow')
plt.legend(loc='upper right')
plt.show()

### EXTENSTION: builds density models for: the month in the year that a transaction will occur on.
n_samples=len(x4)
m4 = KernelDensity()
m4.fit(x4)
samples4 = m4.sample(n_samples=n_samples)
score = m4.score_samples(samples4)

### Create plots showing the distributions that you’ve created.
plt.hist(x4, bins = 12, alpha = 0.5, label ='true months in the year')
plt.hist(samples4, bins = 12, alpha = 0.5, label = 'samples', color = 'yellow')
plt.legend(loc='upper right')
plt.show()

### Sampling from these density models, create a fictitious month of personal transactions.
transactions_in_fict_month = []
dollar = '$'
for i in range(samples1[0]):
    day_of_month = int(samples2[i])
    transaction_size = float(round(float(samples3[i]*100))/100)
    transactions_in_fict_month.append((day_of_month, dollar + `transaction_size`))
transactions_in_fict_month = sorted(transactions_in_fict_month, key = lambda x: x[0])
print 'Fictitious month in format (day of month, transaction amount)', transactions_in_fict_month


#Explain what flaws still remain in your model that a forensic accountant might be able to find
#and determine that this was a fraudulent set of transactions.

'''
1. First of all I get output samples as days of the month that are above 31 or below 1 which would be impossible to
recreate in real life.
2. The actual values in the transaction amounts might give away that this was a fraudelent set of transactions for
several reasons. Firstly, the values in the 'cents' might give it away if they are different from the usual
distribution of remainders. For example if I start creating transactions that, most of which end in .00 cents
and if this is an absolute rarity for the true cardholder then this might be spotted by a forensic accountant.
There also might not be enough ending in .95 or .99 to make the transactions believable.
3. Just because we modeled the distribution of the days of the month and # of transactions per month doesnt mean this
is a perfect representation of the regular yearly transaction habits of the cardholder. We are still missing a model
for which months of the year have more transactions than others because there most likely will be seasonal data there
(might get paid in June and spend a lot more in July for example.) I added a fourth variable
4. This is a little less likely but we can check for it: there might be seasonal patterns with yearly data. For example
it might be the case that the cardholder spends more on even numbered years because they get paid at the end of odd
years. We can easily check this because we have so much data on their spending habits and can incorporate it to our
fictious data.
5. Benford's Law is very commonly tested in fraudulent account detection. From http://www.testingbenfordslaw.com/:
"It occurs so regularly that it is even used in fraudulent accounting detection."

'''

### (Optional) How well does the data follow Benford’s law?

#adapted code from https://www.johndcook.com/blog/2011/10/19/benfords-law-and-scipy/
from math import log10, floor
from scipy.constants import codata
from sklearn.metrics import mean_squared_error

def most_significant_digit(x):
    e = floor(log10(x))
    return int(x*10**-e)

# count how many constants have each leading digit
count = [0]*10

for i in range(len(amount)):
    x = abs(float(amount[i]))
    count[most_significant_digit(x)] += 1
total = sum(count)

# expected number of each leading digit per Benford's law
benford = [total*log10(1 + 1./i) for i in range(1, 10)]

count = count[1:]

plt.bar(range(1,10),count,alpha = 0.5, tick_label = range(1,10))
plt.bar(range(1,10),benford, color = 'yellow', alpha = 0.5, label = 'expected')
plt.legend("upper right")
plt.xlabel('Observed leading Digits')
plt.title('Benfords expected vs observed leading digit: actual')
plt.show()

uniform = []
for i in range(1,10):
    uniform.append(float(total/9.0))

print 'MSE for expected digits to follow Benfords Law and observed', mean_squared_error(benford, count)
print 'MSE for observed digits and uniform distribution: anti-Benford', mean_squared_error(uniform,count)


'''
Both visually on the histogram output and using MSE we see that the actual data follows Benford's Law very well.
Now I will test my fictious generated data.
'''


# count how many constants have each leading digit
count = [0]*10

for i in range(len(samples3)):
    x = abs(float(samples3[i]))
    count[most_significant_digit(x)] += 1
total = sum(count)

# expected number of each leading digit per Benford's law
benford = [total*log10(1 + 1./i) for i in range(1, 10)]

count = count[1:]

plt.bar(range(1,10),count,alpha = 0.5, tick_label = range(1,10))
plt.bar(range(1,10),benford, color = 'yellow', alpha = 0.5, label = 'expected')
plt.legend("upper right")
plt.xlabel('Observed leading Digits')
plt.title('Benfords expected vs observed leading digit: fictitious')
plt.show()

uniform = []
for i in range(1,10):
    uniform.append(float(total/9.0))

print 'MSE for expected digits to follow Benfords Law and observed', mean_squared_error(benford, count)
print 'MSE for observed digits and uniform distribution: anti-Benford', mean_squared_error(uniform,count)

''' It seems from the histogram and from the MSE that the sampled data still follows Benford law very well
but it is significantly closer to the uniform distribution. The MSE between the fictitious data and uniform
is 35,881 while the real data has MSE of 45,102. I believe given the graphs that the forensic accountant wouldn't
catch me based on Benford's law alone given how much it still follows it.
'''
