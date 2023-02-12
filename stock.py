# Stock price Prediction using Linear regression model
# Importance of Stock market
"""Helps companies to raise capital
Helps create personal wealth
Servers as an indicator of the state of the economy
Helps to incerase investemnt"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""import plotly.graph_objs as go
from plotly.offline import plot

from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
init_notebook_mode(connected=True)"""

tesla = pd.read_csv("C:/Users/Mohan Gola/Desktop/Study/stockprice/TSLA.csv")
print(tesla.head())

# Dataset info in our dataset we have to convert Date into int
print(tesla.info())

tesla['Date'] = pd.to_datetime(tesla['Date'])
print('Dataframe have date between {0} and {1}'.format((tesla.Date.max),(tesla.Date.min)))

print(f'Total days = {(tesla.Date.max() - tesla.Date.min()).days} days')

"""print(tesla.describe())
plt.boxplot(tesla['Open','High','Low','Close','Adj Close'])
plt.show()"""

#data1 = [tesla['Open','High','Low','Close','Adj Close']]
"""data = tesla[['Open','Low','High','Close','Adj Close']]
fig = plt.figure(figsize=(20,20))
ax = fig.add_axes([0,0,1,1])
ax.boxplot(data)
plt.show()"""

# Graph of Data and price 
plt.figure(figsize=(20,20))
plt.xlabel("Year")
plt.ylabel("Price")
plt.plot(tesla['Date'],tesla['Close'],color='Red')
plt.show()

#spliting data for Training and testing the datset

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

x = np.array(tesla.index).reshape(-1,1)
print(x)
print(x.shape)
y = tesla['Close']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=101)
print(x_train[1:12])
print(y_train[1:12])

scaler = StandardScaler().fit(x_train)

"""from sklearn.linear_model import LinearRegression

lm = LinearRegression()
model = lm.fit(x_train,y_train)
print(model)"""
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train,y_train)

y_pred = lm.predict(x_test)
print(y_pred[1:10])

plt.figure(figsize=(20,20))
plt.xlabel("Date")
plt.ylabel("Price")
plt.scatter(x_test,y_test,color='red')
plt.scatter(x_test,y_pred,color='green')
plt.show()

scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(y_train,lm.predict(x_train))}\t{r2_score(y_test, lm.predict(x_test))}
{'MSE'.ljust(10)}{mse(y_train,lm.predict(x_train))}\t{mse(y_test, lm.predict(x_test))}
'''
print(scores)



