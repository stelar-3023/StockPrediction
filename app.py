import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


dataset = pd.read_csv(r'E:\MachineLearning\StockPrediction\AAPL.csv')
dataset.shape
print(dataset.describe())

dataset.plot(x='High', y='Low', style='o')  
plt.title('High vs Low')  
plt.xlabel('High')  
plt.ylabel('Low')  
plt.show()

# plot average High
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['High'])

# Divide data into attributes & Labels
X = dataset['Low'].values.reshape(-1,1)
y = dataset['High'].values.reshape(-1,1)

# Split 85% of the data to train and 15% of the data to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Train the algorithm
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)

# make the prediction 
y_pred = regressor.predict(X_test)

# compare the output actual values vs predicted values
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

# Plot straight line with the test data
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))