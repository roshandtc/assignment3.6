import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('50_Startups.csv')

# Visualize data
plt.figure(figsize=(8,6))
plt.scatter(df['R&D Spend'], df['Profit'])
plt.title('Profit vs R&D Spend')
plt.xlabel('R&D Spend') 
plt.ylabel('Profit')
plt.show()

test_sizes = [0.4]

for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(df[['R&D Spend']], df['Profit'], test_size=test_size, random_state=42)

     # Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

 # Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'RMSE for test_size {test_size}: {rmse}')

    # Plot the data
    plt.figure(figsize=(8,6))
    plt.scatter(X_test, y_test, label='Actual')
    plt.scatter(X_test, y_pred, label='Predicted')
    plt.title(f'Actual vs Predicted (test_size {test_size})')
    plt.xlabel('RDSpend')
    plt.ylabel('Profit')
    plt.legend()
    plt.show()