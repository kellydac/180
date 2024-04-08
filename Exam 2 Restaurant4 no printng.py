import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the data from the Excel file
file_path = r'C:\Users\DacanayKC20\Downloads\Restaurant Revenue.xlsx'
data = pd.read_excel(file_path)

# Extract relevant columns
X = data[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 'Average_Customer_Spending', 'Promotions', 'Reviews']]
y = data['Monthly_Revenue']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R-squared score
r_squared = r2_score(y_test, y_pred)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the data from the Excel file
file_path = r'C:\Users\DacanayKC20\Downloads\Restaurant Revenue.xlsx'
data = pd.read_excel(file_path)

# Extract relevant columns
X = data[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 'Average_Customer_Spending', 'Promotions', 'Reviews']]
y = data['Monthly_Revenue']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R-squared score
r_squared = r2_score(y_test, y_pred)

# Print summary
print("Summary of Important Data:")
print("---------------------------")
print("Number of samples in dataset:", len(data))
print("Number of features:", X.shape[1])
print("Features used for prediction:", list(X.columns))
print("")

print("Model Performance:")
print("-----------------")
print("R-squared score:", r_squared)
print("")

print("Model Coefficients:")
print("-------------------")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

print("Intercept:", model.intercept_)

print("Go Brewers") 

# Plotting the results
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')

# Plotting the regression line
plt.plot(y_test, y_test, color='red', label='Regression Line')

plt.title('Monthly Revenue Prediction')
plt.xlabel('Actual Monthly Revenue')
plt.ylabel('Predicted Monthly Revenue')
plt.legend()

# Annotating R-squared score on the graph
plt.text(0.1, 0.9, f'R-squared: {r_squared:.2f}', transform=plt.gca().transAxes, fontsize=12)

plt.show()
