# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset (Replace 'your_data.csv' with your actual dataset)
data = pd.read_csv('Train.csv')

# Step 1: Data Preprocessing

# Feature engineering: Convert categorical features like store location, season, etc.
# (You may need to change these based on your actual dataset)


data['Date'] = pd.to_datetime(data['date'])

# Feature Engineering: Create new date-related features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day


data['family'] = data['family'].astype('category').cat.codes
data['city'] = data['city'].astype('category').cat.codes
data['type_y'] = data['type_y'].astype('category').cat.codes
data['locale'] = data['locale'].astype('category').cat.codes
data['transferred'] = data['transferred'].astype('category').cat.codes
data['description'] = data['description'].astype('category').cat.codes

# Handling missing values (if any)
#data.fillna(method='ffill', inplace=True)

# Separating target variable (sales) from features
X = data.drop(['sales', 'Date', 'date', 'state', 'type_x',
               'Unnamed: 17', 'locale_name',], axis=1)  # Assuming 'sales' is the target column
y = data['sales']

# Splitting data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Training (XGBoost)
# Define the model
xgboost_model = xgb.XGBRegressor(
    n_estimators=1000,         # Number of trees
    learning_rate=0.05,        # Learning rate (how quickly the model adapts to the problem)
    max_depth=5,               # Maximum depth of a tree
    subsample=0.8,             # Fraction of samples used for training each tree
    colsample_bytree=0.8,      # Fraction of features used for each tree
    objective='reg:squarederror',  # Error function to be minimized (regression)
    random_state=42
)

# Train the model
xgboost_model.fit(X_train, y_train)

# Step 3: Model Evaluation
# Predict on the test set
y_pred = xgboost_model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print performance metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 4: Feature Importance (Optional)
# Plot feature importance for interpretation
xgb.plot_importance(xgboost_model)
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:50], label='Actual Sales', color='blue')
plt.plot(y_pred[:50], label='Predicted Sales', linestyle='dashed', color='red')
plt.title('Actual vs Predicted Sales (First 50 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 8: Feature importance
importances = xgboost_model.feature_importances_
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='green')
plt.title('Feature Importance in XGBoost Model')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()





# Step 5: Predict future sales
# Assuming you have new data for which you want to forecast sales
# Replace 'new_data.csv' with the dataset containing the future features
new_data = pd.read_csv('new_data.csv')

# Preprocess the new data similarly to the training data
new_data['store_location'] = new_data['store_location'].astype('category').cat.codes
new_data['season'] = new_data['season'].astype('category').cat.codes
new_data['marketing'] = new_data['marketing'].astype('category').cat.codes
new_data['time_of_day'] = new_data['time_of_day'].astype('category').cat.codes

# Predict sales for the new data
sales_forecast = xgboost_model.predict(new_data)

# Print the sales forecast
print("Predicted sales for new data:", sales_forecast)
