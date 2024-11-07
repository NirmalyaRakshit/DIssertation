# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
# Replace 'sales_data.csv' with the actual path to your dataset
data = pd.read_csv('Train.csv')

# Display first few rows of the data
#print("First few rows of the data:")
#print(data.head())

# Step 2: Data Preprocessing

# Convert 'Date' column to datetime if it's in string format
data['Date'] = pd.to_datetime(data['date'],  dayfirst=True)

# Feature Engineering: Create new date-related features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
#data['DayOfWeek'] = data['Date'].dt.dayofweek

label_encoder = LabelEncoder()
data['family_encoded'] = label_encoder.fit_transform(data['family'])
data['city_encoded'] = label_encoder.fit_transform(data['city'])
data['type_y_encoded'] = label_encoder.fit_transform(data['type_y'])
data['locale_encoded'] = label_encoder.fit_transform(data['locale'])
data['transferred_encoded'] = label_encoder.fit_transform(data['transferred'])
data['description_encoded'] = label_encoder.fit_transform(data['description'])



# Drop irrelevant columns, including 'Date' (since we now have date features)
X = data.drop(['sales', 'Date','date', 'state', 'type_x', 'locale_name', 'description', 'family', 'city', 'type_y', 'locale',
               'transferred', 'description'], axis=1)  # Features

y = data['sales']  # Target variable

# Check for missing values
#print("\nMissing values in the dataset:")
#print(data.isnull().sum())

# Handle missing values if any
#X.fillna(method='ffill', inplace=True)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

accuracyl = accuracy_score(y_test, y_pred)
precisionl = precision_score(y_test, y_pred)
recalll = recall_score(y_test, y_pred)
f1l = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracyl:.4f}\nPrecisionl: {precisionl:.4f}\nRecall: {recalll:.4f}\nF1 Score: {f1l:.4f}")


print(f"\nMean Squared Error (MSE): {mse}")
print(f"R-squared (R2 Score): {r2}")

# Step 7: Visualize the actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:50], label='Actual Sales', color='blue')
plt.plot(y_pred[:50], label='Predicted Sales', linestyle='dashed', color='red')
plt.title('Actual vs Predicted Sales (First 50 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 8: Feature importance
importances = rf_model.feature_importances_
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='green')
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
