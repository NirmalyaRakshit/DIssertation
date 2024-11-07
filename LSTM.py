# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('Train.csv')

# Step 2: Data Preprocessing

# Convert 'Date' column to datetime if it's in string format
data['Date'] = pd.to_datetime(data['date'])

# Feature Engineering: Create new date-related features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

label_encoder = LabelEncoder()
data['family_encoded'] = label_encoder.fit_transform(data['family'])
data['city_encoded'] = label_encoder.fit_transform(data['city'])
data['type_y_encoded'] = label_encoder.fit_transform(data['type_y'])
data['locale_encoded'] = label_encoder.fit_transform(data['locale'])
data['transferred_encoded'] = label_encoder.fit_transform(data['transferred'])
data['description_encoded'] = label_encoder.fit_transform(data['description'])

# Drop irrelevant columns, including 'Date' (since we now have date features)
X = data.drop(['sales', 'Date', 'date', 'state', 'type_x',
               'Unnamed: 17', 'locale_name', 'description', 'family', 'city', 'type_y', 'locale',
               'transferred', 'description'], axis=1)  # Features
y = data['sales']  # Target variable

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input for LSTM: (samples, time_steps, features)
# We treat each row as one timestep. If you have sequential data, modify the time_steps accordingly.
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Initialize and build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Step 5: Train the LSTM model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse}")
print(f"R-squared (R2 Score): {r2}")

# Step 8: Visualize the actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:50], label='Actual Sales', color='blue')
plt.plot(y_pred[:50], label='Predicted Sales', linestyle='dashed', color='red')
plt.title('Actual vs Predicted Sales (First 50 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
