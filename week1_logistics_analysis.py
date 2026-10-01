import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the sample logistics dataset
df = pd.read_csv("logistics_orders.csv")

# Basic cleaning
df = df.drop_duplicates()

# Create delay feature
df["delay_min"] = df["actual_time_min"] - df["planned_time_min"]

# KPI calculations
on_time_rate = (df["delay_min"] <= 0).mean() * 100
avg_delivery_time = df["actual_time_min"].mean()
avg_distance = df["distance_km"].mean()

print("On-Time Delivery Rate:", round(on_time_rate, 2), "%")
print("Average Delivery Time:", round(avg_delivery_time, 2), "minutes")
print("Average Distance:", round(avg_distance, 2), "km")

# Features for delivery-time prediction
features = [
    "distance_km",
    "parcel_count",
    "traffic_level",
    "hour",
    "vehicle_capacity"
]

X = df[features]
y = df["actual_time_min"]

# Preprocess categorical and numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("traffic", OneHotEncoder(handle_unknown="ignore"), ["traffic_level"]),
    ],
    remainder="passthrough"
)

# Regression pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5

print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))

# Route optimization concept
print("\nRoute Optimization Concept:")
print("- Minimize distance + travel time + delay risk")
print("- Respect vehicle capacity")
print("- Respect delivery time windows")
print("- Consider vehicle availability")

# Note:
# logistics_orders.csv is a synthetic sample dataset created for
# demonstrating the Week 1 analytical workflow. It is not company data.
