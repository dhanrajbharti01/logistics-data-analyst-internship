# Load logistics data
df = pd.read_csv("logistics_orders.csv")

# Basic cleaning
df = df.drop_duplicates()
df["delay_min"] = df["actual_time_min"] - df["planned_time_min"]

# KPI calculations
on_time_rate = (df["delay_min"] <= 0).mean() * 100
avg_delivery_time = df["actual_time_min"].mean()
avg_distance = df["distance_km"].mean()

# Prepare features for delivery-time prediction
features = [
    "distance_km", "parcel_count", "traffic_level",
    "hour", "vehicle_capacity"
]

# Train/test split and regression model
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["actual_time_min"], test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Route optimization concept
# Objective: minimize weighted cost of distance + travel time + delay risk
# Constraints: vehicle capacity, delivery time windows, vehicle availability
