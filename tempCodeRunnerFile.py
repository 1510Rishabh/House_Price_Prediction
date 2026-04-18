import kagglehub
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

path = kagglehub.dataset_download("harishkumardatalab/housing-price-prediction")
files = os.listdir(path)
df = pd.read_csv(os.path.join(path, files[0]))
df = df.dropna()

encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

target_column = 'price'
X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

print("\nFeatures:")
for col in X.columns:
    print("-", col)

labels = {
    "area": "area",
    "bedrooms": "bedrooms (1,2,3)",
    "bathrooms": "bathrooms (1,2,3)",
    "stories": "stories (1,2,3)",
    "parking": "parking space (0,1,2,3)",
    "mainroad": "mainroad (yes/no)",
    "guestroom": "guestroom (yes/no)",
    "basement": "basement (yes/no)",
    "hotwaterheating": "hotwaterheating (yes/no)",
    "airconditioning": "airconditioning (yes/no)",
    "prefarea": "prefarea (yes/no)",
    "furnishingstatus": "furnishingstatus (furnished/semi-furnished/unfurnished)"
}

print("\n--- Input ---")

input_data = {}

for col in X.columns:
    prompt = labels.get(col, col)

    if col in encoders:
        while True:
            value = input(f"{prompt}: ").lower()
            if value in encoders[col].classes_:
                value = encoders[col].transform([value])[0]
                break
            else:
                print("Invalid")
    else:
        while True:
            try:
                value = float(input(f"{prompt}: "))
                break
            except ValueError:
                print("Invalid")

    input_data[col] = value

input_df = pd.DataFrame([input_data])
predicted_price = model.predict(input_df)[0]

print("\nPredicted Price:", predicted_price)