import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load dataset

df = pd.read_csv(r"D:\crop\Fertilizer Prediction.csv")

# Label Encoding

soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fert_encoder = LabelEncoder()

df['Soil Type'] = soil_encoder.fit_transform(df['Soil Type'])
df['Crop Type'] = crop_encoder.fit_transform(df['Crop Type'])

df['Fertilizer Name'] = fert_encoder.fit_transform(
    df['Fertilizer Name']
)

# Features & Target

X = df.drop('Fertilizer Name', axis=1)

y = df['Fertilizer Name']

# Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model

model = RandomForestClassifier()

# Train

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("Fertilizer Model Accuracy:",
      round(accuracy * 100, 2), "%")

# Save model

pickle.dump(model, open("fertilizer_model.pkl", "wb"))

pickle.dump(soil_encoder, open("soil_encoder.pkl", "wb"))

pickle.dump(crop_encoder, open("crop_encoder.pkl", "wb"))

pickle.dump(fert_encoder, open("fert_encoder.pkl", "wb"))

print("Fertilizer Model Saved ✅")