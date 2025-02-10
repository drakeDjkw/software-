import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('/content/data.csv')

# Load and preprocess the data
data = pd.read_csv("/content/data.csv")  # Replace "data.csv" with the actual file name

X = data.drop("target", axis=1)  # Replace "target" with the actual target column name
y = data["target"]  # Replace "target" with the actual target column name

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()  # Replace with the actual model you want to use
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# Save the model
model.save("model.pkl")

# Load the model
loaded_model = load("model.pkl")

# Evaluate the model again
y_pred_loaded = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)
print("Loaded model accuracy:", loaded_accuracy)