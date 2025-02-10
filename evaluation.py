import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression  # Example model

# Load the data, assuming it's in a file named 'data.pkl'
df = pd.read_pickle("data.pkl")

# Check if the target column exists
target_column = "target"  # Replace with the actual target column name
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

# Drop the 'target' column from the features (X)
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Example 80/20 split

# Initialize and train the model
model = LogisticRegression(solver='liblinear')  # Specify solver to avoid warnings
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Create and display the confusion matrix
cm = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')

# Add labels and title
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

# Set ticks and labels
ax.set_xticks(range(len(set(y))))  # Assuming y is categorical
ax.set_yticks(range(len(set(y))))
ax.set_xticklabels(set(y))
ax.set_yticklabels(set(y))

# Add colorbar
fig.colorbar(im)

# Save the figure
plt.savefig("confusion_matrix.png")
plt.show()  # Display the plot