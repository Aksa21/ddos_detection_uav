import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import time

# Load and preprocess the dataset
data = pd.read_csv('final_data.csv')

X = data.drop('Scenario', axis=1).values
y = data['Scenario'].values



# Initialize the scalers
scaler_total_packets = StandardScaler()
scaler_total_size = StandardScaler()
scaler_mean_size = StandardScaler()
#scaler_mean_interarrival = StandardScaler()

# Normalize 'Total packets'
X[:, 0] = scaler_total_packets.fit_transform(X[:, 0].reshape(-1, 1)).flatten()
#X_test[:, 0] = scaler_total_packets.transform(X_test[:, 0].reshape(-1, 1)).flatten()

# Normalize 'Total size'
X[:, 1] = scaler_total_size.fit_transform(X[:, 1].reshape(-1, 1)).flatten()
#X_test[:, 1] = scaler_total_size.transform(X_test[:, 1].reshape(-1, 1)).flatten()

# Normalize 'Mean size'
X[:, 2] = scaler_mean_size.fit_transform(X[:, 2].reshape(-1, 1)).flatten()
#X_test[:, 2] = scaler_mean_size.transform(X_test[:, 2].reshape(-1, 1)).flatten()

# Normalize 'Mean inter-arrival'
#X_train[:, 3] = scaler_mean_interarrival.fit_transform(X_train[:, 3].reshape(-1, 1)).flatten()
#X_test[:, 3] = scaler_mean_interarrival.transform(X_test[:, 3].reshape(-1, 1)).flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Initialize the Random Forest Classifier
num_trees = 100
classifier = RandomForestClassifier(n_estimators=num_trees, criterion='entropy', random_state=0)

# Training the Random Forest Classifier
classifier.fit(X_train, y_train)

start_time = time.time()  # Start time for runtime measurement
# Predicting the Test set results
y_pred = classifier.predict(X_test)
end_time = time.time()
# Monitoring the training progress
train_accuracy = classifier.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualizing the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Display confusion matrix from estimator
target_names = ['Normal', 'Anomaly']
ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, display_labels=target_names, normalize=None, cmap="Blues")
plt.title('Testing Confusion Matrix')
plt.show()

# Calculate runtime


total_runtime = end_time - start_time

print("Total runtime:", total_runtime, "seconds")

