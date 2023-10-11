import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.metrics import ConfusionMatrixDisplay

# Load the dataset
data = pd.read_csv('final_data.csv')
#
# Separate features and target variable
X = data.drop('Scenario', axis=1).values
y = data['Scenario'].values

#print(X)

# Initialize the scalers(
scaler_total_packets = StandardScaler()
scaler_total_size = StandardScaler()
scaler_mean_size = StandardScaler()
# scaler_mean_interarrival = StandardScaler()

# Normalize 'Total packets'
X[:, 0] = scaler_total_packets.fit_transform(X[:, 0].reshape(-1, 1)).flatten()

# Normalize 'Total size'
X[:, 1] = scaler_total_size.fit_transform(X[:, 1].reshape(-1, 1)).flatten()

# Normalize 'Mean inter-arrival'
X[:, 2] = scaler_mean_size.fit_transform(X[:, 2].reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
# X_train = sc.fit_transform(X_train)  # Scaling training features
# X_test = sc.transform(X_test)        # Scaling test features
#print(X_train.shape)
#print(X)
#print(y)
# Training the Naive Bayes model
classifier = GaussianNB()
classifier.fit(X_train, y_train)
start_time = time.time()
# Predicting the Test set results
y_pred = classifier.predict(X_test)

end_time = time.time()
# Calculating accuracy and confusion matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Display classification report (includes precision, recall, F1-score)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

# Visualizing the Confusion Matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.title("Confusion Matrix")
# plt.savefig('Confusion Matrix.png', dpi=600)  # Save confusion matrix as PNG
# plt.show()

# Display confusion matrix from estimator
target_names = ['Normal', 'Anomaly']
ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, display_labels=target_names, normalize=None, cmap="GnBu")
plt.title('Testing Confusion Matrix')
plt.savefig('Testing Confusion Matrix.png', dpi=600)
plt.show()

# Calculate runtime
  # Start time for runtime measurement

total_runtime = end_time - start_time

print("Total runtime:", total_runtime, "seconds")
