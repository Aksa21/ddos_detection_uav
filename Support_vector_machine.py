# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay


# Importing the dataset
dataset = pd.read_csv('final_data.csv')

# Separate features and target variable
X = dataset.drop('Scenario', axis=1).values
y = dataset['Scenario'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Initialize the scalers
scaler_total_packets = StandardScaler()
scaler_total_size = StandardScaler()
scaler_mean_size = StandardScaler()
scaler_mean_interarrival = StandardScaler()

# Normalize 'Total packets'
X_train[:, 0] = scaler_total_packets.fit_transform(X_train[:, 0].reshape(-1, 1)).flatten()
X_test[:, 0] = scaler_total_packets.transform(X_test[:, 0].reshape(-1, 1)).flatten()

# Normalize 'Total size'
X_train[:, 1] = scaler_total_size.fit_transform(X_train[:, 1].reshape(-1, 1)).flatten()
X_test[:, 1] = scaler_total_size.transform(X_test[:, 1].reshape(-1, 1)).flatten()

# Normalize 'Mean size'
X_train[:, 2] = scaler_mean_size.fit_transform(X_train[:, 2].reshape(-1, 1)).flatten()
X_test[:, 2] = scaler_mean_size.transform(X_test[:, 2].reshape(-1, 1)).flatten()

# Normalize 'Mean inter-arrival'
X_train[:, 3] = scaler_mean_interarrival.fit_transform(X_train[:, 3].reshape(-1, 1)).flatten()
X_test[:, 3] = scaler_mean_interarrival.transform(X_test[:, 3].reshape(-1, 1)).flatten()
#print(X_train)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel='linear', random_state=0)
#classifier = SVC(kernel='poly', degree=3, C=1.0, random_state=0)
#classifier = SVC(kernel='poly', degree=3, C=1.0, random_state=0)
classifier.fit(X_train, y_train)

start_time = time.time() 
# Predicting the Test set results
y_pred = classifier.predict(X_test)
end_time = time.time()

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Calculating accuracy and confusion matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'])
print("Classification Report:\n", report)

#Visualizing the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('Confusion Matrix.png', dpi=600)  # Save confusion matrix as PNG
plt.show()


# Display confusion matrix from estimator
target_names = ['Normal', 'Anomaly']
ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, display_labels=target_names, normalize=None, cmap="OrRd")
plt.title('Testing Confusion Matrix SVM')
plt.savefig('Testing Confusion Matrix SVM.png', dpi=600)
plt.show()


total_runtime = end_time - start_time

print("Total runtime:", total_runtime, "seconds")