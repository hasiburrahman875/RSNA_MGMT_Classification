import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Read the CSV file
data = pd.read_csv("/cluster/pixstor/madrias-lab/jewel/radiomic_feature/bratsRadiomicData_on_585_samples.csv")  # csv contains BraTS21ID, Energy_Value, Volume, MGMT_value
data = data[:450] # using first 450 samples

# Drop rows with missing values for Energy_Value or Volume
data.dropna(subset=['Energy_Value', 'Volume'], inplace=True)
print("total data:", len(data))

# Split the data into features (X) and target (y)
X = data[['BraTS21ID', 'Energy_Value', 'Volume']]
y = data['MGMT_value']


# Split data into training and testing sets, with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Convert X_train and X_test to DataFrame
X_train_df = pd.DataFrame(X_train, columns=['BraTS21ID', 'Energy_Value', 'Volume'])
X_test_df = pd.DataFrame(X_test, columns=['BraTS21ID', 'Energy_Value', 'Volume'])

# Convert y_train and y_test to DataFrame
y_train_df = pd.DataFrame(y_train, columns=['MGMT_Value'])
y_test_df = pd.DataFrame(y_test, columns=['MGMT_Value'])

# Merge the label with features for train and test sets
train_set = pd.concat([X_train_df, y_train_df], axis=1)
test_set = pd.concat([X_test_df, y_test_df], axis=1)

# Save train and test sets with BraTS21ID, Energy_Value, Volume, MGMT_Value
train_set.to_csv("train.csv", index=False)
test_set.to_csv("test.csv", index=False)

brats21id_test = X_test['BraTS21ID']
brats21id_train = X_train['BraTS21ID']
X_train = X_train[['Volume']]
X_test = X_test[['Volume']]


# Normalize features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_train_poly = np.stack((X_train, X_train**2), axis=1).reshape(len(X_train), 2)
print(X_train_poly.shape)
X_test_poly = np.stack((X_test, X_test**2), axis=1).reshape(len(X_test), 2)
print(X_test_poly.shape)

# print(X_normalized[0:5])

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, probability=True)

# Train the classifier
svm_classifier.fit(X_train_poly, y_train)

# Predict on the testing set
y_pred = svm_classifier.predict(X_test_poly)
print(y_pred[0])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy from radiomic features:", accuracy)

# Predict probabilities on the testing set
y_pred_proba = svm_classifier.predict_proba(X_test_poly)[:, 1]

# # Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC Score:", auc_score)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
# Save the plot to an image file
plt.savefig('roc_curve.png')

# Save BraTS21ID and predicted probabilities to CSV file
# for test
test_results = pd.DataFrame({'BraTS21ID': brats21id_test, 'MGMT_Value': y_pred, 'Predicted_Probability': y_pred_proba})
test_results.to_csv('test_results.csv', index=False)
# for train
# Predict on the training set
y_pred = svm_classifier.predict(X_train_poly)

# Calculate accuracy
accuracy = accuracy_score(y_train, y_pred)
print("Accuracy from radiomic features for train:", accuracy)

# Predict probabilities on the testing set
y_pred_proba = svm_classifier.predict_proba(X_train_poly)[:, 1]

train_results = pd.DataFrame({'BraTS21ID': brats21id_train, 'MGMT_Value': y_pred, 'Predicted_Probability': y_pred_proba})
train_results.to_csv('train_results.csv', index=False)

# Save the model to disk
joblib.dump(svm_classifier, 'svm_model_1.pkl')
