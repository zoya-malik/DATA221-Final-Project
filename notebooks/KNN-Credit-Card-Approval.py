# Author Fazel Rabbi
# Data 221 Final Project
# KNN Model
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the data and identify the target variable
Data_frame_creditcard_approval = pd.read_csv("cleaned_data.csv")

X = Data_frame_creditcard_approval.drop('TARGET', axis=1)
Y = Data_frame_creditcard_approval['TARGET']

# Encode Categorical variables
# Converts Categorical values into numerical ones
# drop_true=True avoids duplicate columns, do this so it doesn't confuse the model
X = pd.get_dummies(X, drop_first=True)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    random_state=42,
    test_size=0.2,
    stratify=Y
)

# Scale and standardize the input data so KNN distance calculations work properly
scaler = StandardScaler()

# Fit the scaler on the training set then transform it, then only transform the testing set
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set the K values we wish to test
k_values = range(1, 21)

# Create storage for evaluation metrics
accuracy_scores = []
recall_scores = []
roc_auc_scores = []

# Build the KNN model for each value of k
for x in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=x)
    knn_model.fit(X_train, Y_train)
    knn_model_predict = knn_model.predict(X_test)
    knn_model_prob = knn_model.predict_proba(X_test)[:,1]

    # Append the results to the evaluation metrics storage
    accuracy_scores.append(accuracy_score(Y_test, knn_model_predict))
    recall_scores.append(recall_score(Y_test, knn_model_predict))
    roc_auc_scores.append(roc_auc_score(Y_test, knn_model_prob))

# Create a storage for the combined value of ROC-AUC and f1 Scores
combined_score = []

# Append all combined scores to the empty list
# Give a higher weight to ROC-AUC score because it is a more important metric
for i in range(len(k_values)):
    score = (0.5 * roc_auc_scores[i] + 0.5 * recall_scores[i])
    combined_score.append(score)

# Find the best K based on the combined score
best_index = combined_score.index(max(combined_score))
best_k = list(k_values)[best_index]

# Train the model using the best K value
best_knn_model = KNeighborsClassifier(n_neighbors=best_k)
best_knn_model.fit(X_train, Y_train)
best_knn_predict = best_knn_model.predict(X_test)
best_knn_prob = best_knn_model.predict_proba(X_test)[:, 1]

# Final evaluation
best_accuracy = accuracy_score(Y_test, best_knn_predict)
best_recall = recall_score(Y_test, best_knn_predict)
best_precision = precision_score(Y_test, best_knn_predict)
best_f1 = f1_score(Y_test, best_knn_predict)
best_roc_auc = roc_auc_score(Y_test, best_knn_prob)
best_confusion_matrix = confusion_matrix(Y_test, best_knn_predict)

print(f"Best K: {best_k}")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"Recall: {best_recall:.4f}")
print(f"Precision: {best_precision:.4f}")
print(f"F1 Score: {best_f1:.4f}")
print(f"ROC-AUC: {best_roc_auc:.4f}")

# ROC curve for the best K
false_positives_rate, true_positives_rate, thresholds = roc_curve(Y_test, best_knn_prob)

plt.figure()
plt.plot(false_positives_rate, true_positives_rate, label=f"KNN ROC Curve")
plt.plot([0,1], [0,1], linestyle = "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for the best K value")
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix plot
labels = np.array([["True Negative", "False Positive"], ["False Negative", "True Positive"]])


plt.figure()
plt.imshow(best_confusion_matrix, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])

for x in range(2):
    for y in range(2):
        plt.text(y, x, f"{labels[x][y]}\n{best_confusion_matrix[x][y]}", ha="center", va="center", color="black")

plt.show()







