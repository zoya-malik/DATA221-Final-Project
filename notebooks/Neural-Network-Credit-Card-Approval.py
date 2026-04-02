# Author Fazel Rabbi
# Data 221 Final Project
# Neural Network
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler


# Load the data and identify the target variable
Data_frame_creditcard_approval = pd.read_csv("cleaned_data.csv")

X = Data_frame_creditcard_approval.drop('TARGET', axis=1)
Y = Data_frame_creditcard_approval['TARGET']

# Encode Categorical variables
# Converts Categorical values into numerical ones
# drop_true=True avoids duplicate columns, do this so it doesnt confuse the model
X = pd.get_dummies(X, drop_first=True)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    random_state=42,
    test_size=0.2,
    stratify=Y
)

# Scale and standardize the input data so the neural network trains better
scaler = StandardScaler()

# Fit the scaler on the training set then transform it, then only transform the testing set
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network
neural_model = Sequential()

# Input Layer
neural_model.add(Input(shape=(X_train.shape[1],)))

# First Hidden Layer
# relu activation helps the model learn non-linear patterns
neural_model.add(Dense(64, activation="relu"))

# Second Hidden Layer
# relu activation helps the model learn non-linear patterns
neural_model.add(Dense(32, activation="relu"))

# Output layer
# Sigmoid is used for binary classification
neural_model.add(Dense(1, activation="sigmoid"))

# Compile the model
# adam decides how the model updates its weights to reduce error
# binary-crossentropy measures how wrong the predictions are
neural_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the neural network
# Validation split uses part of the training data to check the models performance
trained_neural_network = neural_model.fit(
    X_train,
    Y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Predict probabilities for the test set
# Use flatten() so it can be used with evaluation metrics
neural_model_prob = neural_model.predict(X_test).flatten()

# Convert Probabilities into 0 or 1 for predictions
neural_model_predict = (neural_model_prob >= 0.3).astype(int)


# ROC curve
# helps visualize how the model separates the two classes
false_positives_rate, true_positives_rate, threshold = roc_curve(Y_test, neural_model_prob)

plt.figure()
plt.plot(false_positives_rate, true_positives_rate, label="Neural Network")
plt.plot([0,1], [0,1], linestyle = "--", label="Random Guess")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Confusion Matrix
# Shows the true positives, true negatives, false positives and false negatives
neural_model_cm = confusion_matrix(Y_test, neural_model_predict)

labels = np.array([["True Negative", "False Positive"], ["False Negative", "True Positive"]])


plt.figure()
plt.imshow(neural_model_cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])

for x in range(2):
    for y in range(2):
        plt.text(y, x, f"{labels[x][y]}\n{neural_model_cm[x][y]}", ha="center", va="center", color="black")

plt.show()

# Evaluation Metrics

print(f"Accuracy: {accuracy_score(Y_test, neural_model_predict):.4f}")
print(f"Precision: {precision_score(Y_test, neural_model_predict, zero_division=0):.4f}")
print(f"Recall: {recall_score(Y_test, neural_model_predict, zero_division=0):.4f}")
print(f"F1-Score: {f1_score(Y_test, neural_model_predict, zero_division=0):.4f}")
print(f"ROC-AUC: {roc_auc_score(Y_test, neural_model_prob):.4f}")





