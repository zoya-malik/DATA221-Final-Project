#DATA 221 — Introduction to Data Science
#Author: Siyi Ye
#Dataset: Credit Card Approval Prediction (Kaggle)

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score

# Load data
df = pd.read_csv('../data/cleaned_data.csv')
print(f"Original Dataset Shape: {df.shape}")

y = df['TARGET']
X = df.drop(columns=['TARGET'])
print(f"Features (X) Shape: {X.shape}")
print(f"Target (y) Shape: {y.shape}")

# Divide the training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identify Category Feature Columns
# A Pandas warning appeared when running the code.
# Asked Gemini for help to modify this line of code to eliminate the warning.
cat_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()

# Encode separately and remove redundant dimensions
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# Align the column structure of the test set and the training set
# to avoid code crashes due to mismatches.
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Model Training

# First attempt
# Set max_depth to limit depth to prevent overfitting

# model = DecisionTreeClassifier(max_depth=5, random_state=42)
# model.fit(X_train, y_train)

# Result: High Accuracy (0.88), but Recall for Class 1 is near 0.
# Observation: Model is biased towards the majority class (Class 0).

# Second attempt
# Use "class_weight=balanced" to avoid errors in minority classes.
model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n              Classification Report ")
print(classification_report(y_test, y_pred))

# Result:
# The model showed a significant improvement in performance for the minority group (Class 1).
# The recall rate for high-risk applicants increased from 0.01 to 0.39.
# Although the overall accuracy decreased to 0.67, this version is more practical for credit risk management
# because it successfully identifies a larger proportion of potential defaulters.


# Continue to try changing max_depth multiple times to obtain the optimal result
from sklearn.metrics import accuracy_score, recall_score
depths = range(2, 16)
accuracy_scores = []
recall_scores = []
for d in depths:
    test_model = DecisionTreeClassifier(max_depth=d, random_state=42, class_weight='balanced')
    test_model.fit(X_train, y_train)
    y_pred = test_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    accuracy_scores.append(acc)
    recall_scores.append(rec)
    print(f"Checking depth {d}: Accuracy is {acc:.2f}, Recall is {rec:.2f}")

# Seeking syntax correction and assistance from Gemini in this section of the code.
plt.figure(figsize=(10, 6))
plt.plot(list(depths), accuracy_scores, label='Overall Accuracy', marker='o')
plt.plot(list(depths), recall_scores, label='Recall (Class 1)', marker='s')

plt.xlabel('Tree Depth (max_depth)')
plt.ylabel('Score')
plt.title('Finding the Optimal Tree Depth (Balanced)')
plt.legend()
plt.grid(True)

plt.savefig('../results/depth_tuning_plot.png')
print("\n[SUCCESS] Tuning plot saved to ../results/depth_tuning_plot.png")

# Result:
# Although the highest Recall was achieved at depth 15, I chose depth 5 as the final model.
# As seen in the tuning plot, beyond depth 5, the Overall Accuracy begins to decline while Recall continues to rise.
# This is a classic sign of Overfitting, where the model memorizes noise instead of learning general patterns.
# A depth of 5 provides the best balance between predictive power and model stability.

# Results visualization
importances = pd.Series(model.feature_importances_, index=X_train.columns)
plt.figure(figsize=(10, 6))
importances.nlargest(10).sort_values().plot(kind='barh', color='salmon')
plt.title('Top 10 Decision Factors for Credit Approval')
plt.tight_layout()
plt.savefig('../results/decision_tree_feature_importance.png')

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Reds')
plt.savefig('../results/decision_tree_confusion_matrix.png')

from sklearn.tree import plot_tree
# Train a model with a depth of 3 specifically for visualization.
viz_model = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
viz_model.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(viz_model,
          feature_names=X_train.columns.tolist(),
          class_names=['Good', 'Bad'],
          filled=True,
          proportion=True,
          rounded=True,
          precision=2,
          fontsize=12,
          impurity=True)

plt.title("Final Decision Tree Structure (3 Levels Only)")
plt.savefig('../results/clean_tree_no_placeholder.png', dpi=300)

