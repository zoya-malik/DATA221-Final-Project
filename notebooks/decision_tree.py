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