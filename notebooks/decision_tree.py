import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score

#Load data
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

cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Identify Category Feature Columns
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Encode separately and remove redundant dimensions
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)


print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")