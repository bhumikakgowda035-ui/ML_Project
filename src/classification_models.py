# ============================================================
# CLASSIFICATION MODELS - SUPERVISED LEARNING PROJECT
# ============================================================

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ------------------------------------------------------------
# 1. Load Iris dataset
# ------------------------------------------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(
    iris.target,
    name="target"
)


# ------------------------------------------------------------
# 2. Split dataset into training and testing sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 3. Logistic Regression model
# ------------------------------------------------------------

logistic_model = LogisticRegression(max_iter=200)

logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)


# ------------------------------------------------------------
# 4. Decision Tree model
# ------------------------------------------------------------

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)


# ------------------------------------------------------------
# 5. Evaluate Logistic Regression
# ------------------------------------------------------------

logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
logistic_precision = precision_score(
    y_test, y_pred_logistic, average="weighted"
)
logistic_recall = recall_score(
    y_test, y_pred_logistic, average="weighted"
)
logistic_f1 = f1_score(
    y_test, y_pred_logistic, average="weighted"
)


# ------------------------------------------------------------
# 6. Evaluate Decision Tree
# ------------------------------------------------------------

tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_precision = precision_score(
    y_test, y_pred_tree, average="weighted"
)
tree_recall = recall_score(
    y_test, y_pred_tree, average="weighted"
)
tree_f1 = f1_score(
    y_test, y_pred_tree, average="weighted"
)


# ------------------------------------------------------------
# 7. Display results
# ------------------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        logistic_accuracy,
        tree_accuracy
    ],
    "Precision": [
        logistic_precision,
        tree_precision
    ],
    "Recall": [
        logistic_recall,
        tree_recall
    ],
    "F1 Score": [
        logistic_f1,
        tree_f1
    ]
})

print("\n===== CLASSIFICATION MODEL RESULTS =====\n")

print(results.round(4))


# ------------------------------------------------------------
# 8. Save results
# ------------------------------------------------------------

results.to_csv(
    "classification_results.csv",
    index=False
)

print("\nClassification results saved successfully!")
