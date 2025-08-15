# Blood Donation Prediction Project

# 📦 Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tpot import TPOTClassifier

# 📥 Step 2: Load Dataset
df = pd.read_csv('transfusion.data')

# 🧹 Step 3: Rename Target Column
df.rename(columns={'whether he/she donated blood in March 2007': 'target'}, inplace=True)

# 📊 Step 4: Analyze Target Distribution
print("Target Distribution:\n", df['target'].value_counts(normalize=True))

# ✂️ Step 5: Train-Test Split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 🤖 Step 6: TPOT AutoML
print("\nRunning TPOT AutoML...")
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print("\nTPOT Test Accuracy:", tpot.score(X_test, y_test))
tpot.export('best_model_pipeline.py')

# 📉 Step 7: Log Normalization
X_log = X.copy()
for col in X.columns:
    X_log[col] = np.log1p(X[col])

# 🔄 Step 8: Train-Test Split on Normalized Data
X_train_log, X_test_log, y_train, y_test = train_test_split(
    X_log, y, test_size=0.25, random_state=42, stratify=y
)

# 📈 Step 9: Logistic Regression Model
print("\nTraining Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_log, y_train)
y_pred = model.predict(X_test_log)

# 📋 Step 10: Evaluation
print("\nLogistic Regression Performance:\n")
print(classification_report(y_test, y_pred))