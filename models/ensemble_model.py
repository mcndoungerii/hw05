import sys
from data.input_data import DatasetCreator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pathlib import Path
import numpy as np

# Get the absolute path of the current file
current_file_path = Path('ensemble_model.py').resolve()

# Get the directory of the current file
project_dir = current_file_path.parent

# Add the project directory to sys.path
sys.path.insert(0, str(project_dir))

# Step 1: Create Datasets
dataset_creator = DatasetCreator()
blob_dataset = dataset_creator.create_blob_dataset()
circles_dataset = dataset_creator.create_make_circles_dataset()

# Step 2: Split Data into Training, Validation, and Test Sets
X_blob, y_blob = blob_dataset['X'], blob_dataset['y']
X_circles, y_circles = circles_dataset['X'], circles_dataset['y']

# Split blob dataset into training and temporary (remaining) data
X_blob_train_temp, X_blob_test, y_blob_train_temp, y_blob_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state=42)
X_blob_train, X_blob_val, y_blob_train, y_blob_val = train_test_split(X_blob_train_temp, y_blob_train_temp, test_size=0.25, random_state=42)

print(f"Blob Dataset:")
print(f"Train set: {X_blob_train.shape}, Validation set: {X_blob_val.shape}, Test set: {X_blob_test.shape}")

# Split circles dataset into training and temporary (remaining) data
X_circles_train_temp, X_circles_test, y_circles_train_temp, y_circles_test = train_test_split(X_circles, y_circles, test_size=0.2, random_state=42)
X_circles_train, X_circles_val, y_circles_train, y_circles_val = train_test_split(X_circles_train_temp, y_circles_train_temp, test_size=0.25, random_state=42)

print(f"\nCircles Dataset:")
print(f"Train set: {X_circles_train.shape}, Validation set: {X_circles_val.shape}, Test set: {X_circles_test.shape}")

# Step 3: Define RandomForest Classifiers with different hyperparameters
rf1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf2 = RandomForestClassifier(n_estimators=200, max_depth=10, max_features='sqrt', random_state=42)
rf3 = RandomForestClassifier(n_estimators=150, max_depth=None, bootstrap=False, random_state=42)

# Train the models on blob dataset
rf1.fit(X_blob_train, y_blob_train)
rf2.fit(X_blob_train, y_blob_train)
rf3.fit(X_blob_train, y_blob_train)

# Evaluate the models on validation set
rf1_val_acc = accuracy_score(y_blob_val, rf1.predict(X_blob_val))
rf2_val_acc = accuracy_score(y_blob_val, rf2.predict(X_blob_val))
rf3_val_acc = accuracy_score(y_blob_val, rf3.predict(X_blob_val))

print(f"Validation Accuracy of RF1: {rf1_val_acc}")
print(f"Validation Accuracy of RF2: {rf2_val_acc}")
print(f"Validation Accuracy of RF3: {rf3_val_acc}")

# Step 4: Create a Voting Classifier with different models
voting_clf_hard = VotingClassifier(estimators=[
    ('rf1', rf1), ('rf2', rf2), ('rf3', rf3)],
    voting='hard')

voting_clf_soft = VotingClassifier(estimators=[
    ('rf1', rf1), ('rf2', rf2), ('rf3', rf3)],
    voting='soft', weights=[1, 2, 1])

# Train the Voting Classifier on blob dataset
voting_clf_hard.fit(X_blob_train, y_blob_train)
voting_clf_soft.fit(X_blob_train, y_blob_train)

# Evaluate the Voting Classifier on validation set
voting_hard_val_acc = accuracy_score(y_blob_val, voting_clf_hard.predict(X_blob_val))
voting_soft_val_acc = accuracy_score(y_blob_val, voting_clf_soft.predict(X_blob_val))

print(f"Validation Accuracy of Hard Voting Classifier: {voting_hard_val_acc}")
print(f"Validation Accuracy of Soft Voting Classifier: {voting_soft_val_acc}")

# Step 5: Evaluate the best model on the test set
best_model = voting_clf_soft  # Replace with the best model based on validation accuracy
test_acc = accuracy_score(y_blob_test, best_model.predict(X_blob_test))
print(f"Test Accuracy of the best model: {test_acc}")

# Repeat similar steps for circles dataset
rf1.fit(X_circles_train, y_circles_train)
rf2.fit(X_circles_train, y_circles_train)
rf3.fit(X_circles_train, y_circles_train)

rf1_val_acc = accuracy_score(y_circles_val, rf1.predict(X_circles_val))
rf2_val_acc = accuracy_score(y_circles_val, rf2.predict(X_circles_val))
rf3_val_acc = accuracy_score(y_circles_val, rf3.predict(X_circles_val))

print(f"Validation Accuracy of RF1 on Circles: {rf1_val_acc}")
print(f"Validation Accuracy of RF2 on Circles: {rf2_val_acc}")
print(f"Validation Accuracy of RF3 on Circles: {rf3_val_acc}")

voting_clf_hard.fit(X_circles_train, y_circles_train)
voting_clf_soft.fit(X_circles_train, y_circles_train)

voting_hard_val_acc = accuracy_score(y_circles_val, voting_clf_hard.predict(X_circles_val))
voting_soft_val_acc = accuracy_score(y_circles_val, voting_clf_soft.predict(X_circles_val))

print(f"Validation Accuracy of Hard Voting Classifier on Circles: {voting_hard_val_acc}")
print(f"Validation Accuracy of Soft Voting Classifier on Circles: {voting_soft_val_acc}")

best_model = voting_clf_soft  # Replace with the best model based on validation accuracy
test_acc = accuracy_score(y_circles_test, best_model.predict(X_circles_test))
print(f"Test Accuracy of the best model on Circles: {test_acc}")
