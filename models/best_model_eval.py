import sys
sys.path.append('/Users/ndungajr/PycharmProjects/hw05')
from data.input_data import DatasetCreator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Create Datasets
dataset_creator = DatasetCreator()
blob_dataset = dataset_creator.create_blob_dataset()
circles_dataset = dataset_creator.create_make_circles_dataset()

# Step 2: Split Data into Training, Validation, and Test Sets
# For both datasets (blob_dataset and circles_dataset),
# split into training, validation, and test sets:

# For Blob Dataset
X_blob, y_blob = blob_dataset['X'], blob_dataset['y']
X_circles, y_circles = circles_dataset['X'], circles_dataset['y']

# Split blob dataset into training and temporary (remaining) data
X_blob_train_temp, X_blob_test, y_blob_train_temp, y_blob_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state=42)

# Further split the temporary data into training and validation sets
X_blob_train, X_blob_val, y_blob_train, y_blob_val = train_test_split(X_blob_train_temp, y_blob_train_temp, test_size=0.25, random_state=42)

# Dimensions of each set
print(f"Blob Dataset:")
print(f"Train set: {X_blob_train.shape}, Validation set: {X_blob_val.shape}, Test set: {X_blob_test.shape}")

# For Circles Dataset
# Split circles dataset into training and temporary (remaining) data
X_circles_train_temp, X_circles_test, y_circles_train_temp, y_circles_test = train_test_split(X_circles, y_circles, test_size=0.2, random_state=42)

# Further split the temporary data into training and validation sets
X_circles_train, X_circles_val, y_circles_train, y_circles_val = train_test_split(X_circles_train_temp, y_circles_train_temp, test_size=0.25, random_state=42)

# Dimensions of each set
print(f"\nCircles Dataset:")
print(f"Train set: {X_circles_train.shape}, Validation set: {X_circles_val.shape}, Test set: {X_circles_test.shape}")

# Step 3: Hyperparameter Tuning (Using Validation Set)
# Use the validation set (X_blob_val, y_blob_val for blob dataset and
# X_circles_val, y_circles_val for circles dataset)
# to tune hyperparameters

# Step 4: Define the model and hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Create the SVM model
svm_model = SVC()

# Perform GridSearchCV for the Blob dataset
grid_search_blob = GridSearchCV(svm_model, param_grid, refit=True, verbose=2, cv=5)
grid_search_blob.fit(X_blob_val, y_blob_val)

# Print the best parameters and best score for the Blob dataset
print(f"Best parameters for Blob dataset: {grid_search_blob.best_params_}")
print(f"Best score for Blob dataset: {grid_search_blob.best_score_}")

# Perform GridSearchCV for the Circles dataset
grid_search_circles = GridSearchCV(svm_model, param_grid, refit=True, verbose=2, cv=5)
grid_search_circles.fit(X_circles_val, y_circles_val)

# Print the best parameters and best score for the Circles dataset
print(f"Best parameters for Circles dataset: {grid_search_circles.best_params_}")
print(f"Best score for Circles dataset: {grid_search_circles.best_score_}")

# Step 5: Evaluate the best model on the test set
# For Blob dataset
best_model_blob = grid_search_blob.best_estimator_
y_blob_pred = best_model_blob.predict(X_blob_test)
accuracy_blob = accuracy_score(y_blob_test, y_blob_pred)
print(f"Test set accuracy for Blob dataset: {accuracy_blob}")

# For Circles dataset
best_model_circles = grid_search_circles.best_estimator_
y_circles_pred = best_model_circles.predict(X_circles_test)
accuracy_circles = accuracy_score(y_circles_test, y_circles_pred)
print(f"Test set accuracy for Circles dataset: {accuracy_circles}")