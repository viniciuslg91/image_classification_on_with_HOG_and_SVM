import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

########################## 1. Choose an image classification problem containing only one category of interest (e.g.: pedestrians, faces, hands, flowers, cats, cars, etc.) and find or create a dataset containing approximately the same number of positive and negative examples, e.g. 100 images containing cats (positive) and 100 images not containing cats (negatives). ########################## 

# Paths to folders with images of cats and dogs

cats_folder = 'cats'
dogs_folder = 'dogs'

# Function to load images from a folder

def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder): # Iterates over all files in the folder specified by the parameter and returns a list of all files.
        img_path = os.path.join(folder, filename) # Creates the full path of the image file.
        if os.path.isfile(img_path): # Checks if the specified path (img_path) corresponds to a regular file (and not a folder).
            img = Image.open(img_path) # Opens the image file using the Pillow library (imported as Image).
            img = img.resize((128,128)) # Adjusts the size of the image.
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)
    return images, labels


# Load the sets of images of cats (with value equal to 0) and dog (with value equal to 1).
cat_images, cat_labels = load_images(cats_folder, 0)
dog_images, dog_labels = load_images(dogs_folder, 1)

# Join images of cats and dogs
images = np.array(cat_images + dog_images)
labels = np.array(cat_labels + dog_labels)

x = images
y = labels

# 2. Separate the data into train and test sets (70-30). Both sets should be balanced with 
# approximately the same number of positive and negative samples in each. Calculate the HOG of each image (feature vector).
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

########################## 3. Adjust the hyperparameters and train an SVM model with the training set images described by the HOG feature vector. ##########################

# Function to calculate HOG Features
def compute_hog_features(set_images):
    hog_features = []
    for img in set_images:
        imagem_gray = np.mean(img, axis=-1) # Convert to grayscale
        features, _ = hog(imagem_gray, pixels_per_cell=(8, 8), cells_per_block=(2,2), block_norm='L2-Hys', visualize=True)
        hog_features.append(features)
    return np.array(hog_features)

# Calculate HOG features for the training and testing images
x_train_hog = compute_hog_features(x_train)
x_test_hog = compute_hog_features(x_test)

# Normalize HOG characteristics
scaler = StandardScaler()
x_train_hog = scaler.fit_transform(x_train_hog)
x_test_hog = scaler.transform(x_test_hog)

# Define the SVM model
svm_model = SVC()

# Set parameters for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100], # Regularization
    'kernel': ['linear', 'rbf', 'poly'], # Kernel Type
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1], # Gamma parameter for RBF kernel
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1, verbose=3)

# Train SVM model with GridSearchCV
grid_search.fit(x_train_hog, y_train)

########################## 4. Using the trained SVM, classify the test set images. ##########################

# Print the best parameters found
print("Best parameters:", grid_search.best_params_)

# Making predictions on the test set
y_pred = grid_search.best_estimator_.predict(x_test_hog)

# View the classification report
print(classification_report(y_test, y_pred))

########################## 5. Compute train and test accuracy (or error rate) ##########################
 
x_train_grid_search = grid_search.best_estimator_.predict(x_train_hog)

# Calculate accuracy on training and testing set
accuracy_train = accuracy_score(y_train, x_train_grid_search)
accuracy_test = accuracy_score(y_test, y_pred)

# Error rate
train_error_rate = 1 - accuracy_train
test_error_rate = 1 - accuracy_test

# Print the aaccuracy and error rate
print ("Accuracy of the train", accuracy_train)
print ("Accuracy of the test", accuracy_test)
print("Error rate of the train", train_error_rate)
print("Error rate of the test", test_error_rate)
