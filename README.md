# Developer_HUB_ML_Tasks

## Task 1: Disease Prediction Using Patient Data
### Objective:
Learn basic ML workflows to predict diseases like diabetes or heart disease.
### Dataset:
Used the UCI Heart Disease dataset (CSV format).
### Preprocessing:
- Filled missing values using mean/median/mode.
- Normalized numerical features between 0 and 1.
- Converted categorical features into numerical values using Label Encoding.
### Exploratory Data Analysis (EDA):
- Viewed dataset statistics using describe().
- Plotted a correlation heatmap to understand feature relationships.
### Model Training:
**Trained two models:** Logistic Regression and Random Forest.
### Evaluation:
Accuracy was used as the primary metric.
- **Logistic Regression accuracy:** 82.06%
- **Random Forest accuracy:** 84.78%

Random Forest performed better and was selected as the final model.

## Task 2: Cancer Detection Using Histopathological Images

### Objective:
Learn the basics of image data preprocessing and training CNNs.

### Dataset:
Used a subset of the Breast Cancer Histopathological Dataset from Kaggle (**~500+ images**).
Two classes: **Benign** (0) and **Malignant** (1).

### Data Preprocessing:
- Resized all images to **128×128** pixels.
- Normalized pixel values to range **[0,1]**.
- Applied data augmentation (rotation, shifting, zoom, flipping).

### Model Training (Custom CNN):
- Built a simple CNN with **3 convolutional layers**.
- Trained for **10 epochs** using binary crossentropy and Adam optimizer.

### Achieved:
- Training Accuracy:**~83.7%**
- Validation Accuracy: **~80.3%**
- Test Accuracy: **82.2%**

### Evaluation:
- Plotted accuracy and loss curves for both training and validation.
- Custom CNN showed decent performance but some fluctuations in validation accuracy.

### Bonus (Transfer Learning with VGG16):
Used **VGG16 (ImageNet weights)** as a frozen base model with additional dense layers.

### Achieved improved performance:
- Training Accuracy: **89.2%**
- Validation Accuracy: **84.2%**
- Test Accuracy: **88.4%**
- Training loss reduced significantly, showing better generalization.

### Observations:
- Transfer learning improved accuracy across **training** (0.83 → 0.89), **test** (0.82 → 0.88), and **validation** (0.80 → 0.84).
- Loss also decreased **(Train: 0.43 → 0.26, Test: 0.43 → 0.31)**.
- Validation loss slightly increased **(0.44 → 0.49)**, indicating higher accuracy but less confidence on unseen data.
- Overall, **VGG16** transfer learning outperformed the **custom CNN** with stronger feature extraction and stable results.
