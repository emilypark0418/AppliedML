# Applied ML Coursework HW
- COMS4995 Applied Machine Learning (Spring 2022, Prof. Vijay Pappu)
- Authors: **Chaewon (Emily) Park**

## HW1:

### Linear
- Missing value / inconsistent data unit analysis
- Plot distribution, correlation
- Encode categorial features using target encoder and standardize columns
- Implement Ridge Regression (L2 Reg) from scratch using closed form solution

### Logistic
- Feature correlation analysis and remove multi-collinearity (features with correlation > 0.9)
- Encode categorial features using ordinal encoder and standardize columns
- Implement Logistic Regression with L2 regularization from scratch using gradient descent
- Feature weight importance analysis and visualization 

### SVM
- Train, evaluate, and compare Dual SVM for both linear and non-linear (Gaussian) kernel

## HW2: 

### Decision Trees
- Missing value analysis and handling
- Encode categorial features using target encoder and standardize columns
- Train and eval DT
- Prune trees via cost complexity pruning path and its alpha
- Feature weight importance analysis and visualization 

### Random Forest, Gradient Boosted Trees (GradientBoostingClassifier, HistGradientBoostingClassifier, XGBoost)
- Train, eval, and compare model performance

### Calibration
- Calculate Brier Score before and after applying calibration on the model
- Apply two kinds of calibration: Isotonic Regression, Platt Scaling

## HW3: 

### Imbalanced Dataset
- Apply various balancing techniques: Random Undersampling, Random Oversampling, Synthetic Minority Oversampling Technique (SMOTE), balancing class weights
- Plot ROC curve, PR curve, confusion matrix

### Unsupervised Learning
- PCA
- K-means
- t-SNE

## HW4: 

### Feed Forward Network
- Implement forward and backward pass from scratch

### CNN
- Implement Feed Forward Network and LeNet-5 model from scratch to classify Fashion-MNIST dataset
- Compare model performance, # of total learnable parameters, confusion matrix
- Apply dropout and batch norm to handle overfitting


## HW5: NLP- Twitter Sentiment Classification

- Data Preprocessing [Removing certain regexes, Porter Stemming]
- Feature Extraction [Bag of Words, TF-IDF]
- Model Training & Evaluation
