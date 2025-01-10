# Machine Learning Boilerplate Project

This project provides a comprehensive pipeline for machine learning tasks, enabling seamless preprocessing, model training, and evaluation. The framework supports both classification and regression problems and is highly customizable.

## Features
- **Data Preprocessing**: 
  - One-hot encoding for categorical features.
  - Automatic detection of non-numeric columns.
  - Scaling and normalization using `StandardScaler`.
  - Dimensionality reduction via PCA.
  - Feature selection with statistical methods.
- **Model Selection**: 
  - Predefined pipelines for classification and regression models.
  - Hyperparameter tuning using `GridSearchCV`.
- **Cross-Validation**: 
  - Supports stratified and shuffle-split cross-validation.
- **Evaluation**: 
  - Regression metrics: MSE, MAE, R².
  - Classification metrics: Accuracy, Classification Report, Confusion Matrix.

## Requirements

Install the required libraries:

```bash
pip3 install -r requirements.txt
```

### Libraries Used
- `pandas` for data manipulation.
- `numpy` for numerical operations.
- `scikit-learn` for preprocessing, modeling, and evaluation.
- `seaborn` and `matplotlib` for visualization.

## Usage

### 1. Data Loading and Preprocessing
- Load the dataset from a CSV file.
- Split features (`X`) and target (`y`).
- Automatically handle categorical variables with one-hot encoding.

### 2. Model Training
- Use predefined pipelines for various models:
  - Classification: Random Forest, SVM, XGBoost, MLP.
  - Regression: Linear Regression, Random Forest Regressor, SVR.
- Configure hyperparameters with `GridSearchCV`.

### 3. Evaluation
- Perform cross-validation to select the best model.
- Evaluate the model on the test set using appropriate metrics.
- Visualize classification results using a confusion matrix heatmap.

## Example Workflow
1. Load and preprocess data:

    ```python
    data = pd.read_csv('data/data.csv')
    X, y = preprocess_data(data)
    ```

2. Train models:

    ```python
    grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    ```

3. Evaluate and visualize:

    ```python
    y_pred = grid_search.best_estimator_.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    ```

## Directory Structure
```
project/
├── data/
│   └── data.csv
├── src/
│   └── main.ipynb
├── requirements.txt
├── README.md
```

## Contribution
Feel free to fork this repository, make enhancements, and submit a pull request.

## License
This project is licensed under the MIT License.
