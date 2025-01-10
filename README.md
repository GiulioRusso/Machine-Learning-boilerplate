# ⚙️ Machine Learning Boilerplate Project

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

## Directory Structure
```
project/
├── data/
│   └── data.csv
├── main.ipynb
├── requirements.txt
├── README.md
```

## Contribution
Feel free to fork this repository, make enhancements, and submit a pull request.

## License
This project is licensed under the MIT License.
