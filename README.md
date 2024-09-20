# Insurance-Prediction
## Overview:
This project focuses on predicting the insurance cost for customers based on their demographic and personal information. Using a dataset of customer features, various machine learning models are applied to forecast the insurance premium, helping insurance companies estimate the risk and set appropriate pricing.

## Project Structure:
- Exploratory Data Analysis (EDA): Insights into the dataset using visualization and statistical summaries.
- Feature Engineering: Preprocessing and transforming the data to improve model performance.
- Modeling: Training various machine learning models to predict insurance costs, including:
    - **Linear Regression**
    - **Decision Tree Regressor**
    - **Random Forest Regressor**
    - **Support Vector Regressor (SVR)**
    - **Gradient Boosting Regressor**
    - **XGBoost Regressor**
    - **K-Nearest Neighbors Regressor (KNN)**
- Hyperparameter Tuning: Optimizing model performance using GridSearchCV or RandomizedSearchCV to fine-tune the parameters of models such as Random Forest, Gradient Boosting, and XGBoost.
- Model Evaluation: Assessing model performance using metrics like R² score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
## Installation:
To install the required libraries for this project, run the following commands:
```
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install datasist
pip install scikit-learn
pip install xgboost
```
## Usage:
1. Clone the repository:
```
git clone https://github.com/your-username/insurance-prediction.git
```
2. Navigate to the project directory:
```
cd insurance-prediction
```
3. Install the required libraries (see the installation section).
4. Run the notebook or Python scripts to explore and train models.
## Data:
The dataset contains customer data with features such as:
- Age, gender, and other demographics.
- Income and employment status.
- Health conditions and prior insurance history.
- Vehicle ownership and driving record.
## Models:
The following machine learning models were used:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **K-Nearest Neighbors Regressor**
- **Support Vector Regressor (SVR)**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**
## Hyperparameter Tuning:
To optimize model performance, hyperparameter tuning was performed using GridSearchCV and RandomizedSearchCV. The following parameters were tuned:
- Random Forest Regressor: Number of trees (n_estimators), depth (max_depth), and number of features (max_features).
- Gradient Boosting Regressor: Learning rate (learning_rate), number of estimators (n_estimators), and maximum depth (max_depth).
- XGBoost Regressor: Learning rate (learning_rate), number of boosting rounds (n_estimators), and max_depth.
## Evaluation:
The models are evaluated using several metrics, including:
- R² Score: Measures how well the model predicts the variance in the target.
- Mean Absolute Error (MAE): Average magnitude of errors in predictions.
- Root Mean Squared Error (RMSE): Square root of the average squared differences between predicted and actual values.
## Results:
The best-performing model in this project was XGBRegressor, which achieved an R² score of 0.6424, an MAE of 2491.3619, and an RMSE of 4564.4. After hyperparameter tuning, XGBRegressor showed significant improvement in prediction accuracy. Further improvements can be made by exploring additional feature engineering techniques.

## Contributing:
Contributions are welcome! Feel free to submit a pull request or raise an issue if you have suggestions.

## License:
This project is licensed under the MIT License.


