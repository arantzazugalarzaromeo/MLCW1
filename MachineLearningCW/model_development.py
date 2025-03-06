

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

import shap
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_split

# Loading the data set after being preprocessed
train = pd.read_csv('CW1_train_encoded.csv')
val = pd.read_csv('CW1_val_encoded.csv')

# Handling Missing Values
imputer = SimpleImputer(strategy="mean")

if 'outcome' in train.columns:
    X_train = train.drop(columns=['outcome'])
    y_train = train['outcome']
else:
    raise KeyError("Column 'outcome' is missing in training data!")

if 'outcome' in val.columns:
    X_val = val.drop(columns=['outcome'])
    y_val = val['outcome']
else:
    X_val = val.copy()  
    y_val = None

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

# Training regression models 
easy_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

results_easy = {}
print("\nTraining regression models")
for name, model in easy_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    results_easy[name] = r2
    print(f" {name}: R² Score = {r2:.4f}")

# Training other models
medium_models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=123),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=123),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=123),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=123),
    "CatBoost": CatBoostRegressor(verbose=0, iterations=100, learning_rate=0.1, random_state=123)
}

results_medium = {}
print("\n Training other Models")
for name, model in medium_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    results_medium[name] = r2
    print(f" {name}: R² Score = {r2:.4f}")

# Selecting Best Model
all_results = {**results_easy, **results_medium}
best_model_name = max(all_results, key=all_results.get)
best_model = easy_models.get(best_model_name, medium_models.get(best_model_name))
print(f"\n Best Model: {best_model_name} with R² = {all_results[best_model_name]:.4f}")

if best_model_name == "CatBoost":
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 1000, 4000),  
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),  
            "depth": trial.suggest_int("depth", 4, 12), 
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 20),  
            "border_count": trial.suggest_int("border_count", 32, 200),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "random_strength": trial.suggest_float("random_strength", 0, 2),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide", "Depthwise"]),
            "random_state": 123,
            "verbose": 0
        }

        model = CatBoostRegressor(**params)
        r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
        return r2

    # Optimizing Hyperparameters
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=70)  

    # Get Best Parameters
    best_params = study.best_params
    print("Best Hyperparameters Found:", best_params)

    # Train the Best Model with Optimized Parameters
    best_model = CatBoostRegressor(
        **best_params,  
        early_stopping_rounds=300,  
        verbose=100,
        random_state=123
    )

    best_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=300)

    # Final Model Evaluation
    yhat_best = best_model.predict(X_val)
    final_r2 = r2_score(y_val, yhat_best)
    mse = mean_squared_error(y_val, yhat_best)
    rmse = np.sqrt(mse)

    print(f" Final Fine-Tuned CatBoost Model R² Score: {final_r2:.4f}")
    print(f" Final MSE: {mse:.4f}")
    print(f" Final RMSE: {rmse:.4f}")



# Load the preprocessed test dataset (for the predictions)
df_test = pd.read_csv('CW1_test_encoded.csv')

#  Generate Final Predictions using the best trained model
test_predictions = best_model.predict(df_test)

# submission file of the predictions of the test csv 
submission = pd.DataFrame({
    "yhat": test_predictions  
})

#  Save to CSV 
submission_filename = "CW1_submission_K23101478.csv"  
submission.to_csv(submission_filename, index=False)

print(f" Final test predictions saved as '{submission_filename}'")

