import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from datetime import datetime, timedelta

def train_and_predict():
    df = pd.read_csv('data/preprocessed_data.csv')
    df['Date'] = pd.to_datetime(df['date'])  
    
    X = df[['high_shifted', 'low_shifted', 'open_shifted', 'volume_shifted', 'marketcap_shifted']]
    y = df['prediction_5D']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    # Hyperparameter tuning
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0]
    }
    
    model = xgb.XGBRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train_scaled)
    best_params = grid_search.best_params_
    
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train_scaled, y_train_scaled)

    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    best_mse = mean_squared_error(y_test, y_pred)
    best_mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error: {best_mse:.2f}")
    print(f"Mean Absolute Error: {best_mae:.2f}")
    
    best_model.save_model('src/model.json')
    joblib.dump(scaler_X, 'src/scaler_X.pkl')
    joblib.dump(scaler_y, 'src/scaler_Y.pkl')
    
    last_features = X.iloc[-5:]
    last_features_scaled = scaler_X.transform(last_features)
    
    pred_xgb_scaled = best_model.predict(last_features_scaled)
    pred_xgb = scaler_y.inverse_transform(pred_xgb_scaled.reshape(-1, 1))
    
    with open('data/last_date.txt', 'r') as f:
        last_date_str = f.read().strip()
    last_date = pd.to_datetime(last_date_str)

    predicted_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
    
    training_predictions = pd.DataFrame({
        'Date': df.loc[X_test.index, 'Date'],  
        'Predicted_Price': y_pred.flatten(),
        'Type': 'Training'
    })
    
    future_predictions = pd.DataFrame({
        'Date': predicted_dates,
        'Predicted_Price': pred_xgb.flatten(),
        'Type': 'Future'
    })
    
    combined_predictions = pd.concat([training_predictions, future_predictions], ignore_index=True)
    combined_predictions.to_csv('data/combined_predictions.csv', index=False)
    print("Combined predictions saved to data/combined_predictions.csv")

    future_predictions_df = combined_predictions[combined_predictions['Type'] == 'Future']
    future_predictions_df.to_csv('data/predictions.txt', index=False)
    print("Future predictions saved to data/predictions.txt")

    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE'],
        'Value': [best_mae, best_mse]
    })
    metrics_df.to_csv('data/evaluation_metrics.csv', index=False)

train_and_predict()
