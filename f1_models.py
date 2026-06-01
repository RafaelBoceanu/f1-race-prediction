import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import pickle
from pathlib import Path
from typing import Tuple

class F1PredictionModel:

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        self.results = {}

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, list]:
        feature_cols = [
            'grid_position',
            'driver_points_5race_avg',
            'driver_points_10race_avg',
            'driver_finish_rate_5race',
            'driver_grid_avg_5race',
            'driver_career_wins',
            'driver_races_completed',
            'constructor_points_5race_avg',
            'constructor_dnf_rate_5race',
            'driver_circuit_avg_finish',
        ]

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        self.feature_names = feature_cols

        X = df[feature_cols].fillna(0).values

        return X, feature_cols

    def train_points_finish_model(self, X_train, X_test, y_train, y_test):
        print("\n=== Training Points Finish Prediction Model ===")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['points_finish'] = scaler
        

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        }

        best_model = None
        best_score = 0

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            print(f"\n{name}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC:  {roc_auc:.4f}")

            if roc_auc > best_score:
                best_score = roc_auc
                best_model = model
                best_model_name = name

        print(f"\nBest Model: {best_model_name} (ROC-AUC: {best_score:.4f})")
        self.models['points_finish'] = best_model
        self.results['points_finish'] = {
            'model': best_model_name,
            'best-score': best_score
        }

        return best_model, scaler
    
    def train_position_model(self, X_train, X_test, y_train, y_test):
        print("\n=== Training Position Prediction Model ===")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['position'] = scaler

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        }

        best_model = None
        best_score = float('inf')

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            print(f"\n{name}:")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")

            if mae < best_score:
                best_score = mae
                best_model = model
                best_model_name = name

        print(f"\nBest Model: {best_model_name} (MAE: {best_score:.4f})")
        self.models['position'] = best_model
        self.results['position'] = {
            'model': best_model_name,
            'best-score': best_score
        }

        return best_model, scaler
    
    def predict_points_finish(self, driver_features: np.ndarray) -> dict:
        if 'points_finish' not in self.models:
            raise ValueError("Points Finish model not trained yet.")
        
        model = self.models['points_finish']
        scaler = self.scalers['points_finish']

        X_scaled = scaler.transform(driver_features.reshape(1, -1))
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        return {
            'will_finish_points': bool(prediction),
            'probability': float(probability),
            'confidence': max(probability, 1 - probability)
        }
    
    def predict_position(self, driver_features: np.ndarray) -> dict:
        if 'position' not in self.models:
            raise ValueError("Position model not trained yet.")
        
        model = self.models['position']
        scaler = self.scalers['position']

        X_scaled = scaler.transform(driver_features.reshape(1, -1))
        predicted_position = max(1, min(20, int(round(model.predict(X_scaled)[0]))))

        return {
            'predicted_position': int(round(predicted_position)),
        }
    
    def save_models(self, path: str = './models'):
        Path(path).mkdir(exist_ok=True)

        for name, model in self.models.items():
            with open(f"{path}/{name}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        for name, scaler in self.scalers.items():
            with open(f"{path}/{name}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)

        print(f"Models saved to {path}/")

    def load_models(self, path: str = './models'):
        for model_type in ['points_finish', 'position']:
            try:
                with open(f"{path}/{model_type}_model.pkl", 'rb') as f:
                    self.models[model_type] = pickle.load(f)
                with open(f"{path}/{model_type}_scaler.pkl", 'rb') as f:
                    self.scalers[model_type] = pickle.load(f)
            except FileNotFoundError:
                print(f"Model {model_type} not found at {path}/")

def train_f1_models(df: pd.DataFrame) -> F1PredictionModel:
    print("Preparing features...")

    model_manager = F1PredictionModel()
    X, feature_cols = model_manager.prepare_features(df)

    y_points = df['points_finish'].values
    y_position = df['position_target'].values

    X_train, X_test, y_points_train, y_points_test, y_pos_train, y_pos_test = train_test_split(
        X, y_points, y_position,
        test_size=0.2,
        random_state=42,
        stratify=y_points
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    model_manager.train_points_finish_model(X_train, X_test, y_points_train, y_points_test)
    model_manager.train_position_model(X_train, X_test, y_pos_train, y_pos_test)

    return model_manager