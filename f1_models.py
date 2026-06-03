import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_absolute_error
)
from xgboost import XGBClassifier, XGBRegressor

FEATURE_COLS = [
    "grid_position",
    "driver_points_5race_avg",
    "driver_points_10race_avg",
    "driver_finish_rate_5race",
    "driver_grid_avg_5race",
    "driver_career_wins",
    "driver_races_completed",
    "constructor_points_5race_avg",
    "constructor_dnf_rate_5race",
    "driver_circuit_avg_finish",
]

class F1PredictionModel:

    def __init__(self):
        self.models: Dict = {}
        self.scalers: Dict = {}
        self.results: Dict = {}

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:

        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        return df[FEATURE_COLS].fillna(0).values

    def train_points_finish_model(
        self,
        X_train, X_test,
        y_train, y_test,
        sample_weight=None,
    ) -> None:
        print("\n=== Points Finish Classifier ===")

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        self.scalers["points_finish"] = scaler

        candidates = {
            "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
            "XGBoost":              XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        }

        best_model, best_name, best_score = None, "", 0.0

        for name, model in candidates.items():
            model.fit(X_tr, y_train, sample_weight=sample_weight)
            y_pred  = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:, 1]

            acc     = accuracy_score(y_test, y_pred)
            f1      = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            print(f"  {name:25s}  acc={acc:.3f}  f1={f1:.3f}  roc_auc={roc_auc:.3f}")

            if roc_auc > best_score:
                best_score, best_model, best_name = roc_auc, model, name

        print(f"\n  → Best: {best_name} (ROC-AUC {best_score:.4f})")
        self.models["points_finish"] = best_model
        self.results["points_finish"] = {"model": best_name, "score": best_score, "metric": "ROC-AUC"}
    
    def train_position_model(
        self,
        X_train, X_test,
        y_train, y_test,
        sample_weight=None,
    ) -> None:
        print("\n=== Position Regressor ===")

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        self.scalers["position"] = scaler

        candidates = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "XGBoost":       XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        }

        best_model, best_name, best_mae = None, "", float("inf")

        for name, model in candidates.items():
            model.fit(X_tr, y_train, sample_weight=sample_weight)
            y_pred = model.predict(X_te)
            mae    = mean_absolute_error(y_test, y_pred)
            rmse   = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

            print(f"  {name:25s}  MAE={mae:.3f}  RMSE={rmse:.3f}")

            if mae < best_mae:
                best_mae, best_model, best_name = mae, model, name

        print(f"\n  → Best: {best_name} (MAE {best_mae:.4f})")
        self.models["position"] = best_model
        self.results["position"] = {"model": best_name, "score": best_mae, "metric": "MAE"}
    
    def predict_points_finish(self, features: np.ndarray) -> Dict:
        model  = self.models["points_finish"]
        scaler = self.scalers["points_finish"]
        X      = scaler.transform(features.reshape(1, -1))
        pred   = bool(model.predict(X)[0])
        prob   = float(model.predict_proba(X)[0][1])
        return {
            'will_finish_points': pred,
            'probability': prob,
            'confidence': max(prob, 1 - prob)
        }
    
    def predict_position(self, features: np.ndarray) -> Dict:
        model  = self.models["position"]
        scaler = self.scalers["position"]
        X      = scaler.transform(features.reshape(1, -1))
        raw    = float(model.predict(X)[0])
        pos    = int(max(1, min(20, round(raw))))
        return {
            "predicted_position": pos,
        }
    
    def save_models(self, path: str = './models') -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
             with open(f"{path}/{name}_model.pkl", "wb") as f:
                pickle.dump(model, f)
        for name, scaler in self.scalers.items():
            with open(f"{path}/{name}_scaler_model.pkl", "wb") as f:
                pickle.dump(scaler, f)

    def load_models(self, path: str = './models'):
        for target in ("points_finish", "position"):
            model_path  = f"{path}/{target}_model.pkl"
            scaler_path = f"{path}/{target}_scaler_model.pkl"
            try:
                with open(model_path, 'rb') as f:
                    self.models[target] = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scalers[target] = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: could not load {target} model from {path}/")

def train_f1_models(df: pd.DataFrame) -> F1PredictionModel:
    manager = F1PredictionModel()
    X = manager.prepare_features(df)

    y_points   = df["points_finish"].values
    y_position = df["position_target"].values

    # Weight recent seasons more heavily
    # 2026 = 3x, 2025 = 2x, everything else = 1x
    weights = np.where(df['year'] == 2026, 2.0,
              np.where(df['year'] == 2025, 1.5, 1.0))

    X_train, X_test, yp_train, yp_test, yr_train, yr_test, w_train, w_test = train_test_split(
        X, y_points, y_position, weights,
        test_size=0.2,
        random_state=42,
        stratify=y_points,
    )

    print(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")
    print(f"Points finish rate (train): {yp_train.mean():.2%}")

    manager.train_points_finish_model(X_train, X_test, yp_train, yp_test, w_train)
    manager.train_position_model(X_train, X_test, yr_train, yr_test, w_train)

    return manager


if __name__ == "__main__":
    from f1_data import get_f1_data
    df = get_f1_data(start_year=2020, end_year=2024)
    manager = train_f1_models(df)
    manager.save_models("./models")
    print("\nResults:", manager.results)