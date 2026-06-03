import json
import numpy as np
import pandas as pd
from typing import Dict, List

from f1_models import F1PredictionModel, FEATURE_COLS

class F1RacePredictor:

    def __init__(self, model_path: str = './models'):
        self.manager = F1PredictionModel()
        self.manager.load_models(model_path)

    def predict_race(self, drivers_data: List[Dict]) -> Dict:
        predictions = []

        for driver in drivers_data:
            features = np.array([driver.get(col, 0.0) for col in FEATURE_COLS], dtype=float)

            points_pred = self.manager.predict_points_finish(features)
            position_pred = self.manager.predict_position(features)

            predictions.append({
                "driver_name":        driver.get("driver_name", "Unknown"),
                "grid_position":      driver.get("grid_position"),
                "predicted_position": position_pred["predicted_position"],
                "will_score_points":  points_pred["will_finish_points"],
                "points_probability": round(points_pred["probability"], 4),
                "confidence":         round(points_pred["confidence"], 4),
                '_raw_position':      position_pred['predicted_position'],
            })

        predictions.sort(key=lambda x: x['_raw_position'])

        for i, pred in enumerate(predictions, 1):
            pred['predicted_position'] = i
            del pred['_raw_position']

        return {
            'predictions': predictions,
            'timestamp':   pd.Timestamp.now().isoformat(),
        }
    
    def predict_podium(self, drivers_data: List[Dict]) -> List[str]:
        result = self.predict_race(drivers_data)
        return [d["driver_name"] for d in result["predictions"][:3]]
    
    def predict_points_finishers(self, drivers_data: List[Dict]) -> List[Dict]:
        result = self.predict_race(drivers_data)
        return [d for d in result["predictions"] if d["predicted_position"] <= 10]

    def export_predictions(self, predictions: Dict, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indend=2, default=str)
        print(f"Predictions saved to {filepath}")

def _example_grid() -> List[Dict]:
    return [
        {
            'driver_name': 'Max Verstappen',
            'grid_position': 1,
            'driver_points_5race_avg': 22.4,
            'driver_points_10race_avg': 21.8,
            'driver_finish_rate_5race': 0.95,
            'driver_grid_avg_5race': 1.8,
            'driver_career_wins': 45,
            'driver_races_completed': 200,
            'constructor_points_5race_avg': 28.5,
            'constructor_dnf_rate_5race': 0.03,
            'driver_circuit_avg_finish': 1.2,
        },
        {
            'driver_name': 'Lando Norris',
            'grid_position': 2,
            'driver_points_5race_avg': 15.2,
            'driver_points_10race_avg': 14.5,
            'driver_finish_rate_5race': 0.88,
            'driver_grid_avg_5race': 3.2,
            'driver_career_wins': 2,
            'driver_races_completed': 150,
            'constructor_points_5race_avg': 24.3,
            'constructor_dnf_rate_5race': 0.05,
            'driver_circuit_avg_finish': 2.8,
        },
        {
            'driver_name': 'Lewis Hamilton',
            'grid_position': 3,
            'driver_points_5race_avg': 12.1,
            'driver_points_10race_avg': 11.8,
            'driver_finish_rate_5race': 0.82,
            'driver_grid_avg_5race': 4.5,
            'driver_career_wins': 103,
            'driver_races_completed': 350,
            'constructor_points_5race_avg': 14.2,
            'constructor_dnf_rate_5race': 0.08,
            'driver_circuit_avg_finish': 3.5,
        },
        {
            'driver_name': 'Charles Leclerc',
            'grid_position': 5,
            'driver_points_5race_avg': 10.3,
            'driver_points_10race_avg': 9.7,
            'driver_finish_rate_5race': 0.75,
            'driver_grid_avg_5race': 5.8,
            'driver_career_wins': 5,
            'driver_races_completed': 120,
            'constructor_points_5race_avg': 16.1,
            'constructor_dnf_rate_5race': 0.12,
            'driver_circuit_avg_finish': 4.2,
        },
    ] 
 
if __name__ == '__main__':
    predictor = F1RacePredictor("./models")
    drivers   = _example_grid()
    result    = predictor.predict_race(drivers)

    print("\n=== RACE PREDICTIONS ===\n")
    for p in result["predictions"]:
        print(
            f"  Grid {p['grid_position']:2d} -> Pos {p['predicted_position']:2d}    "
            f"{p['driver_name']:<22s}   "
            f"Points: {p['points_probability']:5.1%%}   "
            f"Conf: {p['confidence']:.1%}"
        )
    
    print("\n=== PREDICTED PODIUM ===")
    for i, name in enumerate(predictor.predict_podium(drivers), 1):
        print(f"    {i}. {name}")
    
    print("\n=== POINTS FINISHERS ===")
    for d in predictor.predict_points_finishers(drivers):
        print(f"    {d['driver_name']:<22s} (P{d['predicted_position']})")

    predictor.export_predictions(result, "./example.predictions.json")