import numpy as np
import pandas as pd
from f1_models import F1PredictionModel
from typing import Dict, List
import json

class F1RacePredictor:

    def __init__(self, model_path: str = './models'):
        self.model_manager = F1PredictionModel()
        self.model_manager.load_models(model_path)

    def predict_race(self, drivers_data: List[Dict]) -> Dict:
        predictions = []

        for driver in drivers_data:
            features = np.array([
                driver['grid_position'],
                driver['driver_points_5race_avg'],
                driver['driver_points_10race_avg'],
                driver['driver_finish_rate_5race'],
                driver['driver_grid_avg_5race'],
                driver['driver_career_wins'],
                driver['driver_races_completed'],
                driver['constructor_points_5race_avg'],
                driver['constructor_dnf_rate_5race'],
                driver['driver_circuit_avg_finish'],
            ])

            points_pred = self.model_manager.predict_points_finish(features)
            position_pred = self.model_manager.predict_position(features)

            predictions.append({
                'driver_name': driver['driver_name'],
                'grid_position': driver['grid_position'],
                'predicted_position': position_pred['predicted_position'],
                'will_score_points': points_pred['will_finish_points'],
                'points_probability': points_pred['probability'],
                'confidence': points_pred['confidence'],
            })

        predictions = sorted(predictions, key=lambda x: x['predicted_position'])

        return {
            'predictions': predictions,
            'timestamp': pd.Timestamp.now().isoformat(),
        }
    
    def predict_podium(self, drivers_data: List[Dict]) -> List[str]:
        race_pred = self.predict_race(drivers_data)
        top_3 = race_pred['predictions'][:3]
        return [driver['driver_name'] for driver in top_3]
    
    def predict_points_finishers(self, drivers_data: List[Dict]) -> List[Dict]:
        race_pred = self.predict_race(drivers_data)
        points_finishers = [
            d for d in race_pred['predictions']
            if d['will_score_points']
        ]
        return points_finishers

    def export_predictions(self, predictions: Dict, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indend=2, default=str)
        print(f"Predictions saved to {filepath}")

def example_prediction():
    predictor = F1RacePredictor()

    drivers = [
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

    predictions = predictor.predict_race(drivers)

    print("\n=== RACE PREDICTIONS ===\n")
    for pred in predictions['predictions']:
        print(f"{pred['grid_position']:2d} → {pred['predicted_position']:2d}  "
              f"{pred['driver_name']:20s}  "
              f"Points: {pred['points_probability']:5.1%}  "
              f"Confidence: {pred['confidence']:.1%}")
    
    print("\n=== PREDICTED PODIUM ===")
    podium = predictor.predict_podium(drivers)
    for i, driver in enumerate(podium, 1):
        print(f"{i}. {driver}")
    
    print("\n=== POINTS FINISHERS ===")
    finishers = predictor.predict_points_finishers(drivers)
    for driver in finishers:
        print(f"  {driver['driver_name']:20s} (Pos {driver['predicted_position']})")
    
    predictor.export_predictions(predictions, './example_predictions.json')
 
 
if __name__ == '__main__':
    example_prediction()