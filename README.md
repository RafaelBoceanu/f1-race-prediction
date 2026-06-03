# F1 Race Prediction Model

A machine learning system that predicts Formula 1 race outcomes using historical data from [f1api.dev](https://f1api.dev).

Built as a portfolio project to demonstrate data engineering, feature engineering, and ML model training on a real-world dataset.

---

## What It Does

- Fetches historical F1 race results from 2015–2026 via f1api.dev
- Engineers 10 predictive features per driver per race (form, reliability, circuit history, team strength)
- Trains and compares multiple ML models, selecting the best performer for each target
- Predicts race outcomes for any upcoming Grand Prix using live qualifying data

**Prediction targets:**

| Target | Type | Model | Metric |
|---|---|---|---|
| Points finish (top 10) | Classification | Gradient Boosting | ROC-AUC ~0.89 |
| Finishing position | Regression | Random Forest | MAE ~3.0 positions |

---

## Features Used

| Feature | Description |
|---|---|
| `grid_position` | Starting position from qualifying |
| `driver_points_5race_avg` | Rolling 5-race average points |
| `driver_points_10race_avg` | Rolling 10-race average points |
| `driver_finish_rate_5race` | Finish rate over last 5 races |
| `driver_grid_avg_5race` | Average grid position over last 5 races |
| `driver_career_wins` | Career wins up to this race |
| `driver_races_completed` | Total races completed |
| `constructor_points_5race_avg` | Team rolling 5-race average points |
| `constructor_dnf_rate_5race` | Team rolling 5-race DNF rate |
| `driver_circuit_avg_finish` | Driver's historical average finish at this circuit |

---

## Project Structure

```
f1-race-prediction/
├── f1_data.py          # Data fetching (f1api.dev) and feature engineering
├── f1_models.py        # Model training and evaluation
├── f1_predictor.py     # Inference API
├── test_pipeline.py    # Train models end-to-end
├── test_predict.py     # Predict an upcoming race
├── F1_Prediction_Model.ipynb  # Full analysis notebook
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the models (fetches 2015-2026 data, takes ~30 mins first run)
python test_pipeline.py

# 4. Run a prediction
python test_predict.py
```

> `models/` and `cache/` are excluded from the repo and generated locally by the above commands. Delete `cache/combined_2015_2026.csv` to force a fresh data fetch. Delete `models/*.pkl` to force a retrain.

---

## Usage

### Train models

```python
from f1_data import get_f1_data
from f1_models import train_f1_models

df = get_f1_data(start_year=2015, end_year=2026)
manager = train_f1_models(df)
manager.save_models('./models')
```

### Predict a race

```python
from f1_predictor import F1RacePredictor

predictor = F1RacePredictor('./models')

drivers = [
    {
        'driver_name': 'Lando Norris',
        'grid_position': 1,
        'driver_points_5race_avg': 20.0,
        'driver_points_10race_avg': 18.5,
        'driver_finish_rate_5race': 0.92,
        'driver_grid_avg_5race': 2.5,
        'driver_career_wins': 8,
        'driver_races_completed': 120,
        'constructor_points_5race_avg': 25.0,
        'constructor_dnf_rate_5race': 0.04,
        'driver_circuit_avg_finish': 3.2,
    },
    # ... rest of grid
]

predictions = predictor.predict_race(drivers)
podium = predictor.predict_podium(drivers)
points = predictor.predict_points_finishers(drivers)
```

### Fetch live qualifying grid

```python
from f1_data import F1DataFetcher

fetcher = F1DataFetcher()
quali = fetcher.fetch_qualifying(2026, 6)  # year, round number
```

---

## Example Output

```
=== 2026 MONACO GP PREDICTION ===

  Pred   Grid   Points%    Driver
  ---------------------------------------------
  P1     P4     95.6%      Lando Norris
  P2     P7     84.0%      Oscar Piastri
  P3     P6     97.9%      Lewis Hamilton
  P4     P2     97.3%      Max Verstappen
  P5     P3     82.5%      Charles Leclerc
  P6     P5     74.1%      George Russell
  P7     P11    51.4%      Liam Lawson
  P8     P13    97.3%      Carlos Sainz
  P9     P8     57.7%      Franco Colapinto
  P10    P12    54.1%      Oliver Bearman

--- Predicted Podium ---
  1. Lando Norris
  2. Oscar Piastri
  3. Lewis Hamilton
```

---

## Data Source

[f1api.dev](https://f1api.dev) — free, open source F1 data API. Replaces the deprecated Ergast API which was retired at the end of the 2024 season.

---

## Tech Stack

Python · pandas · scikit-learn · XGBoost · f1api.dev · Jupyter

---

## Notes

- f1api.dev rate limits requests — the full data fetch includes 300ms pauses and takes approximately 20–30 minutes
- The model is trained on data from 2015–2026 with recent seasons weighted more heavily (2026: 2x, 2025: 1.5x) to account for regulation changes
- Circuit average finishing positions for drivers with only one appearance at a circuit are blended with the field average to reduce noise
- Monaco and other street circuits are inherently harder to predict due to limited overtaking — grid position is a stronger signal at these venues than the model currently reflects

---

## License

MIT