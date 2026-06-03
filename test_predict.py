import os
import pandas as pd
from f1_data import get_f1_data, F1DataFetcher
from f1_predictor import F1RacePredictor

# -----------------------------------------------------------------------
# 1. Cached fetch — only hits the API if CSV doesn't exist
#    Delete cache/combined_2015_2026.csv to force a refresh
# -----------------------------------------------------------------------
COMBINED_CACHE = './cache/combined_2015_2026.csv'
os.makedirs('./cache', exist_ok=True)

if os.path.exists(COMBINED_CACHE):
    print("Loading combined data from cache...")
    df_all = pd.read_csv(COMBINED_CACHE)
else:
    print("Fetching all data from API (this will take a while)...")
    df_all = get_f1_data(start_year=2015, end_year=2026)
    df_all.to_csv(COMBINED_CACHE, index=False)

# -----------------------------------------------------------------------
# 2. Monaco circuit average from all available data
# -----------------------------------------------------------------------
monaco_circuit_avg = (
    df_all[df_all['circuit_id'] == 'monaco']
    .groupby('driver_id')['final_position']
    .mean()
    .to_dict()
)

monaco_sample_counts = (
    df_all[df_all['circuit_id'] == 'monaco']
    .groupby('driver_id')['final_position']
    .count()
    .to_dict()
)

# For drivers with only 1 Monaco race, blend with the field average
field_avg = 11.0
for driver_id, avg in monaco_circuit_avg.items():
    count = monaco_sample_counts.get(driver_id, 0)
    if count == 1:
        monaco_circuit_avg[driver_id] = (avg + field_avg) / 2

print("\nAdjusted Monaco averages (low sample drivers):")
for driver_id, count in monaco_sample_counts.items():
    if count == 1:
        print(f"  {driver_id}: adjusted to {monaco_circuit_avg[driver_id]:.1f}")

# -----------------------------------------------------------------------
# 3. Get each driver's most recent 2026 feature row (current form)
# -----------------------------------------------------------------------
df_2026 = df_all[df_all['year'] == 2026].copy()
latest = (
    df_2026
    .sort_values(['driver_id', 'year', 'round'])
    .groupby('driver_id')
    .last()
    .reset_index()
)

# -----------------------------------------------------------------------
# 4. Fetch Monaco qualifying grid from API (round 6, 2026)
#    Falls back to placeholder if qualifying hasn't happened yet
# -----------------------------------------------------------------------
MONACO_YEAR  = 2026
MONACO_ROUND = 6

print(f"\nFetching Monaco qualifying results (R{MONACO_ROUND})...")
fetcher = F1DataFetcher()
quali_rows = fetcher.fetch_qualifying(MONACO_YEAR, MONACO_ROUND)

if quali_rows:
    quali_rows.sort(key=lambda x: x['grid_position'])
    
    # Find the highest valid grid position
    max_pos = max((r['grid_position'] for r in quali_rows if r['grid_position'] > 0), default=21)
    
    monaco_grid = {}
    pit_lane_pos = max_pos + 1  # start pit lane drivers after the last grid position
    
    for row in quali_rows:
        if row['grid_position'] > 0:
            monaco_grid[row['driver_id']] = row['grid_position']
        else:
            # grid=0 means pit lane / no time set — start at the back
            monaco_grid[row['driver_id']] = pit_lane_pos
            pit_lane_pos += 1
else:
    print("Qualifying not available yet — using placeholder grid")
    monaco_grid = {
        'max_verstappen':  1,
        'leclerc':         2,
        'norris':          3,
        'russell':         4,
        'hamilton':        5,
        'piastri':         6,
        'sainz':           7,
        'antonelli':       8,
        'hadjar':          9,
        'albon':          10,
        'hulkenberg':     11,
        'ocon':            12,
        'bearman':        13,
        'gasly':          14,
        'lawson':         15,
        'stroll':         16,
        'colapinto':      17,
        'bortoleto':      18,
        'alonso':         19,
        'perez':          20,
        'bottas':         21,
        'lindblad':       22,
    }

# -----------------------------------------------------------------------
# 5. Build driver input list
# -----------------------------------------------------------------------
drivers = []
for driver_id, grid_pos in monaco_grid.items():
    row = latest[latest['driver_id'] == driver_id]
    if row.empty:
        print(f"Warning: no 2026 data for {driver_id}, skipping")
        continue

    r = row.iloc[0]
    drivers.append({
        'driver_name':                  str(r['driver_name']),
        'grid_position':                grid_pos,
        'driver_points_5race_avg':      float(r['driver_points_5race_avg']),
        'driver_points_10race_avg':     float(r['driver_points_10race_avg']),
        'driver_finish_rate_5race':     float(r['driver_finish_rate_5race']),
        'driver_grid_avg_5race':        float(r['driver_grid_avg_5race']),
        'driver_career_wins':           float(r['driver_career_wins']),
        'driver_races_completed':       float(r['driver_races_completed']),
        'constructor_points_5race_avg': float(r['constructor_points_5race_avg']),
        'constructor_dnf_rate_5race':   float(r['constructor_dnf_rate_5race']),
        'driver_circuit_avg_finish':    monaco_circuit_avg.get(driver_id, 15.0),
    })

# -----------------------------------------------------------------------
# 6. Predict
# -----------------------------------------------------------------------
predictor = F1RacePredictor('./models')
result = predictor.predict_race(drivers)

print("\n=== 2026 MONACO GP PREDICTION ===\n")
print(f"  {'Pred':<6} {'Grid':<6} {'Points%':<10} Driver")
print(f"  {'-'*45}")
for p in result['predictions']:
    print(
        f"  P{p['predicted_position']:<5} "
        f"P{p['grid_position']:<5} "
        f"{p['points_probability']:.1%}{'':6}"
        f"{p['driver_name']}"
    )

print("\n--- Predicted Podium ---")
for i, name in enumerate(predictor.predict_podium(drivers), 1):
    print(f"  {i}. {name}")

print("\n--- Predicted Points Finishers ---")
for d in predictor.predict_points_finishers(drivers):
    print(f"  P{d['predicted_position']}  {d['driver_name']}")