import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import time

BASE_URL = "https://f1api.dev/api"

class F1DataFetcher:

    def __init__(self, start_year: int = 2015, end_year: int = 2025):
        self.start_year = start_year
        self.end_year = end_year
        self.rounds_per_year = {
            2015: 19, 2016: 21, 2017: 20, 2018: 21, 2019: 21,
            2020: 17, 2021: 22, 2022: 22, 2023: 22, 2024: 24,
            2025: 24, 2026: 24,
        }

    def _get(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        url = f"{BASE_URL}/{path}"
        try:
            response = requests.get(url, params=params or {}, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # 404 normal for rounds that don't exist yet
            if response.status_code != 404:
                print(f"  HTTP error {response.status_code} for {url}: {e}")
            return None
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            return None

    def fetch_race_results(self, year: int, round_num: int) -> List[Dict]:
        data = self._get(f"{year}/{round_num}/race", {"limit": 30})
        if not data or "races" not in data:
            return []

        race_meta = data["races"]
        circuit_data = race_meta.get("circuit", [])

        if isinstance(circuit_data, list) and len(circuit_data) > 0:
            circuit = circuit_data[0]
        else:
            circuit = {}

        circuit_id = circuit.get("circuitId", "")
        circuit_name = circuit.get("circuitName", "")

        rows = []
        for result in race_meta.get("results", []):
            driver = result.get("driver", {})
            team = result.get("team", {})

            # Position: convert to int; DNFs get 999
            raw_pos = result.get("position")
            try:
                final_position = int(raw_pos)
            except (TypeError, ValueError):
                final_position = 999
            
            # DNF flag: retired field is a string like "Engine" or None
            position = result.get("position")
            is_dnf = 1 if position == "NC" else 0

            # Grid position
            raw_grid = result.get("grid", 0)
            try:
                grid_position = int(raw_grid)
            except (TypeError, ValueError):
                grid_position = 0

            rows.append({
                "year":             year,
                "round":            round_num,
                "race_id":          race_meta.get("raceId", ""),
                "race_name":        race_meta.get("raceName", ""),
                "circuit_id":       circuit_id,
                "circuit_name":     circuit_name,
                "date":             race_meta.get("date", ""),
                "driver_id":        driver.get("driverId", ""),
                "driver_name":      f"{driver.get('name', '')} {driver.get('surname', '')}".strip(),
                "constructor_id":   team.get("teamId", ""),
                "constructor_name": team.get("teamName", ""),
                "grid_position":    grid_position,
                "final_position":   final_position,
                "points":           float(result.get("points", 0)),
                "dnf":              is_dnf,
            })

        time.sleep(0.3)
        return rows
        
    def fetch_qualifying(self, year: int, round_num: int) -> List[Dict]:
        data = self._get(f"{year}/{round_num}/qualy", {"limit": 30})
        if not data or "races" not in data:
            return []
        
        rows = []
        for entry in data["races"].get("qualyResults", []):
            driver = entry.get("driver", {})
            team = entry.get("team", {})

            raw_grid = entry.get("gridPosition", 0)
            try:
                grid_pos = int(raw_grid)
            except (TypeError, ValueError):
                grid_pos = 0

            rows.append({
                "year":           year,
                "round":          round_num,
                "driver_id":      driver.get("driverId", entry.get("driverId", "")),
                "constructor_id": team.get("teamId", entry.get("teamId", "")),
                "grid_position":  grid_pos,
                "q1":             entry.get("q1"),
                "q2":             entry.get("q2"),
                "q3":             entry.get("q3"),
            })

        time.sleep(0.3)
        return rows
        
    def fetch_driver_standings(self, year: int) -> pd.DataFrame:
        data = self._get(f"{year}/driver-championship", {"limit": 30})
        if not data or "drivers-championship" not in data:
            return pd.DataFrame()

        rows = []
        for entry in data["drivers-championship"]:
            driver = entry.get("driver", {})
            rows.append({
                "year": year,
                "driver_id":      entry.get("driverId", ""),
                "driver_name":    f"{driver.get('name', '')} {driver.get('surname', '')}".strip(),
                "constructor_id": entry.get("teamId", ""),
                "position":       entry.get("position"),
                "season_points":  float(entry.get("points", 0)),
                "season_wins":    int(entry.get("wins", 0)),
            })

        time.sleep(0.3)
        return pd.DataFrame(rows)
        
    def build_race_dataset(self) -> pd.DataFrame:
        all_rows = []

        for year in range(self.start_year, self.end_year + 1):
            max_rounds = self.rounds_per_year.get(year, 23)
            print(f"\n{year} ({max_rounds} rounds)")

            for round_num in range(1, max_rounds + 1):
                print(f"  R{round_num:02d}...", end=" ", flush=True)
                rows = self.fetch_race_results(year, round_num)
                if rows:
                    all_rows.extend(rows)
                    print(f"({len(rows)} drivers)")
                else:
                    print("---")
            
        df = pd.DataFrame(all_rows)
        print(f"\nTotal rows fetched: {len(df)}")
        return df
        
class FeatureEngineer:
    @staticmethod
    def add_driver_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['driver_id', 'year', 'round']).reset_index(drop=True)

        grp = df.groupby("driver_id")

        # Feature 1: 5-race rolling average of points
        df["driver_points_5race_avg"] = (
            grp["points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

        # Feature 2: 10-race rolling average of points
        df["driver_points_10race_avg"] = (
            grp["points"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        )

        # Feature 3: 5-race finish rate
        df["driver_finish_rate_5race"] = (
            grp["dnf"].transform(lambda s: 1 - s.shift(1).rolling(5, min_periods=1).mean())
        )

        # Feature 4: Average grid position (5 races)
        df["driver_grid_avg_5race"] = (
            grp["grid_position"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

        # Feature 5: Career wins (cumulative count)
        df["_win"] = (df["final_position"] == 1).astype(int)
        df["driver_career_wins"] = grp["_win"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
        df.drop(columns=["_win"], inplace=True)

        # Feature 6: Total races completed by driver
        df['driver_races_completed'] = grp.cumcount()

        # Fill NaN values with 0 (first race will have no history)
        return df.fillna(0)
    
    @staticmethod
    def add_constructor_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['constructor_id', 'year', 'round']).reset_index(drop=True)
        grp = df.groupby("constructor_id")

        # Feature 7: Team's 5-race average points
        df["constructor_points_5race_avg"] = (
           grp["points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

        # Feature 8: Team's 5-race DNF rate
        df["constructor_dnf_rate_5race"] = (
            grp["dnf"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

        return df.fillna(0)
    
    @staticmethod
    def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:

        # Calculate driver's historical average finish position at each circuit
        circuit_avg = (
            df.groupby(["circuit_id", "driver_id"])["final_position"]
            .mean()
            .reset_index()
            .rename(columns={"final_position": "driver_circuit_avg_finish"})
        )
        # Merge back to main dataframe
        df = df.merge(circuit_avg, on=["circuit_id", "driver_id"], how="left")
        # For drivers with no history at a circuit, use default value 15 (mid-field)
        df["driver_circuit_avg_finish"] = df["driver_circuit_avg_finish"].fillna(15)
        return df
    
    @staticmethod
    def create_model_dataset(race_df: pd.DataFrame) -> pd.DataFrame:
        df = race_df.copy()

        print("Engineering driver features...")
        df = FeatureEngineer.add_driver_features(df)

        print("Engineering constructor features...")
        df = FeatureEngineer.add_constructor_features(df)

        print("Engineering circuit features...")
        df = FeatureEngineer.add_circuit_features(df)

        # Only keep rows where driver has completed at least 1 race
        df = df[df["driver_races_completed"] > 0].copy()

        # TARGET 1: Classification target (Will driver score points?)
        df["points_finish"] = (df["final_position"] <= 10).astype(int)
        # TARGET 2: Regression target (What position will driver finish?)
        df["position_target"] = df["final_position"].clip(upper=20)

        print(f"Final dataset: {len(df)} rows, {df['points_finish'].sum()} points finishes")
        return df
    
def get_f1_data(start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
    print("="*60)
    print(f"F1 DATA PIPELINE ({start_year}-{end_year})")
    print("="*60)

    # Stage 1: Fetch raw data
    fetcher = F1DataFetcher(start_year=start_year, end_year=end_year)
    race_results = fetcher.build_race_dataset()

    if race_results.empty:
        raise RuntimeError("No data fetched - check API connectivity")

    print(f"\nFetched {len(race_results)} race result rows")
    print(f"  Drivers:      {race_results['driver_id'].nunique()}")
    print(f"  Constructors: {race_results['constructor_id'].nunique()}")

    # Stage 2: Engineer features
    print(f"\nEngineering features...")
    df = FeatureEngineer.create_model_dataset(race_results)

    print(f"\n Created dataset with {len(df)} samples")
    print(f"  {df['points_finish'].sum()} points finishes, {1-df['points_finish'].sum()} non-points")
    print(f"  Average grid position: {df['grid_position'].mean():.2f}")
    print(f"  Average final position: {df['final_position'].mean():.2f}")
    print("="*60)

    return df

if __name__ == "__main__":
    # Example: Run the data pipeline
    df = get_f1_data(start_year=2015, end_year=2025)

    print("\nSample:")
    print(df[[
        "year", "round", "driver_name", "grid_position",
        "driver_points_5race_avg", "points_finish",
    ]].head(10))