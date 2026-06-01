import requests
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import time

class F1DataFetcher:
    BASE_URL = "https://ergast.com/api/f1"

    def __init__(self, start_year: int = 2015, end_year: int = 2025):
        self.start_year = start_year
        self.end_year = end_year
        self.races_data = []
        self.results_data= []
        self.driver_standings = {}
        self.constructor_standings = {}

    def fetch_races(self) -> pd.DataFrame:
        all_races = []

        # Loop through each year in the range
        for year in range(self.start_year, self.end_year + 1):
            url = f"{self.BASE_URL}/{year}.json"

            try:
                response = requests.get(url)
                response.raise_for_status() # Raise error if status != 200

                data = response.json()

                races = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])

                for race in races:
                    all_races.append({
                        'raceId': race.get('season'),
                        'year': int(race.get('season')),
                        'round': int(race.get('round')),
                        'name': race.get('name'),
                        'circuit': race.get('Circuit', {}).get('circuitId'),
                        'date': race.get('date'),
                        'lat': float(race.get('Circuit', {}).get('Location', {}).get('lat', 0)),
                        'lng': float(race.get('Circuit', {}).get('Location', {}).get('long', 0)),
                    })

                    print(f"Fetched {len(races)} races from {year}")

                    time.sleep(0.5) # Sleep to avoid hitting API rate limits

            except Exception as e:
                print(f"Error fetching races for {year}: {e}")

        return pd.DataFrame(all_races)
    
    def fetch_results(self, year: int, round_num: int) -> List[Dict]:
        url = f"{self.BASE_URL}/{year}/{round_num}/results.json"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status() # Raise error if status != 200
            data = response.json()

            results = data.get('MRData', {}).get('RaceTable', {}).get('Races', [{}])[0].get('Results', [])

            time.sleep(0.3)
            return results
        
        except Exception as e:
            print(f"Error fetching results for {year} round {round_num}: {e}")
            return []
        
    def fetch_qualifying(self, year: int, round_num: int) -> List[Dict]:
        url = f"{self.BASE_URL}/{year}/{round_num}/qualifying.json"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status() # Raise error if status != 200
            data = response.json()

            results = data.get('MRData', {}).get('RaceTable', {}).get('Races', [{}])[0].get('QualifyingResults', [])

            time.sleep(0.3)
            return results
        
        except Exception as e:
            print(f"Error fetching qualifying for {year} round {round_num}: {e}")
            return []
        
    def fetch_driver_standings(self, year: int) -> pd.DataFrame:
        url = f"{self.BASE_URL}/{year}/driverStandings.json"

        try: 
            response = requests.get(url, timeout=10)
            response.raise_for_status() # Raise error if status != 200
            data = response.json()

            standings = data.get('MRData', {}).get('StandingsTable', {}).get('StandingsLists', [{}])[0].get('DriverStandings', [])
            
            standings_list = []
            for standing in standings:
                driver = standing.get('Driver', {})
                standings_list.append({
                    'driverId': driver.get('driverId'),
                    'position': int(standing.get('position')),
                    'points': float(standing.get('points')),
                    'wins': int(standing.get('wins')),
                })

            time.sleep(0.3)
            return pd.DataFrame(standings_list)
        
        except Exception as e:
            print(f"Error fetching driver standings for {year}: {e}")
            return pd.DataFrame()
        
    def build_race_dataset(self) -> pd.DataFrame:
        print("Fetching all races...")
        races_df = self.fetch_races() # Get list of all races

        all_results = []

        for _, race in races_df.iterrows():
            year = int(race['year'])
            round_num = int(race['round'])

            print(f"Fetching results for {year} R{round_num}...", end=' ')

            results = self.fetch_results(year, round_num)
            qualifying = self.fetch_qualifying(year, round_num)

            quali_pos = {}
            for q_result in qualifying:
                quali_pos[q_result('Driver', {}).get('driverId')] = int(q_result.get('position', 0))

            for result in results:
                driver = result.get('Driver', {})
                constructor = result.get('Constructor', {})

                # Determine if DNF
                # Status can be "Finished", "+1 Lap", "Crashed", "Engine", etc.

                status = result.get('status', '')
                is_dnf = 1 if status not in ['Finished', '+1 Lap', '+2 Laps', '+3 Laps'] else 0

                all_results.append({
                    'year': year,
                    'round': round_num,
                    'race_name': race['name'],
                    'circuit': race['circuit'],
                    'driverId': driver.get('driverId'),
                    'driverName': f"{driver.get('givenName', '')} {driver.get('familyName', '')}",
                    'constructorId': constructor.get('constructorId'),
                    'constructorName': constructor.get('name'),
                    'grid_position': int(result.get('grid', 0)) or 0,
                    'final_position': int(result.get('position', 999)) if result.get('position') != '\\N' else 999,
                    'points': float(result.get('points', 0)),
                    'dnf': is_dnf,
                })

                print (f"✓")

            print(f"\nCreated dataset with {len(all_results)} rows (driver-race combinations)")
            return pd.DataFrame(all_results)
        
class FeatureEngineer:
    @staticmethod
    def add_driver_history_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['driverId', 'year', 'round']).reset_index(drop=True)

        # Feature 1: 5-race rolling average of points
        df['driver_points_5race_avg'] = (
            df.groupby('driverId')['points']
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Feature 2: 10-race rolling average of points
        df['driver_points_10race_avg'] = (
            df.groupby('driverId')['points']
            .shift(1)
            .rolling(window=10, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Feature 3: 5-race finish rate
        df['driver_finish_rate_5race'] = (
            1 - df.groupby('driverId')['dnf']
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Feature 4: Average grid position (5 races)
        df['driver_grid_avg_5race'] = (
            df.groupby('driverId')['grid_position']
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Feature 5: Career wins (cumulative count)
        df['driver_career_wins'] = (
            df.groupby('driverId')
            .apply(lambda x: (x['final_position'] == 1).cumsum().shift(1))
            .reset_index(0, drop=True)
        )

        # Feature 6: Total races completed by driver
        df['driver_races_completed'] = df.groupby('driverId').cumcount()

        # Fill NaN values with 0 (first race will have no history)
        return df.fillna(0)
    
    @staticmethod
    def add_constructor_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['constructorId', 'year', 'round']).reset_index(drop=True)

        # Feature 7: Team's 5-race average points
        df['constructor_points_5race_avg'] = (
            df.groupby('constructorId')['points']
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Feature 8: Team's 5-race DNF rate
        df['constructor_dnf_rate_5race'] = (
            df.groupby('constructorId')['dnf']
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        return df.fillna(0)
    
    @staticmethod
    def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:

        # Calculate driver's historical average finish position at each circuit
        circuit_driver_avg = (
            df.groupby(['circuit', 'driverId'])['final_position']
            .mean()
            .reset_index()
        )
        circuit_driver_avg.columns = ['circuit', 'driverId', 'driver_circuit_avg_finish']

        # Merge back to main dataframe
        df = df.merge(circuit_driver_avg, on=['circuit', 'driverId'], how='left')

        # For drivers with no history at a circuit, use default value 15 (mid-field)
        df['driver_circuit_avg_finish'] = df['driver_circuit_avg_finish'].fillna(15)

        return df
    
    @staticmethod
    def create_model_dataset(race_results_df: pd.DataFrame) -> pd.DataFrame:
        df = race_results_df.copy()

        print("Adding driver history features...")
        df = FeatureEngineer.add_driver_history_features(df)

        print("Adding constructor features...")
        df = FeatureEngineer.add_constructor_features(df)

        print("Adding circuit features...")
        df = FeatureEngineer.add_circuit_features(df)

        # Only keep rows where driver has completed at least 1 race
        df = df[df['driver_races_completed'] > 0].copy()

        print(f"Creating target variables...")

        # TARGET 1: Classification target (Will driver score points?)
        df['points_finish'] = (df['final_position'] <= 10).astype(int)

        # TARGET 2: Regression target (What position will driver finish?)
        df['position_target'] = df['final_position'].clip(upper=20)

        return df
    
def get_f1_data(start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
    print("="*60)
    print("F1 DATA PIPELINE")
    print("="*60)

    print(f"\nFetching F1 data from Ergast API ({start_year}-{end_year})...")

    # Stage 1: Fetch raw data
    fetcher = F1DataFetcher(start_year=start_year, end_year=end_year)
    race_results = fetcher.build_race_dataset()

    print(f"\n Fetched {len(race_results)} race results")
    print(f"  Unique drivers: {race_results['driverId'].nunique()}")
    print(f"  Unique constructors: {race_results['constructorId'].nunique()}")

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

    # Inspect the result
    print("\nDataset preview:")
    print(df.head())

    print("\nFeature summary:")
    print(df[[
        'grid_position',
        'driver_points_5race_avg',
        'driver_points_10race_avg',
        'driver_finish_rate_5race',
        'driver_circuit_avg_finish',
        'points_finish',
    ]].describe())