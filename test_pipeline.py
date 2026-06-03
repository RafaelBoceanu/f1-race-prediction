from f1_data import get_f1_data
from f1_models import train_f1_models

df = get_f1_data(start_year=2015, end_year=2026)
manager = train_f1_models(df)
manager.save_models('./models')
print(manager.results)