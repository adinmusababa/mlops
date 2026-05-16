from raw.consume_data_supa import get_data
import pandas as pd

df = pd.DataFrame(get_data())

print(df.head())
