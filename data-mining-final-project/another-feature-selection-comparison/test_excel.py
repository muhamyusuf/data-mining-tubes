import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent.parent / 'dataset-type-2'
file = list(data_dir.glob('*.xlsx'))[0]

print(f"Reading: {file.name}")
df = pd.read_excel(file, engine='openpyxl', header=None)

print(f"\nShape: {df.shape}")
print(f"\nFirst 10 rows:")
print(df.head(10))
print(f"\nData types:")
print(df.dtypes.value_counts())

# Try with header
df_header = pd.read_excel(file, engine='openpyxl')
print(f"\n\nWith header:")
print(f"Columns: {df_header.columns.tolist()}")
print(f"Dtypes: {df_header.dtypes.to_dict()}")
print(f"\nNumeric columns: {df_header.select_dtypes(include=['number']).columns.tolist()}")
