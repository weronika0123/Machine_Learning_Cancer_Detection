import pandas as pd

#Original dataset links:
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183635   ->   https://ftp.ncbi.nlm.nih.gov/geo/series/GSE183nnn/GSE183635/matrix/


df = pd.read_csv("data_sources/liquid_biopsy_data.csv", sep=",", low_memory=False)

print("Liczba wierszy:", len(df))
print("Liczba kolumn:", len(df.columns))
print("\nPierwsze 20 kolumn:", list(df.columns[:20]))
print("\nOstatnie 20 kolumn:", list(df.columns[-20:]))
print("\nTypy danych kolumn:\n", df.dtypes.tail(20))
print("\nLiczba unikalnych wartości w każdej kolumnie:\n", df.nunique())

for col in df.columns[-16:]:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False).head(15))