import pandas as pd

#Original dataset links:
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183635   ->   https://ftp.ncbi.nlm.nih.gov/geo/series/GSE183nnn/GSE183635/matrix/


df = pd.read_csv("C:/Users/weron/Downloads/Machine_Learning_Cancer_Detection-main/Machine_Learning_Cancer_Detection/src/data_sources/liquid_biopsy_data.csv", sep=",", low_memory=False)





print("=== Podstawowa diagnostyka danych ===")

# 1. Sprawdź brakujące wartości
print("\n======================================Brakujące wartości (NaN) w każdej kolumnie:")
print(df.isna().sum().sort_values(ascending=False).head(10))

# 2. Sprawdź typy danych
print("\n=========================================Typy danych:")
print(df.dtypes.value_counts())

# 3. Podsumowanie statystyczne (średnia, min, max itd.)
print("\n=========================================Podstawowe statystyki:")
print(df.describe(include='all').transpose().head(10))

# Kolumny genowe
gene_cols = [c for c in df.columns if c.startswith("ENSG")]
X = df[gene_cols]

print("===================================== NA by column (only columns with any NA) ===")
na_counts = X.isna().sum()
na_cols = na_counts[na_counts > 0].sort_values(ascending=False)
if na_cols.empty:
    print("Brak kolumn z NA wśród genów.")
else:
    na_df = pd.DataFrame({
        "na_count": na_cols,
        "na_percent": (na_cols / len(X) * 100).round(2)
    })
    print(na_df.head(30))  # podgląd (pierwsze 30)
    na_df.to_csv("diag_na_columns.csv")
    print("Zapisano pełną listę do: diag_na_columns.csv")

# Globalne maksimum
col_max_series = X.max(skipna=True)         # max w każdej kolumnie
global_max_val = col_max_series.max()       # globalny max
global_max_col = col_max_series.idxmax()    # kolumna z globalnym maxem
global_max_row = X[global_max_col].idxmax() # indeks wiersza z globalnym maxem
print("\n=========================== Global MAX (genes only) ===")
print(f"Wartość: {global_max_val}")
print(f"Kolumna (gen): {global_max_col}")
print(f"Wiersz (index w df): {global_max_row}")

# Globalne minimum
col_min_series = X.min(skipna=True)         # min w każdej kolumnie
global_min_val = col_min_series.min()       # globalne min
global_min_col = col_min_series.idxmin()    # kolumna z globalnym min
global_min_row = X[global_min_col].idxmin() # indeks wiersza z globalnym min
print("\n============================ Global MIN (genes only) ===")
print(f"Wartość: {global_min_val}")
print(f"Kolumna (gen): {global_min_col}")
print(f"Wiersz (index w df): {global_min_row}")

