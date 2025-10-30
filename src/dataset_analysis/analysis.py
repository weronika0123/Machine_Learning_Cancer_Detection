import pandas as pd
import argparse
import sys

# Original dataset links:
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183635
# https://ftp.ncbi.nlm.nih.gov/geo/series/GSE183nnn/GSE183635/matrix/


def main():
    parser = argparse.ArgumentParser(description="Dataset diagnostic tool for RNA expression data")
    parser.add_argument("--data", required=True, help="Path to CSV file with RNA expression data")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data, sep=",", low_memory=False)
    except FileNotFoundError:
        print(f"[ERROR] Cannot find file: {args.data}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("=== Basic Data Diagnostics ===")
    print("=" * 80)

    # 1. Check missing values
    print("\n[INFO] Missing values (NaN) per column (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    # 2. Check data types
    print("\n[INFO] Data types distribution:")
    print(df.dtypes.value_counts())

    # 3. Statistical summary (mean, min, max, etc.)
    print("\n[INFO] Basic statistics (first 10 columns):")
    print(df.describe(include='all').transpose().head(10))

    # Gene columns
    gene_cols = [c for c in df.columns if c.startswith("ENSG")]
    X = df[gene_cols]

    print("\n" + "=" * 80)
    print("=== NA Analysis by Column (genes only) ===")
    print("=" * 80)
    na_counts = X.isna().sum()
    na_cols = na_counts[na_counts > 0].sort_values(ascending=False)
    if na_cols.empty:
        print("[INFO] No columns with NA values among genes")
    else:
        na_df = pd.DataFrame({
            "na_count": na_cols,
            "na_percent": (na_cols / len(X) * 100).round(2)
        })
        print(na_df.head(30))  # Preview (first 30)
        na_df.to_csv("diag_na_columns.csv")
        print("[INFO] Full list saved to: diag_na_columns.csv")

    # Global maximum
    col_max_series = X.max(skipna=True)         # Max per column
    global_max_val = col_max_series.max()       # Global max
    global_max_col = col_max_series.idxmax()    # Column with global max
    global_max_row = X[global_max_col].idxmax() # Row index with global max
    print("\n" + "=" * 80)
    print("=== Global MAX (genes only) ===")
    print("=" * 80)
    print(f"Value: {global_max_val}")
    print(f"Column (gene): {global_max_col}")
    print(f"Row (index in df): {global_max_row}")

    # Global minimum
    col_min_series = X.min(skipna=True)         # Min per column
    global_min_val = col_min_series.min()       # Global min
    global_min_col = col_min_series.idxmin()    # Column with global min
    global_min_row = X[global_min_col].idxmin() # Row index with global min
    print("\n" + "=" * 80)
    print("=== Global MIN (genes only) ===")
    print("=" * 80)
    print(f"Value: {global_min_val}")
    print(f"Column (gene): {global_min_col}")
    print(f"Row (index in df): {global_min_row}")
    print("=" * 80)


if __name__ == "__main__":
    main()
