
#original_dataset_analysis.py

#Minimalny podgląd 'head' dla GEO Series Matrix (GSE183635).
#- Czyta .txt i .txt.gz
#- Dla czytelności pokazuje podzbiór najważniejszych kolumn (GSM, title, source, age, sex, institute, disease itp.).
#- Jeśli podasz wiele --series, skrypt je sklei (po wierszach) i pokaże head() z całości.

#Uwaga: nic nie zapisuje na dysk.
#input: 

#cd original_datasets_comparison

# python original_dataset_analysis.py --series C:\Users\weron\Downloads\Machine_Learning_Cancer_Detection-main\Machine_Learning_Cancer_Detection\src\data_sources\GSE183635-GPL16791_series_matrix.txt --series C:\Users\weron\Downloads\Machine_Learning_Cancer_Detection-main\Machine_Learning_Cancer_Detection\src\data_sources\GSE183635-GPL20301_series_matrix.txt  --out "geo_merged_head.txt" --head-rows 15


import argparse, gzip, io, re, csv, sys, textwrap
from typing import List, Optional, Dict
import pandas as pd

def _read_text(path: str) -> str:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _load_sample_table(text: str) -> Optional[pd.DataFrame]:
    """Spróbuj wyciągnąć blok sample table (TSV) między !sample_table_begin a !sample_table_end."""
    lines = text.splitlines()
    begin_idx = next((i for i, ln in enumerate(lines) if ln.lower().startswith("!sample_table_begin")), None)
    end_idx   = next((i for i, ln in enumerate(lines) if ln.lower().startswith("!sample_table_end")), None)
    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        return None
    # linia po 'begin' to nagłówek tabeli
    table_lines = lines[begin_idx+1:end_idx]
    tsv = "\n".join(table_lines)
    # niektóre pliki mają BOM/niechciane znaki – pandas sobie poradzi
    df = pd.read_csv(io.StringIO(tsv), sep="\t", dtype=str, low_memory=False)
    # standaryzuj nazwy kluczowych kolumn (częste nazwy w GEO)
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "geo_accession": rename_map[c] = "GSM"
        if lc.startswith("title"): rename_map[c] = "title"
        if lc.startswith("source_name_ch1"): rename_map[c] = "source_name_ch1"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def _split_values_rhs(rhs: str) -> List[str]:
    q = re.findall(r'"([^"]*)"', rhs)
    if q: return q
    return rhs.strip().split()

def _get_values_from_line(ln: str) -> List[str]:
    if "\t" in ln:
        parts = ln.split("\t")
        vals  = [p.strip() for p in parts[1:]]
        vals  = [v[1:-1] if len(v) >= 2 and v[0] == v[-1] == '"' else v for v in vals]
        return vals
    if "=" in ln:
        rhs = ln.split("=", 1)[1].strip()
        reader = csv.reader([rhs], skipinitialspace=True)
        vals = next(reader)
        return [v.strip().strip('"') for v in vals]
    return []

def _parse_series_matrix_fallback(text: str) -> pd.DataFrame:
    """Fallback: wektory !Sample_* i !Sample_characteristics_ch1 (rzadziej spotykane w tym GSE)."""
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.lstrip().startswith("!")]
    def vec(keys: List[str]) -> Optional[List[str]]:
        for ln in lines:
            if any(ln.startswith(k) for k in keys):
                vals = _get_values_from_line(ln)
                # pojedynczy łańcuch z wieloma GSM
                if len(vals) == 1 and re.match(r"^GSM\d+(?:\s+GSM\d+)+$", vals[0]):
                    return vals[0].split()
                return vals
        return None

    gsm   = vec(["!Sample_geo_accession", "!Series_sample_id"]) or []
    title = vec(["!Sample_title"]) or [None]*len(gsm)
    src   = vec(["!Sample_source_name_ch1"]) or [None]*len(gsm)

    char_rows: List[List[str]] = []
    for ln in lines:
        if ln.startswith("!Sample_characteristics_ch1"):
            vals = _get_values_from_line(ln)
            char_rows.append(vals)

    n = max(len(gsm), len(title), len(src), max((len(r) for r in char_rows), default=0))
    if len(gsm) == 0 and n > 0:
        gsm = [f"S{i+1}" for i in range(n)]
    def pad(v: List[Optional[str]], m: int) -> List[Optional[str]]:
        return (v + [None]*(m - len(v)))[:m]
    gsm, title, src = pad(gsm, n), pad(title, n), pad(src, n)

    key_re = re.compile(r"^\s*([^:]+)\s*:\s*(.*)$")
    data: Dict[str, List[Optional[str]]] = {"GSM": gsm, "title": title, "source_name_ch1": src}
    buckets: Dict[str, List[Optional[str]]] = {}
    def ensure(col: str):
        if col not in buckets:
            buckets[col] = [None]*n
    for row in char_rows:
        row = pad(row, n)
        for i, item in enumerate(row):
            if item is None: continue
            m = key_re.match(item)
            if m:
                key = m.group(1).strip().lower()
                val = m.group(2).strip()
                ensure(key); buckets[key][i] = val
            else:
                ensure("misc"); buckets["misc"][i] = item
    data.update(buckets)
    df = pd.DataFrame(data)
    if "GSM" in df.columns:
        df = df.drop_duplicates(subset=["GSM"])
    return df

def _explode_characteristics(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Z kolumn characteristics_ch1* wyciągnij 'key: value' do osobnych kolumn."""
    out = sample_df.copy()
    char_cols = [c for c in out.columns if c.lower().startswith("characteristics_ch1")]
    key_val: Dict[str, List[str]] = {}
    for c in char_cols:
        for idx, cell in out[c].items():
            if pd.isna(cell): continue
            # w niektórych plikach w jednej kolumnie jest już jedna para "key: value"
            parts = [cell]
            for p in parts:
                m = re.match(r"\s*([^:]+)\s*:\s*(.*)$", str(p))
                if not m: continue
                key = m.group(1).strip().lower()
                val = m.group(2).strip()
                col = key
                if col not in key_val:
                    key_val[col] = [None]*len(out)
                key_val[col][idx] = val
    if key_val:
        for k, v in key_val.items():
            # nie nadpisuj istniejących
            if k in out.columns: k = f"{k}_from_char"
            out[k] = v
    return out

def load_series_files(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        txt = _read_text(p)
        df = _load_sample_table(txt)
        mode = "sample_table"
        if df is None:
            df = _parse_series_matrix_fallback(txt)
            mode = "fallback"
        print(f"[INFO] {p} -> rows={len(df)}, cols={len(df.columns)} ({mode})")
        dfs.append(df)
    if len(dfs) == 1:
        return _explode_characteristics(dfs[0])
    # wyrównaj kolumny, sklej, deduplikuj po GSM/geo_accession
    all_cols = sorted(set().union(*[set(d.columns) for d in dfs]))
    dfs = [d.reindex(columns=all_cols) for d in dfs]
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    main_id = "GSM" if "GSM" in merged.columns else ("geo_accession" if "geo_accession" in merged.columns else None)
    if main_id:
        merged = merged.drop_duplicates(subset=[main_id])
    return _explode_characteristics(merged)

def _wrap(s: str, width: int = 120) -> str:
    if s is None or s != s:  # NaN
        return ""
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False, replace_whitespace=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True,
                    help="Ścieżka do Series Matrix (.txt lub .txt.gz). Możesz podać wiele razy.")
    ap.add_argument("--out", default="geo_preview.txt", help="Plik TXT z raportem.")
    ap.add_argument("--head-rows", type=int, default=12, help="Ile wierszy head() pokazać/zapisać.")
    args = ap.parse_args()

    geo = load_series_files(args.series)

    # wybierz czytelne kolumny do podglądu
    candidates = ["GSM", "geo_accession", "title", "source_name_ch1",
                  "classification group", "patient group", "cell type",
                  "disease state", "cancer type", "institute", "age", "sex"]
    cols = [c for c in candidates if c in geo.columns]
    if not cols:  # fallback: pierwsze kilka kolumn
        cols = list(geo.columns[:8])

    # przygotuj raport tekstowy (z zawijaniem)
    lines = []
    lines.append("================= GEO SERIES MATRIX PREVIEW =================")
    lines.append(f"Files: {len(args.series)}")
    for p in args.series: lines.append(f" - {p}")
    lines.append(f"Rows: {len(geo):,} | Cols: {len(geo.columns):,}")
    lines.append("")
    lines.append("[HEAD] Selected columns:")
    head = geo[cols].head(args.head_rows).copy()
    for c in cols:
        head[c] = head[c].map(lambda x: _wrap(x, 100))
    lines.append(head.to_string(index=False))
    lines.append("")
    # proste zliczenia, jeśli są
    for key in ["classification group", "patient group", "disease state"]:
        if key in geo.columns:
            lines.append(f"[COUNTS] {key}:")
            lines.append(str(geo[key].value_counts(dropna=False)))
            lines.append("")
    # pokaż listę kolumn na końcu (krótko)
    lines.append("[COLS] First 40 columns:")
    for c in list(geo.columns)[:40]:
        lines.append(f" - {c}")
    if len(geo.columns) > 40: lines.append(" ...")

    report = "\n".join(lines)
    print(report)

    # zapis do pliku
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[SAVED] Report saved to: {args.out}")
    except Exception as e:
        print(f"\n[WARN] Could not save report: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()