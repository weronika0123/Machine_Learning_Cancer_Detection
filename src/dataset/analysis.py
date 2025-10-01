import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Original dataset links:
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183635   ->   https://ftp.ncbi.nlm.nih.gov/geo/series/GSE183nnn/GSE183635/matrix/


df = pd.read_csv(r"C:\Users\weron\Downloads\Machine_Learning_Cancer_Detection-main\Machine_Learning_Cancer_Detection\src\data_sources\liquid_biopsy_data.csv", sep=",", low_memory=False)

numeric_df = df.iloc[:,1:-16]

# Globalne statystyki
global_stats = {
    "Non-NA count": numeric_df.notna().sum().sum(),
    "NA count": numeric_df.isna().sum().sum(),
    "NA %": round((numeric_df.isna().sum().sum() / numeric_df.size) * 100, 2),
    "Min": numeric_df.min().min(),
    "Max": numeric_df.max().max(),
    "Mean": numeric_df.stack().mean(),
    "Median": numeric_df.stack().median(),
    "Std": numeric_df.stack().std()
}

# Zamiana do DataFrame dla czytelności
global_summary = pd.DataFrame(global_stats, index=["Dataset overview"]).T
pd.options.display.float_format = '{:,.4f}'.format  # 4 miejsca po przecinku
print(global_summary)
global_summary.to_csv("dataset_global_summary.csv", float_format="%.4f")

top_features = numeric_df.mean()

plt.figure(figsize=(8,5))
plt.hist(top_features, bins=50, color="steelblue")

# Add vertical lines for global mean and median
plt.axvline(global_stats["Mean"], color="red", linestyle="--", linewidth=2, label=f"Mean = {global_stats['Mean']:.2f}")
plt.axvline(global_stats["Median"], color="green", linestyle="--", linewidth=2, label=f"Median = {global_stats['Median']:.2f}")

plt.title("Distribution of mean expression across genes")
plt.xlabel("Mean expression value")
plt.ylabel("Number of genes")
plt.legend()
plt.savefig("mean_dist_with_lines.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()



# Class distribution
class_counts = df["cancer"].value_counts().sort_index()
class_percent = df["cancer"].value_counts(normalize=True).sort_index() * 100

# Create summary table
class_summary = pd.DataFrame({
    "Count": class_counts,
    "Percentage (%)": class_percent.round(2)
})
print("\nClass distribution (Cancer vs Healthy):\n")
print(class_summary)

# Plot distribution
plt.figure(figsize=(6,5))
class_counts.plot(kind="bar", color=["steelblue", "indianred"])
plt.title("Class Distribution: Cancer vs Non-cancer")
plt.xlabel("Class (0 = non-cancer sample, 1 = cancer sample)")
plt.ylabel("Number of samples")

# Add labels on bars
for i, v in enumerate(class_counts):
    plt.text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("class_distribution.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()


