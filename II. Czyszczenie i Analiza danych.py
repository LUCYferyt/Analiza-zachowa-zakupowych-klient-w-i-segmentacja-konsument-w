import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, shapiro, zscore


FILE_PATH = "data/marketing_campaign.csv"
OUTPUT_DIR = "outputs_ii"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/tables", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

COLOR_PALETTE = [
    "#675285",  # fiolet
    "#D16BA5",  # magenta
    "#6C63FF",  # błękitowo-fioletowy
    "#00B4D8",  # błękit
    "#A23E48",  # granatowo-bordowy
    "#F4A6C1",  # jasny róż
    "#B8C0FF",  # pastelowy niebieski
]

PRIMARY_COLOR = "#675285"
SECONDARY_COLOR = "#D16BA5"
ACCENT_COLOR = "#A23E48"
LIGHT_COLOR = "#F9EFF5"
TEXT_COLOR = "#4A3B5F"
GRID_COLOR = "#E6DFF0"
EDGE_COLOR = "#D8CBE6"
AXIS_BACKGROUND = "#FCF8FD"
FIGURE_BACKGROUND = "#FFFDFE"

NUMERIC_VARS = [
    "NumWebPurchases",
    "Income",
    "Recency",
    "NumWebVisitsMonth",
    "MntWines",
    "Kidhome",
    "Response",
    "NumCatalogPurchases",
]

CATEGORICAL_VARS = [
    "Marital_Status",
]

BOXPLOT_CATEGORIZED_PAIRS = [
    ("MntWines", "Marital_Status"),
    ("Income", "Marital_Status"),
]

SCATTER_PAIRS = [
    ("Income", "NumWebPurchases"),
    ("Income", "MntWines"),
    ("NumCatalogPurchases", "Response"),
]

SCATTER_GROUP_VAR = "Marital_Status"

#styl wykresów

def set_plot_style():
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette(COLOR_PALETTE)

    plt.rcParams["figure.figsize"] = (9, 5.5)
    plt.rcParams["figure.facecolor"] = FIGURE_BACKGROUND
    plt.rcParams["axes.facecolor"] = AXIS_BACKGROUND
    plt.rcParams["axes.edgecolor"] = EDGE_COLOR
    plt.rcParams["axes.labelcolor"] = TEXT_COLOR
    plt.rcParams["axes.titlecolor"] = TEXT_COLOR
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.color"] = TEXT_COLOR
    plt.rcParams["ytick.color"] = TEXT_COLOR
    plt.rcParams["text.color"] = TEXT_COLOR
    plt.rcParams["grid.color"] = GRID_COLOR
    plt.rcParams["grid.alpha"] = 0.85
    plt.rcParams["axes.grid"] = True


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()

    required_cols = list(set(NUMERIC_VARS + CATEGORICAL_VARS))
    df = df.dropna(subset=required_cols)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    for col in NUMERIC_VARS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=NUMERIC_VARS)

    if "Marital_Status" in df.columns:
        df["Marital_Status"] = df["Marital_Status"].replace({
            "Together": "Partner",
            "Married": "Partner",
            "Single": "Single",
            "Alone": "Single",
            "Divorced": "Single",
            "Widow": "Single",
        })

        df = df[~df["Marital_Status"].isin(["YOLO", "Absurd"])]

    return df


#statysttyki opisowe

def descriptive_stats(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []

    for col in numeric_cols:
        series = df[col].dropna()
        mode_values = series.mode()

        rows.append({
            "zmienna": col,
            "liczba_przypadkow": series.count(),
            "srednia": series.mean(),
            "mediana": series.median(),
            "moda": mode_values.iloc[0] if not mode_values.empty else np.nan,
            "minimum": series.min(),
            "maksimum": series.max(),
            "odchylenie_std": series.std(),
            "wariancja": series.var(),
        })

    result = pd.DataFrame(rows)
    result.to_csv(f"{OUTPUT_DIR}/tables/statystyki_opisowe.csv", index=False)
    return result


#tab liczebnosci

def frequency_tables(df: pd.DataFrame, categorical_cols: list[str]) -> dict:
    results = {}

    for col in categorical_cols:
        freq = df[col].value_counts(dropna=False).reset_index()
        freq.columns = [col, "liczebnosc"]
        freq["procent"] = 100 * freq["liczebnosc"] / freq["liczebnosc"].sum()
        freq.to_csv(f"{OUTPUT_DIR}/tables/tabela_licznosci_{col}.csv", index=False)
        results[col] = freq

    return results

#tab krzyzowa

def multivariate_table(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(df["Marital_Status"], df["Response"], margins=True)
    table.to_csv(f"{OUTPUT_DIR}/tables/tabela_krzyzowa_marital_response.csv")
    return table


#histogram

def plot_histograms(df: pd.DataFrame, numeric_cols: list[str]):
    for i, col in enumerate(numeric_cols):
        plt.figure(figsize=(9, 5.5))
        sns.histplot(
            df[col],
            kde=True,
            bins=20,
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.9
        )
        plt.title(f"Histogram zmiennej {col}", pad=14)
        plt.xlabel(col)
        plt.ylabel("Liczebność")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/hist_{col}.png", dpi=220, bbox_inches="tight")
        plt.close()


def categorized_histograms(df: pd.DataFrame):
    df["Income_group"] = pd.qcut(df["Income"], q=4, duplicates="drop")
    df["Recency_group"] = pd.cut(df["Recency"], bins=4)

    plt.figure(figsize=(9, 5.5))
    sns.histplot(
        data=df,
        x="NumWebPurchases",
        hue="Income_group",
        multiple="stack",
        bins=15,
        palette=COLOR_PALETTE[:4],
        edgecolor="white",
        linewidth=1
    )
    plt.title("Histogram skategoryzowany: NumWebPurchases względem Income_group", pad=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/hist_cat_NumWebPurchases_Income_group.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    sns.histplot(
        data=df,
        x="MntWines",
        hue="Marital_Status",
        multiple="stack",
        bins=20,
        palette=COLOR_PALETTE,
        edgecolor="white",
        linewidth=1
    )
    plt.title("Histogram skategoryzowany: MntWines względem Marital_Status", pad=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/hist_cat_MntWines_Marital_Status.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    sns.histplot(
        data=df,
        x="Response",
        hue="Recency_group",
        multiple="stack",
        bins=2,
        palette=COLOR_PALETTE[:4],
        edgecolor="white",
        linewidth=1
    )
    plt.title("Histogram skategoryzowany: Response względem Recency_group", pad=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/hist_cat_Response_Recency_group.png", dpi=220, bbox_inches="tight")
    plt.close()


#średnie w grupach

def interaction_plots(df: pd.DataFrame):
    mean_table = df.groupby("Marital_Status", as_index=False)["MntWines"].mean()
    mean_table.to_csv(f"{OUTPUT_DIR}/tables/srednie_w_grupach_MntWines_Marital_Status.csv", index=False)

    plt.figure(figsize=(9, 5.5))
    sns.pointplot(
        data=df,
        x="Marital_Status",
        y="MntWines",
        color=PRIMARY_COLOR,
        errorbar=None
    )
    plt.title("Wykres średnich w grupach: MntWines względem Marital_Status", pad=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/interaction_MntWines_Marital_Status.png", dpi=220, bbox_inches="tight")
    plt.close()

    mean_table_2 = df.groupby("Income_group", as_index=False)["NumWebPurchases"].mean()
    mean_table_2.to_csv(f"{OUTPUT_DIR}/tables/srednie_w_grupach_NumWebPurchases_Income_group.csv", index=False)


    plt.figure(figsize=(9, 5.5))
    plt.bar(
        mean_table_2["Income_group"].astype(str),
        mean_table_2["NumWebPurchases"],
        color=PRIMARY_COLOR
    )
    plt.title("Średnia liczba zakupów internetowych w grupach dochodu", pad=14)
    plt.xlabel("Grupa dochodu")
    plt.ylabel("Średnia NumWebPurchases")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/plots/bar_mean_NumWebPurchases_Income_group.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close()


    plt.figure(figsize=(9, 5.5))
    sns.pointplot(
        data=df,
        x="Income_group",
        y="NumWebPurchases",
        color=SECONDARY_COLOR,
        errorbar=None
    )
    plt.title("Wykres średnich w grupach: NumWebPurchases względem Income_group", pad=14)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/interaction_NumWebPurchases_Income_group.png", dpi=220, bbox_inches="tight")
    plt.close()


#macierz korelacji

def correlation_matrix(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    corr = df[numeric_cols].corr()
    corr.to_csv(f"{OUTPUT_DIR}/tables/macierz_korelacji.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=sns.light_palette(PRIMARY_COLOR, as_cmap=True),
        linewidths=1,
        linecolor="white"
    )
    plt.title("Macierz korelacji", pad=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/macierz_korelacji.png", dpi=240, bbox_inches="tight")
    plt.close()

    return corr


#diagram waznosci

def chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    contingency = pd.crosstab(df["Marital_Status"], df["Response"])
    chi2_stat, p_value, dof, _ = chi2_contingency(contingency)

    results.append({
        "zmienna_1": "Marital_Status",
        "zmienna_2": "Response",
        "chi2": chi2_stat,
        "p_value": p_value,
        "df": dof
    })

    numeric_predictors = [
        "Income",
        "Recency",
        "NumCatalogPurchases",
        "NumWebVisitsMonth",
        "Kidhome",
        "MntWines"
    ]

    for col in numeric_predictors:
        binned = pd.qcut(df[col], q=4, duplicates="drop")
        contingency = pd.crosstab(binned, df["Response"])
        chi2_stat, p_value, dof, _ = chi2_contingency(contingency)

        results.append({
            "zmienna_1": col,
            "zmienna_2": "Response",
            "chi2": chi2_stat,
            "p_value": p_value,
            "df": dof
        })

    result_df = pd.DataFrame(results).sort_values("chi2", ascending=False)
    result_df.to_csv(f"{OUTPUT_DIR}/tables/testy_chi2.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=result_df,
        x="chi2",
        y="zmienna_1",
        palette=COLOR_PALETTE
    )
    plt.title("Diagram ważności na podstawie statystyki Chi^2", pad=14)
    plt.xlabel("Wartość statystyki Chi^2")
    plt.ylabel("Zmienna")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/diagram_waznosci_chi2.png", dpi=240, bbox_inches="tight")
    plt.close()

    return result_df


#boxplot

def boxplots(df: pd.DataFrame, numeric_cols: list[str]):
    for i, col in enumerate(numeric_cols):
        plt.figure(figsize=(9, 5.5))
        sns.boxplot(
            x=df[col],
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            linewidth=1.5,
            fliersize=5
        )
        plt.title(f"Wykres ramka-wąsy: {col}", pad=14)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/boxplot_{col}.png", dpi=220, bbox_inches="tight")
        plt.close()


def categorized_boxplots(df: pd.DataFrame, pairs: list[tuple[str, str]]):
    for numeric_col, cat_col in pairs:
        plt.figure(figsize=(9, 5.5))
        sns.boxplot(
            data=df,
            x=cat_col,
            y=numeric_col,
            palette=COLOR_PALETTE,
            linewidth=1.4
        )
        plt.title(f"Skategoryzowany boxplot: {numeric_col} względem {cat_col}", pad=14)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/boxplot_cat_{numeric_col}_{cat_col}.png", dpi=220, bbox_inches="tight")
        plt.close()


#normalne/odstajace

def normality_and_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    results = []
    outlier_summary = []

    for col in numeric_cols:
        series = df[col].dropna()

        if len(series) > 5000:
            sample = series.sample(5000, random_state=42)
        else:
            sample = series

        stat, p_value = shapiro(sample)

        z_scores = np.abs(zscore(series))
        outliers_count = (z_scores > 3).sum()

        results.append({
            "zmienna": col,
            "shapiro_stat": stat,
            "p_value": p_value,
            "czy_normalny_przy_0_05": "tak" if p_value > 0.05 else "nie"
        })

        outlier_summary.append({
            "zmienna": col,
            "liczba_odstajacych_zscore_gt_3": int(outliers_count)
        })

    normality_df = pd.DataFrame(results)
    outliers_df = pd.DataFrame(outlier_summary)

    normality_df.to_csv(f"{OUTPUT_DIR}/tables/test_normalnosci.csv", index=False)
    outliers_df.to_csv(f"{OUTPUT_DIR}/tables/wartosci_odstajace.csv", index=False)

    return normality_df.merge(outliers_df, on="zmienna")


#wykres rozrzutu

def scatter_plots(df: pd.DataFrame, pairs: list[tuple[str, str]], group_var: str):
    for i, (x_col, y_col) in enumerate(pairs):
        plt.figure(figsize=(9, 5.5))
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            s=55,
            alpha=0.75
        )
        plt.title(f"Wykres rozrzutu: {y_col} względem {x_col}", pad=14)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/scatter_{x_col}_{y_col}.png", dpi=240, bbox_inches="tight")
        plt.close()

        if group_var in df.columns:
            plt.figure(figsize=(9, 5.5))
            sns.scatterplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=group_var,
                palette=COLOR_PALETTE,
                s=55,
                alpha=0.75
            )
            plt.title(f"Wykres rozrzutu: {y_col} względem {x_col}, grupowanie: {group_var}", pad=14)
            plt.tight_layout()
            plt.savefig(
                f"{OUTPUT_DIR}/plots/scatter_cat_{x_col}_{y_col}_{group_var}.png",
                dpi=240,
                bbox_inches="tight"
            )
            plt.close()


def main():
    set_plot_style()

    df = load_data(FILE_PATH)
    df = clean_data(df)

    print("Dane po czyszczeniu:", df.shape)

    stats_df = descriptive_stats(df, NUMERIC_VARS)
    print("\nStatystyki opisowe:")
    print(stats_df)

    freq_tables = frequency_tables(df, CATEGORICAL_VARS)
    for col, table in freq_tables.items():
        print(f"\nTabela liczności dla {col}:")
        print(table)

    multi_table = multivariate_table(df)
    print("\nTabela wielodzielcza:")
    print(multi_table)

    plot_histograms(df, NUMERIC_VARS)
    categorized_histograms(df)
    interaction_plots(df)

    corr_df = correlation_matrix(df, NUMERIC_VARS)
    print("\nMacierz korelacji:")
    print(corr_df)

    chi2_df = chi_square_tests(df)
    print("\nTesty Chi^2:")
    print(chi2_df)

    boxplots(df, NUMERIC_VARS)
    categorized_boxplots(df, BOXPLOT_CATEGORIZED_PAIRS)

    normality_df = normality_and_outliers(df, NUMERIC_VARS)
    print("\nNormalność i odstające:")
    print(normality_df)

    scatter_plots(df, SCATTER_PAIRS, SCATTER_GROUP_VAR)

    print(f"\nWyniki zapisano w folderze: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()