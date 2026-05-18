import os
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, shapiro, zscore
from sklearn.feature_selection import chi2
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data", "marketing_campaign.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_ii")

RANDOM_STATE = 42

COLOR_PALETTE = [
    "#675285",
    "#D16BA5",
    "#6C63FF",
    "#00B4D8",
    "#A23E48",
    "#F4A6C1",
    "#B8C0FF",
]

PRIMARY_COLOR = "#675285"
SECONDARY_COLOR = "#D16BA5"
TEXT_COLOR = "#4A3B5F"
GRID_COLOR = "#E6DFF0"
EDGE_COLOR = "#D8CBE6"
AXIS_BACKGROUND = "#FCF8FD"
FIGURE_BACKGROUND = "#FFFDFE"


HYPOTHESES = {
    "hipoteza_1": {
        "target": "NumWebPurchases",
        "predictors": ["Income", "Recency", "NumWebVisitsMonth"],
    },
    "hipoteza_2": {
        "target": "MntWines",
        "predictors": ["Income", "Kidhome", "Marital_Status"],
    },
    "hipoteza_3": {
        "target": "Response",
        "predictors": ["Income", "Recency", "NumCatalogPurchases"],
    },
}


def create_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for hypothesis_name in HYPOTHESES:
        os.makedirs(os.path.join(OUTPUT_DIR, hypothesis_name, "tables"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, hypothesis_name, "plots"), exist_ok=True)


def set_plot_style():
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette(COLOR_PALETTE)

    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.facecolor"] = FIGURE_BACKGROUND
    plt.rcParams["axes.facecolor"] = AXIS_BACKGROUND
    plt.rcParams["axes.edgecolor"] = EDGE_COLOR
    plt.rcParams["axes.labelcolor"] = TEXT_COLOR
    plt.rcParams["axes.titlecolor"] = TEXT_COLOR
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["xtick.color"] = TEXT_COLOR
    plt.rcParams["ytick.color"] = TEXT_COLOR
    plt.rcParams["grid.color"] = GRID_COLOR
    plt.rcParams["text.color"] = TEXT_COLOR


def get_paths(hypothesis_name):
    hypothesis_dir = os.path.join(OUTPUT_DIR, hypothesis_name)
    tables_dir = os.path.join(hypothesis_dir, "tables")
    plots_dir = os.path.join(hypothesis_dir, "plots")
    return hypothesis_dir, tables_dir, plots_dir


def save_table(df, path):
    df.to_csv(path, index=False)
    print(f"Zapisano: {path}")


def load_data():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Nie znaleziono pliku: {FILE_PATH}")

    return pd.read_csv(FILE_PATH, sep="\t")


def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

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


def get_hypothesis_columns(config):
    return [config["target"]] + config["predictors"]


def get_numeric_columns(df, columns):
    return [column for column in columns if pd.api.types.is_numeric_dtype(df[column])]


def get_categorical_columns(df, columns):
    return [column for column in columns if not pd.api.types.is_numeric_dtype(df[column])]


def descriptive_statistics(df, columns):
    rows = []

    for column in columns:
        values = df[column].dropna()

        if values.empty:
            continue

        mode_values = values.mode()
        mode_value = mode_values.iloc[0] if len(mode_values) > 0 else np.nan

        rows.append({
            "zmienna": column,
            "liczba_przypadkow": int(values.count()),
            "srednia": values.mean(),
            "mediana": values.median(),
            "moda": mode_value,
            "minimum": values.min(),
            "maksimum": values.max(),
            "odchylenie_std": values.std(),
            "wariancja": values.var(),
        })

    return pd.DataFrame(rows).round(3)


def frequency_table(df, column):
    counts = df[column].value_counts(dropna=False).reset_index()
    counts.columns = [column, "liczebnosc"]
    counts["procent"] = counts["liczebnosc"] / counts["liczebnosc"].sum() * 100
    return counts.round(3)


def correlation_interpretation(value):
    abs_value = abs(value)

    if abs_value < 0.1:
        strength = "brak lub bardzo słaba zależność"
    elif abs_value < 0.3:
        strength = "słaba zależność"
    elif abs_value < 0.5:
        strength = "umiarkowana zależność"
    elif abs_value < 0.7:
        strength = "dość silna zależność"
    else:
        strength = "silna zależność"

    if value > 0:
        direction = "dodatnia"
    elif value < 0:
        direction = "ujemna"
    else:
        direction = "brak kierunku"

    if direction == "brak kierunku":
        return strength

    return f"{strength} {direction}"


def correlation_table(df, config):
    target = config["target"]
    predictors = config["predictors"]

    columns = [target] + predictors
    numeric_columns = get_numeric_columns(df, columns)

    rows = []

    for left, right in itertools.combinations(numeric_columns, 2):
        corr = df[[left, right]].dropna().corr(method="pearson").iloc[0, 1]

        if target in [left, right]:
            relation_type = "zmienna zależna vs predyktor"
        else:
            relation_type = "zależność między predyktorami"

        rows.append({
            "badana_zaleznosc": f"{left} - {right}",
            "typ_zaleznosci": relation_type,
            "korelacja_Pearsona": corr,
            "interpretacja": correlation_interpretation(corr),
        })

    return pd.DataFrame(rows).round(3)


def normality_and_outliers(df, columns):
    rows = []

    for column in columns:
        values = df[column].dropna()

        if values.empty:
            continue

        sample = values.sample(min(len(values), 5000), random_state=RANDOM_STATE)

        shapiro_stat, p_value = shapiro(sample)

        if values.std() == 0:
            outliers_count = 0
        else:
            outliers_count = int((np.abs(zscore(values)) > 3).sum())

        rows.append({
            "zmienna": column,
            "shapiro_stat": shapiro_stat,
            "p_value": p_value,
            "czy_normalny_przy_0_05": "tak" if p_value > 0.05 else "nie",
            "liczba_odstajacych_zscore_gt_3": outliers_count,
        })

    return pd.DataFrame(rows).round(6)


def response_report_for_hypothesis_3(df):
    table = df.groupby("Response").agg(
        liczebnosc=("Response", "size"),
        sredni_Income=("Income", "mean"),
        mediana_Income=("Income", "median"),
        sredni_Recency=("Recency", "mean"),
        mediana_Recency=("Recency", "median"),
        sredni_NumCatalogPurchases=("NumCatalogPurchases", "mean"),
        mediana_NumCatalogPurchases=("NumCatalogPurchases", "median"),
    ).reset_index()

    table["procent"] = table["liczebnosc"] / table["liczebnosc"].sum() * 100

    table = table[
        [
            "Response",
            "liczebnosc",
            "procent",
            "sredni_Income",
            "mediana_Income",
            "sredni_Recency",
            "mediana_Recency",
            "sredni_NumCatalogPurchases",
            "mediana_NumCatalogPurchases",
        ]
    ]

    return table.round(3)


def chi_square_importance(df, config, hypothesis_name):
    _, tables_dir, plots_dir = get_paths(hypothesis_name)

    target = config["target"]
    predictors = config["predictors"]

    data = df[[target] + predictors].dropna().copy()

    if data[target].nunique() > 10 and pd.api.types.is_numeric_dtype(data[target]):
        target_discretizer = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
        y = target_discretizer.fit_transform(data[[target]]).ravel()
    else:
        y = data[target]

    if not pd.api.types.is_numeric_dtype(pd.Series(y)):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    processed_predictors = []

    for predictor in predictors:
        if pd.api.types.is_numeric_dtype(data[predictor]):
            discretizer = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
            transformed = discretizer.fit_transform(data[[predictor]]).ravel()
            processed_predictors.append(pd.Series(transformed, name=predictor, index=data.index))
        else:
            encoded = LabelEncoder().fit_transform(data[predictor].astype(str))
            processed_predictors.append(pd.Series(encoded, name=predictor, index=data.index))

    x = pd.concat(processed_predictors, axis=1)
    x = MinMaxScaler().fit_transform(x)

    chi_values, p_values = chi2(x, y)

    result = pd.DataFrame({
        "zmienna": predictors,
        "chi2": chi_values,
        "p_value": p_values,
    }).sort_values("chi2", ascending=False)

    save_table(result.round(6), os.path.join(tables_dir, "chi2_waznosc_predyktorow.csv"))

    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=result, y="zmienna", x="chi2", palette=COLOR_PALETTE)
    plt.title(f"Diagram ważności na podstawie statystyki Chi^2 - {hypothesis_name}", pad=14)
    plt.xlabel("Wartość statystyki Chi^2")
    plt.ylabel("Zmienna")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "chi2_waznosc_predyktorow.png"), dpi=220, bbox_inches="tight")
    plt.close()

    return result


def plot_histograms(df, numeric_columns, hypothesis_name):
    _, _, plots_dir = get_paths(hypothesis_name)

    for column in numeric_columns:
        plt.figure(figsize=(9, 5.5))
        sns.histplot(data=df, x=column, bins=30, kde=True, color=PRIMARY_COLOR)
        plt.title(f"Histogram zmiennej {column} - {hypothesis_name}", pad=14)
        plt.xlabel(column)
        plt.ylabel("Liczebność")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"histogram_{column}.png"), dpi=220, bbox_inches="tight")
        plt.close()


def plot_boxplots(df, numeric_columns, hypothesis_name):
    _, _, plots_dir = get_paths(hypothesis_name)

    for column in numeric_columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, y=column, color=PRIMARY_COLOR)
        plt.title(f"Wykres ramka-wąsy dla {column} - {hypothesis_name}", pad=14)
        plt.ylabel(column)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"boxplot_{column}.png"), dpi=220, bbox_inches="tight")
        plt.close()


def plot_categorized_boxplots(df, config, hypothesis_name):
    _, _, plots_dir = get_paths(hypothesis_name)

    columns = get_hypothesis_columns(config)
    numeric_columns = get_numeric_columns(df, columns)
    categorical_columns = get_categorical_columns(df, columns)

    for categorical_column in categorical_columns:
        for numeric_column in numeric_columns:
            plt.figure(figsize=(9, 5.5))
            sns.boxplot(
                data=df,
                x=categorical_column,
                y=numeric_column,
                palette=COLOR_PALETTE,
            )
            plt.title(f"Skategoryzowany boxplot: {numeric_column} względem {categorical_column}", pad=14)
            plt.xlabel(categorical_column)
            plt.ylabel(numeric_column)
            plt.tight_layout()
            plt.savefig(
                os.path.join(plots_dir, f"boxplot_{numeric_column}_wg_{categorical_column}.png"),
                dpi=220,
                bbox_inches="tight",
            )
            plt.close()


def plot_group_means(df, config, hypothesis_name):
    _, tables_dir, plots_dir = get_paths(hypothesis_name)

    target = config["target"]
    predictors = config["predictors"]

    for predictor in predictors:
        if not pd.api.types.is_numeric_dtype(df[predictor]):
            group_table = df.groupby(predictor)[target].mean().reset_index()
            group_table.columns = [predictor, f"srednia_{target}"]

            save_table(
                group_table.round(3),
                os.path.join(tables_dir, f"srednie_{target}_wg_{predictor}.csv")
            )

            plt.figure(figsize=(9, 5.5))
            sns.barplot(data=group_table, x=predictor, y=f"srednia_{target}", palette=COLOR_PALETTE)
            plt.title(f"Średnia {target} względem {predictor}", pad=14)
            plt.xlabel(predictor)
            plt.ylabel(f"Średnia {target}")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"srednia_{target}_wg_{predictor}.png"), dpi=220, bbox_inches="tight")
            plt.close()
        else:
            groups = pd.qcut(df[predictor], q=4, duplicates="drop")
            group_table = df.groupby(groups)[target].mean().reset_index()
            group_table.columns = [f"grupa_{predictor}", f"srednia_{target}"]
            group_table[f"grupa_{predictor}"] = group_table[f"grupa_{predictor}"].astype(str)

            save_table(
                group_table.round(3),
                os.path.join(tables_dir, f"srednie_{target}_wg_grup_{predictor}.csv")
            )

            plt.figure(figsize=(10, 5.5))
            sns.barplot(data=group_table, x=f"grupa_{predictor}", y=f"srednia_{target}", color=PRIMARY_COLOR)
            plt.title(f"Średnia {target} w grupach zmiennej {predictor}", pad=14)
            plt.xlabel(f"Grupa {predictor}")
            plt.ylabel(f"Średnia {target}")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"srednia_{target}_wg_grup_{predictor}.png"), dpi=220, bbox_inches="tight")
            plt.close()


def plot_correlation_matrix(df, numeric_columns, hypothesis_name):
    _, tables_dir, plots_dir = get_paths(hypothesis_name)

    corr = df[numeric_columns].corr(method="pearson")

    corr_table = corr.reset_index().rename(columns={"index": "zmienna"})
    save_table(corr_table.round(3), os.path.join(tables_dir, "macierz_korelacji.csv"))

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=sns.light_palette(PRIMARY_COLOR, as_cmap=True),
        linewidths=1,
        linecolor="white",
    )
    plt.title(f"Macierz korelacji - {hypothesis_name}", pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "macierz_korelacji.png"), dpi=220, bbox_inches="tight")
    plt.close()


def plot_scatterplots(df, config, hypothesis_name):
    _, _, plots_dir = get_paths(hypothesis_name)

    target = config["target"]
    predictors = config["predictors"]
    numeric_predictors = [column for column in predictors if pd.api.types.is_numeric_dtype(df[column])]

    for predictor in numeric_predictors:
        plt.figure(figsize=(8, 5.5))
        sns.scatterplot(data=df, x=predictor, y=target, color=PRIMARY_COLOR, alpha=0.6)
        plt.title(f"Wykres rozrzutu: {target} względem {predictor}", pad=14)
        plt.xlabel(predictor)
        plt.ylabel(target)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"scatter_{target}_vs_{predictor}.png"), dpi=220, bbox_inches="tight")
        plt.close()

        if target == "Response":
            grouped_predictor = pd.qcut(df[predictor], q=4, duplicates="drop")
            plot_data = df.copy()
            plot_data[f"grupa_{predictor}"] = grouped_predictor.astype(str)

            plt.figure(figsize=(10, 5.5))
            sns.histplot(
                data=plot_data,
                x=predictor,
                hue="Response",
                bins=30,
                multiple="stack",
                palette=COLOR_PALETTE[:2],
            )
            plt.title(f"Skategoryzowany histogram: {predictor} względem Response", pad=14)
            plt.xlabel(predictor)
            plt.ylabel("Liczebność")
            plt.tight_layout()
            plt.savefig(
                os.path.join(plots_dir, f"histogram_{predictor}_wg_Response.png"),
                dpi=220,
                bbox_inches="tight",
            )
            plt.close()


def run_hypothesis_analysis(df, hypothesis_name, config):
    print(f"\n{hypothesis_name}")

    _, tables_dir, _ = get_paths(hypothesis_name)

    columns = get_hypothesis_columns(config)
    data = df[columns].dropna().copy()

    numeric_columns = get_numeric_columns(data, columns)
    categorical_columns = get_categorical_columns(data, columns)

    stats = descriptive_statistics(data, numeric_columns)
    save_table(stats, os.path.join(tables_dir, "statystyki_opisowe.csv"))

    for categorical_column in categorical_columns:
        freq = frequency_table(data, categorical_column)
        save_table(freq, os.path.join(tables_dir, f"licznosci_{categorical_column}.csv"))

    if hypothesis_name == "hipoteza_3":
        response_table = response_report_for_hypothesis_3(data)
        save_table(response_table, os.path.join(tables_dir, "charakterystyka_klientow_wedlug_Response.csv"))

    corr_table = correlation_table(data, config)
    save_table(corr_table, os.path.join(tables_dir, "korelacje_istotne.csv"))

    normality_table = normality_and_outliers(data, numeric_columns)
    save_table(normality_table, os.path.join(tables_dir, "test_normalnosci_i_odstajace.csv"))

    chi_square_importance(data, config, hypothesis_name)

    plot_histograms(data, numeric_columns, hypothesis_name)
    plot_boxplots(data, numeric_columns, hypothesis_name)
    plot_categorized_boxplots(data, config, hypothesis_name)
    plot_group_means(data, config, hypothesis_name)
    plot_correlation_matrix(data, numeric_columns, hypothesis_name)
    plot_scatterplots(data, config, hypothesis_name)


def main():
    create_directories()
    set_plot_style()

    df = load_data()
    df = clean_data(df)

    for hypothesis_name, config in HYPOTHESES.items():
        run_hypothesis_analysis(df, hypothesis_name, config)

    print(f"\nWyniki zapisano w: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
