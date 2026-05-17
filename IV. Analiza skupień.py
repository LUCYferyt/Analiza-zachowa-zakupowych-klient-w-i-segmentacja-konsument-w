import os
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data", "marketing_campaign.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_iv")

RANDOM_STATE = 42
N_CLUSTERS = 3
RESET_OUTPUT_DIR = True

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


def reset_output_directory():
    if RESET_OUTPUT_DIR and os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)


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


def save_dataframe(df, path):
    df.to_csv(path, index=False)


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


def split_columns_by_type(df, columns):
    numeric_columns = []
    categorical_columns = []

    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


def create_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_data_for_clustering(df, config):
    columns = get_hypothesis_columns(config)
    missing_columns = [column for column in columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Brak kolumn w danych: {missing_columns}")

    data = df[columns].copy()
    data = data.dropna()

    numeric_columns, categorical_columns = split_columns_by_type(data, columns)

    scaler = StandardScaler()
    numeric_array = scaler.fit_transform(data[numeric_columns]) if numeric_columns else np.empty((len(data), 0))

    if categorical_columns:
        encoder = create_encoder()
        categorical_array = encoder.fit_transform(data[categorical_columns])

        try:
            categorical_names = encoder.get_feature_names_out(categorical_columns).tolist()
        except AttributeError:
            categorical_names = encoder.get_feature_names(categorical_columns).tolist()
    else:
        categorical_array = np.empty((len(data), 0))
        categorical_names = []

    x_scaled = np.hstack([numeric_array, categorical_array])
    transformed_columns = numeric_columns + categorical_names
    transformed_data = pd.DataFrame(x_scaled, columns=transformed_columns, index=data.index)

    return data, transformed_data, x_scaled, numeric_columns, categorical_columns


def save_variable_table(hypothesis_name, config, numeric_columns, categorical_columns):
    _, tables_dir, _ = get_paths(hypothesis_name)

    rows = []
    for column in get_hypothesis_columns(config):
        role = "zmienna zależna" if column == config["target"] else "zmienna objaśniająca"

        if column in numeric_columns:
            variable_type = "ilościowa"
        elif column in categorical_columns:
            variable_type = "jakościowa"
        else:
            variable_type = "inna"

        rows.append({
            "zmienna": column,
            "rola": role,
            "typ": variable_type,
        })

    save_dataframe(
        pd.DataFrame(rows),
        os.path.join(tables_dir, "zmienne_do_analizy_skupien.csv")
    )


def evaluate_kmeans_cluster_count(x_scaled, hypothesis_name, k_range=range(2, 7)):
    _, tables_dir, plots_dir = get_paths(hypothesis_name)

    rows = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(x_scaled)

        rows.append({
            "k": k,
            "silhouette": silhouette_score(x_scaled, labels),
            "calinski_harabasz": calinski_harabasz_score(x_scaled, labels),
            "davies_bouldin": davies_bouldin_score(x_scaled, labels),
            "inertia": model.inertia_,
        })

    result = pd.DataFrame(rows)

    save_dataframe(
        result,
        os.path.join(tables_dir, "ocena_liczby_klastrow_kmeans.csv")
    )

    plt.figure(figsize=(9, 5.5))
    sns.lineplot(data=result, x="k", y="inertia", marker="o", color=PRIMARY_COLOR)
    plt.title(f"Metoda łokcia dla k-średnich - {hypothesis_name}", pad=14)
    plt.xlabel("Liczba klastrów")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "kmeans_elbow.png"), dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    sns.lineplot(data=result, x="k", y="silhouette", marker="o", color=SECONDARY_COLOR)
    plt.title(f"Silhouette score dla k-średnich - {hypothesis_name}", pad=14)
    plt.xlabel("Liczba klastrów")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "kmeans_silhouette.png"), dpi=220, bbox_inches="tight")
    plt.close()

    return result


def kmeans_cross_validation(x_scaled, hypothesis_name, n_clusters=N_CLUSTERS, n_splits=10):
    _, tables_dir, _ = get_paths(hypothesis_name)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_index, test_index) in enumerate(kfold.split(x_scaled), start=1):
        x_train = x_scaled[train_index]
        x_test = x_scaled[test_index]

        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
        model.fit(x_train)

        labels = model.predict(x_test)

        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(x_test, labels)
            calinski = calinski_harabasz_score(x_test, labels)
            davies = davies_bouldin_score(x_test, labels)
        else:
            silhouette = np.nan
            calinski = np.nan
            davies = np.nan

        rows.append({
            "fold": fold,
            "silhouette": silhouette,
            "calinski_harabasz": calinski,
            "davies_bouldin": davies,
        })

    result = pd.DataFrame(rows)

    save_dataframe(
        result,
        os.path.join(tables_dir, "kmeans_10fold_cv.csv")
    )

    summary = pd.DataFrame([{
        "silhouette_mean": result["silhouette"].mean(),
        "silhouette_std": result["silhouette"].std(),
        "calinski_mean": result["calinski_harabasz"].mean(),
        "calinski_std": result["calinski_harabasz"].std(),
        "davies_mean": result["davies_bouldin"].mean(),
        "davies_std": result["davies_bouldin"].std(),
    }])

    save_dataframe(
        summary,
        os.path.join(tables_dir, "kmeans_10fold_cv_podsumowanie.csv")
    )

    return result, summary


def run_kmeans(data, x_scaled, hypothesis_name, n_clusters=N_CLUSTERS):
    _, tables_dir, _ = get_paths(hypothesis_name)

    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(x_scaled)

    result = data.copy()
    result["cluster_kmeans"] = labels

    save_dataframe(
        result,
        os.path.join(tables_dir, "dane_z_klastrami_kmeans.csv")
    )

    metrics = pd.DataFrame([{
        "silhouette": silhouette_score(x_scaled, labels),
        "calinski_harabasz": calinski_harabasz_score(x_scaled, labels),
        "davies_bouldin": davies_bouldin_score(x_scaled, labels),
        "inertia": model.inertia_,
    }])

    save_dataframe(
        metrics,
        os.path.join(tables_dir, "kmeans_metryki.csv")
    )

    return result, model, metrics


def evaluate_em_cluster_count(x_scaled, hypothesis_name, component_range=range(2, 7)):
    _, tables_dir, plots_dir = get_paths(hypothesis_name)

    rows = []

    for n_components in component_range:
        model = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE)
        model.fit(x_scaled)
        labels = model.predict(x_scaled)

        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(x_scaled, labels)
            calinski = calinski_harabasz_score(x_scaled, labels)
            davies = davies_bouldin_score(x_scaled, labels)
        else:
            silhouette = np.nan
            calinski = np.nan
            davies = np.nan

        rows.append({
            "n_components": n_components,
            "bic": model.bic(x_scaled),
            "aic": model.aic(x_scaled),
            "silhouette": silhouette,
            "calinski_harabasz": calinski,
            "davies_bouldin": davies,
        })

    result = pd.DataFrame(rows)

    save_dataframe(
        result,
        os.path.join(tables_dir, "ocena_liczby_klastrow_em.csv")
    )

    plt.figure(figsize=(9, 5.5))
    sns.lineplot(data=result, x="n_components", y="bic", marker="o", color=COLOR_PALETTE[2])
    plt.title(f"Kryterium BIC dla EM - {hypothesis_name}", pad=14)
    plt.xlabel("Liczba komponentów")
    plt.ylabel("BIC")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "em_bic.png"), dpi=220, bbox_inches="tight")
    plt.close()

    return result


def run_em(data, x_scaled, hypothesis_name, n_components=N_CLUSTERS):
    _, tables_dir, _ = get_paths(hypothesis_name)

    model = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE)
    model.fit(x_scaled)

    labels = model.predict(x_scaled)
    probabilities = model.predict_proba(x_scaled)

    result = data.copy()
    result["cluster_em"] = labels
    result["cluster_em_probability"] = probabilities.max(axis=1)

    save_dataframe(
        result,
        os.path.join(tables_dir, "dane_z_klastrami_em.csv")
    )

    metrics = pd.DataFrame([{
        "silhouette": silhouette_score(x_scaled, labels),
        "calinski_harabasz": calinski_harabasz_score(x_scaled, labels),
        "davies_bouldin": davies_bouldin_score(x_scaled, labels),
        "bic": model.bic(x_scaled),
        "aic": model.aic(x_scaled),
        "srednie_prawdopodobienstwo_przynaleznosci": probabilities.max(axis=1).mean(),
    }])

    save_dataframe(
        metrics,
        os.path.join(tables_dir, "em_metryki.csv")
    )

    return result, model, metrics


def create_cluster_profiles(
    data,
    hypothesis_name,
    cluster_column,
    prefix,
    numeric_columns,
    categorical_columns,
):
    _, tables_dir, _ = get_paths(hypothesis_name)

    numeric_profile = data.groupby(cluster_column)[numeric_columns].mean().reset_index()

    save_dataframe(
        numeric_profile,
        os.path.join(tables_dir, f"{prefix}_profil_numeryczny.csv")
    )

    counts = data[cluster_column].value_counts().sort_index().reset_index()
    counts.columns = [cluster_column, "liczebnosc"]
    counts["procent"] = 100 * counts["liczebnosc"] / counts["liczebnosc"].sum()

    save_dataframe(
        counts,
        os.path.join(tables_dir, f"{prefix}_liczebnosci_klastrow.csv")
    )

    for column in categorical_columns:
        table = pd.crosstab(data[cluster_column], data[column], normalize="index") * 100
        table = table.reset_index()

        save_dataframe(
            table,
            os.path.join(tables_dir, f"{prefix}_profil_{column}.csv")
        )

    binary_columns = [
        column for column in numeric_columns
        if set(data[column].dropna().unique()).issubset({0, 1})
    ]

    for column in binary_columns:
        table = pd.crosstab(data[cluster_column], data[column], normalize="index") * 100
        table = table.reset_index()

        save_dataframe(
            table,
            os.path.join(tables_dir, f"{prefix}_profil_{column}_procentowo.csv")
        )

    return numeric_profile, counts


def plot_clusters_pca(x_scaled, labels, hypothesis_name, filename, title):
    _, _, plots_dir = get_paths(hypothesis_name)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(x_scaled)

    plot_data = pd.DataFrame({
        "PC1": coordinates[:, 0],
        "PC2": coordinates[:, 1],
        "cluster": labels,
    })

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=plot_data,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette=COLOR_PALETTE,
        s=70,
        alpha=0.8,
    )
    plt.title(title, pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=220, bbox_inches="tight")
    plt.close()


def plot_profile_heatmap(profile, cluster_column, hypothesis_name, filename, title):
    _, _, plots_dir = get_paths(hypothesis_name)

    values = profile.set_index(cluster_column)
    standardized = values.copy()

    for column in standardized.columns:
        std = standardized[column].std()
        if std == 0 or pd.isna(std):
            standardized[column] = 0
        else:
            standardized[column] = (standardized[column] - standardized[column].mean()) / std

    plt.figure(figsize=(10, 5.5))
    sns.heatmap(
        standardized,
        annot=True,
        fmt=".2f",
        cmap=sns.light_palette(PRIMARY_COLOR, as_cmap=True),
        linewidths=1,
        linecolor="white",
    )
    plt.title(title, pad=14)
    plt.xlabel("Zmienna")
    plt.ylabel("Klaster")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=220, bbox_inches="tight")
    plt.close()


def plot_profile_bars(profile, cluster_column, hypothesis_name, prefix):
    _, _, plots_dir = get_paths(hypothesis_name)

    for column in profile.columns:
        if column == cluster_column:
            continue

        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=profile,
            x=cluster_column,
            y=column,
            palette=COLOR_PALETTE,
        )
        plt.title(f"Średnia zmiennej {column} w klastrach - {prefix}", pad=14)
        plt.xlabel("Klaster")
        plt.ylabel(f"Średnia {column}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{prefix}_profil_{column}.png"), dpi=220, bbox_inches="tight")
        plt.close()


def compare_clusters_with_hypothesis(hypothesis_name, kmeans_profile, em_profile, numeric_columns):
    _, tables_dir, _ = get_paths(hypothesis_name)

    comparison = pd.DataFrame({
        "zmienna": numeric_columns,
        "kmeans_min": [kmeans_profile[column].min() for column in numeric_columns],
        "kmeans_max": [kmeans_profile[column].max() for column in numeric_columns],
        "em_min": [em_profile[column].min() for column in numeric_columns],
        "em_max": [em_profile[column].max() for column in numeric_columns],
    })

    comparison["roznica_kmeans"] = comparison["kmeans_max"] - comparison["kmeans_min"]
    comparison["roznica_em"] = comparison["em_max"] - comparison["em_min"]

    save_dataframe(
        comparison,
        os.path.join(tables_dir, "porownanie_klastrow_z_hipoteza.csv")
    )

    return comparison


def run_hypothesis(df, hypothesis_name, config):
    print(f"\n{hypothesis_name}")

    data, transformed_data, x_scaled, numeric_columns, categorical_columns = prepare_data_for_clustering(df, config)

    save_variable_table(hypothesis_name, config, numeric_columns, categorical_columns)

    kmeans_evaluation = evaluate_kmeans_cluster_count(x_scaled, hypothesis_name)
    kmeans_cv, kmeans_cv_summary = kmeans_cross_validation(x_scaled, hypothesis_name)

    kmeans_result, kmeans_model, kmeans_metrics = run_kmeans(data, x_scaled, hypothesis_name)
    kmeans_profile, kmeans_counts = create_cluster_profiles(
        kmeans_result,
        hypothesis_name,
        "cluster_kmeans",
        "kmeans",
        numeric_columns,
        categorical_columns,
    )

    plot_clusters_pca(
        x_scaled,
        kmeans_result["cluster_kmeans"],
        hypothesis_name,
        "kmeans_pca.png",
        f"Klastry k-średnich - {hypothesis_name}",
    )

    plot_profile_heatmap(
        kmeans_profile,
        "cluster_kmeans",
        hypothesis_name,
        "kmeans_profile_heatmap.png",
        f"Profil klastrów k-średnich - {hypothesis_name}",
    )

    plot_profile_bars(
        kmeans_profile,
        "cluster_kmeans",
        hypothesis_name,
        "kmeans",
    )

    em_evaluation = evaluate_em_cluster_count(x_scaled, hypothesis_name)

    em_result, em_model, em_metrics = run_em(data, x_scaled, hypothesis_name)
    em_profile, em_counts = create_cluster_profiles(
        em_result,
        hypothesis_name,
        "cluster_em",
        "em",
        numeric_columns,
        categorical_columns,
    )

    plot_clusters_pca(
        x_scaled,
        em_result["cluster_em"],
        hypothesis_name,
        "em_pca.png",
        f"Klastry EM - {hypothesis_name}",
    )

    plot_profile_heatmap(
        em_profile,
        "cluster_em",
        hypothesis_name,
        "em_profile_heatmap.png",
        f"Profil klastrów EM - {hypothesis_name}",
    )

    plot_profile_bars(
        em_profile,
        "cluster_em",
        hypothesis_name,
        "em",
    )

    comparison = compare_clusters_with_hypothesis(
        hypothesis_name,
        kmeans_profile,
        em_profile,
        numeric_columns,
    )

    return {
        "kmeans_evaluation": kmeans_evaluation,
        "kmeans_cv": kmeans_cv,
        "kmeans_cv_summary": kmeans_cv_summary,
        "kmeans_metrics": kmeans_metrics,
        "kmeans_profile": kmeans_profile,
        "kmeans_counts": kmeans_counts,
        "em_evaluation": em_evaluation,
        "em_metrics": em_metrics,
        "em_profile": em_profile,
        "em_counts": em_counts,
        "comparison": comparison,
    }


def save_global_summary(results):
    rows = []

    for hypothesis_name, result in results.items():
        kmeans_metrics = result["kmeans_metrics"].iloc[0]
        em_metrics = result["em_metrics"].iloc[0]

        rows.append({
            "hipoteza": hypothesis_name,
            "kmeans_silhouette": kmeans_metrics["silhouette"],
            "kmeans_calinski_harabasz": kmeans_metrics["calinski_harabasz"],
            "kmeans_davies_bouldin": kmeans_metrics["davies_bouldin"],
            "em_silhouette": em_metrics["silhouette"],
            "em_calinski_harabasz": em_metrics["calinski_harabasz"],
            "em_davies_bouldin": em_metrics["davies_bouldin"],
            "em_bic": em_metrics["bic"],
            "em_aic": em_metrics["aic"],
            "em_srednie_prawdopodobienstwo": em_metrics["srednie_prawdopodobienstwo_przynaleznosci"],
        })

    save_dataframe(
        pd.DataFrame(rows),
        os.path.join(OUTPUT_DIR, "podsumowanie_analizy_skupien.csv")
    )


def main():
    reset_output_directory()
    create_directories()
    set_plot_style()

    df = load_data()
    df = clean_data(df)

    results = {}

    for hypothesis_name, config in HYPOTHESES.items():
        results[hypothesis_name] = run_hypothesis(df, hypothesis_name, config)

    save_global_summary(results)

    print(f"\nWyniki zapisano w: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
