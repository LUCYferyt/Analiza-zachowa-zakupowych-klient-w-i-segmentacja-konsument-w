import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data", "marketing_campaign.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_iv")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


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

RANDOM_STATE = 42

#zmienne DO klasteryzacji

CLUSTER_FEATURES = [
    "Income",
    "NumWebPurchases",
    "MntWines",
    "NumCatalogPurchases",
    "Recency",
    "Kidhome",
]

PROFILE_CATEGORICAL = [
    "Marital_Status",
    "Response",
]

PROFILE_NUMERICAL = [
    "Income",
    "NumWebPurchases",
    "MntWines",
    "NumCatalogPurchases",
    "Recency",
    "Kidhome",
]

N_CLUSTERS = 3



def create_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


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


def save_dataframe(df: pd.DataFrame, filename: str):
    df.to_csv(filename, index=False)
    print(f"Zapisano tabelę: {filename}")


def load_data(path: str) -> pd.DataFrame:
    print(f"Wczytywanie danych z: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku CSV pod ścieżką:\n{path}\n"
            f"Sprawdź, czy plik marketing_campaign.csv jest w folderze data."
        )

    df = pd.read_csv(path, sep="\t")
    print(f"Wczytano dane. Rozmiar: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    if "Marital_Status" in df.columns:
        df["Marital_Status"] = df["Marital_Status"].replace({
            "Together": "Partner",
            "Married": "Partner",
            "Single": "Single",
            "Divorced": "Single",
            "Widow": "Single",
            "YOLO": "Other",
            "Absurd": "Other",
        })

    return df


def prepare_clustering_data(df: pd.DataFrame):
    needed_cols = list(set(CLUSTER_FEATURES + PROFILE_CATEGORICAL + PROFILE_NUMERICAL))
    data = df[needed_cols].copy()
    data = data.dropna()

    X = data[CLUSTER_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Dane do klasteryzacji: {X.shape}")
    return data, X, X_scaled, scaler


#wybór liczby klastrów

def evaluate_k_range(X_scaled, k_range=range(2, 7)):
    rows = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_scaled)

        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        davies = davies_bouldin_score(X_scaled, labels)
        inertia = model.inertia_

        rows.append({
            "k": k,
            "silhouette": silhouette,
            "calinski_harabasz": calinski,
            "davies_bouldin": davies,
            "inertia": inertia,
        })

    results = pd.DataFrame(rows)
    save_dataframe(results, os.path.join(TABLES_DIR, "ocena_liczby_klastrow_kmeans.csv"))

    plt.figure(figsize=(9, 5.5))
    sns.lineplot(data=results, x="k", y="inertia", marker="o", color=PRIMARY_COLOR)
    plt.title("Metoda łokcia dla k-średnich", pad=14)
    plt.xlabel("Liczba klastrów")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "kmeans_elbow.png"), dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    sns.lineplot(data=results, x="k", y="silhouette", marker="o", color=COLOR_PALETTE[1])
    plt.title("Silhouette score dla k-średnich", pad=14)
    plt.xlabel("Liczba klastrów")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "kmeans_silhouette.png"), dpi=220, bbox_inches="tight")
    plt.close()

    return results


#k-means

def kmeans_cross_validation(X_scaled, n_clusters=3, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), start=1):
        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]

        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
        model.fit(X_train)

        test_labels = model.predict(X_test)

        if len(np.unique(test_labels)) > 1:
            silhouette = silhouette_score(X_test, test_labels)
            calinski = calinski_harabasz_score(X_test, test_labels)
            davies = davies_bouldin_score(X_test, test_labels)
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

    cv_df = pd.DataFrame(rows)
    save_dataframe(cv_df, os.path.join(TABLES_DIR, "kmeans_10fold_cv.csv"))

    summary_df = pd.DataFrame([{
        "silhouette_mean": cv_df["silhouette"].mean(),
        "silhouette_std": cv_df["silhouette"].std(),
        "calinski_mean": cv_df["calinski_harabasz"].mean(),
        "calinski_std": cv_df["calinski_harabasz"].std(),
        "davies_mean": cv_df["davies_bouldin"].mean(),
        "davies_std": cv_df["davies_bouldin"].std(),
    }])
    save_dataframe(summary_df, os.path.join(TABLES_DIR, "kmeans_10fold_cv_podsumowanie.csv"))

    return cv_df, summary_df


def run_kmeans(data: pd.DataFrame, X_scaled, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(X_scaled)

    result = data.copy()
    result["cluster_kmeans"] = labels

    save_dataframe(result, os.path.join(TABLES_DIR, "dane_z_klastrami_kmeans.csv"))

    metrics_df = pd.DataFrame([{
        "silhouette": silhouette_score(X_scaled, labels),
        "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, labels),
        "inertia": model.inertia_,
    }])
    save_dataframe(metrics_df, os.path.join(TABLES_DIR, "kmeans_metryki.csv"))

    return result, model, metrics_df


#em/ gaussian mixture


def evaluate_em_components(X_scaled, component_range=range(2, 7)):
    rows = []

    for k in component_range:
        model = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
        model.fit(X_scaled)
        labels = model.predict(X_scaled)

        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X_scaled, labels)
            calinski = calinski_harabasz_score(X_scaled, labels)
            davies = davies_bouldin_score(X_scaled, labels)
        else:
            silhouette = np.nan
            calinski = np.nan
            davies = np.nan

        rows.append({
            "n_components": k,
            "bic": model.bic(X_scaled),
            "aic": model.aic(X_scaled),
            "silhouette": silhouette,
            "calinski_harabasz": calinski,
            "davies_bouldin": davies,
        })

    results = pd.DataFrame(rows)
    save_dataframe(results, os.path.join(TABLES_DIR, "ocena_liczby_klastrow_em.csv"))

    plt.figure(figsize=(9, 5.5))
    sns.lineplot(data=results, x="n_components", y="bic", marker="o", color=COLOR_PALETTE[2])
    plt.title("Kryterium BIC dla modelu EM", pad=14)
    plt.xlabel("Liczba komponentów")
    plt.ylabel("BIC")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "em_bic.png"), dpi=220, bbox_inches="tight")
    plt.close()

    return results


def run_em(data: pd.DataFrame, X_scaled, n_components=3):
    model = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE)
    model.fit(X_scaled)
    labels = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    result = data.copy()
    result["cluster_em"] = labels
    result["cluster_em_probability"] = probs.max(axis=1)

    save_dataframe(result, os.path.join(TABLES_DIR, "dane_z_klastrami_em.csv"))

    metrics_df = pd.DataFrame([{
        "silhouette": silhouette_score(X_scaled, labels),
        "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, labels),
        "bic": model.bic(X_scaled),
        "aic": model.aic(X_scaled),
        "srednie_prawdopodobienstwo_przynaleznosci": probs.max(axis=1).mean(),
    }])
    save_dataframe(metrics_df, os.path.join(TABLES_DIR, "em_metryki.csv"))

    return result, model, metrics_df


#charakterystyki klastrow

def cluster_profiles(data: pd.DataFrame, cluster_col: str, prefix: str):
    numeric_profile = data.groupby(cluster_col)[PROFILE_NUMERICAL].mean().reset_index()
    save_dataframe(numeric_profile, os.path.join(TABLES_DIR, f"{prefix}_profil_numeryczny.csv"))

    counts = data[cluster_col].value_counts().sort_index().reset_index()
    counts.columns = [cluster_col, "liczebnosc"]
    counts["procent"] = 100 * counts["liczebnosc"] / counts["liczebnosc"].sum()
    save_dataframe(counts, os.path.join(TABLES_DIR, f"{prefix}_liczebnosci_klastrow.csv"))

    for cat_col in PROFILE_CATEGORICAL:
        table = pd.crosstab(data[cluster_col], data[cat_col], normalize="index") * 100
        table = table.reset_index()
        save_dataframe(table, os.path.join(TABLES_DIR, f"{prefix}_profil_{cat_col}.csv"))

    return numeric_profile, counts


#wizualizacja klastrow

def plot_clusters_pca(X_scaled, labels, filename, title):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "cluster": labels
    })

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette=COLOR_PALETTE,
        s=70,
        alpha=0.8
    )
    plt.title(title, pad=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close()


def plot_cluster_profiles(numeric_profile: pd.DataFrame, cluster_col: str, filename: str, title: str):
    plot_df = numeric_profile.melt(id_vars=cluster_col, var_name="zmienna", value_name="srednia")

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=plot_df,
        x="zmienna",
        y="srednia",
        hue=cluster_col,
        palette=COLOR_PALETTE
    )
    plt.title(title, pad=14)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close()




def compare_with_hypotheses(kmeans_profile: pd.DataFrame, em_profile: pd.DataFrame):
    comparison = pd.DataFrame({
        "zmienna": PROFILE_NUMERICAL,
        "srednia_kmeans_min": [kmeans_profile[col].min() for col in PROFILE_NUMERICAL],
        "srednia_kmeans_max": [kmeans_profile[col].max() for col in PROFILE_NUMERICAL],
        "srednia_em_min": [em_profile[col].min() for col in PROFILE_NUMERICAL],
        "srednia_em_max": [em_profile[col].max() for col in PROFILE_NUMERICAL],
    })

    save_dataframe(comparison, os.path.join(TABLES_DIR, "porownanie_klastrow_z_hipotezami.csv"))
    return comparison




def main():

    create_directories()
    set_plot_style()

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"FILE_PATH: {FILE_PATH}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    df = load_data(FILE_PATH)
    df = clean_data(df)

    data, X, X_scaled, scaler = prepare_clustering_data(df)

    print("\nOcena liczby klastrów dla k-średnich...")
    kmeans_eval = evaluate_k_range(X_scaled, k_range=range(2, 7))

    print("\nOcena liczby komponentów dla EM...")
    em_eval = evaluate_em_components(X_scaled, component_range=range(2, 7))

    print("\n10-krotny sprawdzian krzyżowy dla k-średnich...")
    kmeans_cv_df, kmeans_cv_summary = kmeans_cross_validation(X_scaled, n_clusters=N_CLUSTERS, n_splits=10)

    print("\nUruchamianie k-średnich...")
    kmeans_result, kmeans_model, kmeans_metrics = run_kmeans(data, X_scaled, n_clusters=N_CLUSTERS)

    print("\nProfilowanie klastrów k-średnich...")
    kmeans_numeric_profile, kmeans_counts = cluster_profiles(kmeans_result, "cluster_kmeans", "kmeans")

    plot_clusters_pca(
        X_scaled,
        kmeans_result["cluster_kmeans"],
        os.path.join(PLOTS_DIR, "kmeans_pca.png"),
        "Klastry k-średnich w przestrzeni PCA"
    )

    plot_cluster_profiles(
        kmeans_numeric_profile,
        "cluster_kmeans",
        os.path.join(PLOTS_DIR, "kmeans_profile.png"),
        "Charakterystyka skupień - k-średnich"
    )

    print("\nUruchamianie EM...")
    em_result, em_model, em_metrics = run_em(data, X_scaled, n_components=N_CLUSTERS)

    print("\nProfilowanie klastrów EM...")
    em_numeric_profile, em_counts = cluster_profiles(em_result, "cluster_em", "em")

    plot_clusters_pca(
        X_scaled,
        em_result["cluster_em"],
        os.path.join(PLOTS_DIR, "em_pca.png"),
        "Klastry EM w przestrzeni PCA"
    )

    plot_cluster_profiles(
        em_numeric_profile,
        "cluster_em",
        os.path.join(PLOTS_DIR, "em_profile.png"),
        "Charakterystyka skupień - EM"
    )

    print("\nPorównanie skupień z hipotezami...")
    comparison_df = compare_with_hypotheses(kmeans_numeric_profile, em_numeric_profile)

    print("\n=== KONIEC ===")
    print(f"Wyniki zapisano w: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nWYSTĄPIŁ BŁĄD:")
        print(type(e).__name__, "-", e)