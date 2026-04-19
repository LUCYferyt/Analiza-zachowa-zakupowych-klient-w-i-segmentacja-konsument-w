import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# =========================
# ŚCIEŻKI
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data", "marketing_campaign.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_v")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# =========================
# STYL
# =========================

COLOR_PALETTE = [
    "#675285",
    "#D16BA5",
    "#6C63FF",
    "#00B4D8",
    "#A23E48",
    "#F4A6C1",
    "#B8C0FF",
]

PRIMARY_COLOR = COLOR_PALETTE[0]
TEXT_COLOR = "#4A3B5F"
GRID_COLOR = "#E6DFF0"
EDGE_COLOR = "#D8CBE6"
AXIS_BACKGROUND = "#FCF8FD"
FIGURE_BACKGROUND = "#FFFDFE"

RANDOM_STATE = 42

# =========================
# HIPOTEZY
# =========================

HYPOTHESES = {
    "hipoteza_1": {
        "target": "NumWebPurchases",
        "features": ["Income", "Recency", "NumWebVisitsMonth"],
        "type": "regression",
    },
    "hipoteza_2": {
        "target": "MntWines",
        "features": ["Income", "Kidhome", "Marital_Status"],
        "type": "regression",
    },
    "hipoteza_3": {
        "target": "Response",
        "features": ["Income", "Recency", "NumCatalogPurchases"],
        "type": "classification",
    },
}

# =========================
# FUNKCJE OGÓLNE
# =========================

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


def prepare_dataset(df: pd.DataFrame, target: str, features: list[str]):
    print(f"Przygotowanie danych dla target={target}, features={features}")

    data = df[features + [target]].copy()
    data = data.dropna()

    encoders = {}

    for col in features:
        if not pd.api.types.is_numeric_dtype(data[col]):
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

            mapping_df = pd.DataFrame({
                "kategoria": le.classes_,
                "kod": list(range(len(le.classes_)))
            })
            save_dataframe(mapping_df, os.path.join(TABLES_DIR, f"mapowanie_{col}.csv"))

    if not pd.api.types.is_numeric_dtype(data[target]):
        le = LabelEncoder()
        data[target] = le.fit_transform(data[target].astype(str))
        encoders[target] = le

        mapping_df = pd.DataFrame({
            "kategoria": le.classes_,
            "kod": list(range(len(le.classes_)))
        })
        save_dataframe(mapping_df, os.path.join(TABLES_DIR, f"mapowanie_{target}.csv"))

    X = data[features]
    y = data[target]

    print(f"Zbiór po przygotowaniu: X={X.shape}, y={y.shape}")
    return X, y, encoders, data


def plot_feature_importance(importances_df: pd.DataFrame, title: str, filename: str):
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=importances_df, x="waznosc", y="predyktor", palette=COLOR_PALETTE)
    plt.title(title, pad=14)
    plt.xlabel("Ważność")
    plt.ylabel("Predyktor")
    plt.tight_layout()
    plt.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Zapisano wykres: {filename}")


# =========================
# REGRESJA - RANDOM FOREST
# =========================

def run_rf_regression(hypothesis_name, X, y):
    print(f"Trenowanie Random Forest Regressor dla {hypothesis_name}...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    metrics_df = pd.DataFrame([{
        "hipoteza": hypothesis_name,
        "train_R2": train_r2,
        "test_R2": test_r2,
        "train_MSE": train_mse,
        "test_MSE": test_mse,
        "train_RMSE": train_rmse,
        "test_RMSE": test_rmse,
        "train_MAE": train_mae,
        "test_MAE": test_mae,
        "CV_mean_R2": cv_scores.mean(),
        "CV_std_R2": cv_scores.std(),
    }])

    save_dataframe(metrics_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_rf_metryki_regresja.csv"))

    importances_df = pd.DataFrame({
        "predyktor": X.columns,
        "waznosc": model.feature_importances_
    }).sort_values("waznosc", ascending=False)

    save_dataframe(importances_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_rf_waznosc_predyktorow.csv"))

    plot_feature_importance(
        importances_df,
        f"Random Forest - ważność predyktorów - {hypothesis_name}",
        os.path.join(PLOTS_DIR, f"{hypothesis_name}_rf_waznosc_predyktorow.png")
    )

    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred_test
    })
    save_dataframe(pred_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_rf_predykcje_test.csv"))

    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_test, y=y_pred_test, color=PRIMARY_COLOR, alpha=0.75)
    plt.xlabel("Wartości rzeczywiste")
    plt.ylabel("Wartości przewidywane")
    plt.title(f"Random Forest - rzeczywiste vs przewidywane - {hypothesis_name}", pad=14)

    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], color=COLOR_PALETTE[1], linewidth=2)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{hypothesis_name}_rf_real_vs_pred.png"), dpi=220, bbox_inches="tight")
    plt.close()

    return metrics_df, importances_df


# =========================
# KLASYFIKACJA - RANDOM FOREST
# =========================

def run_rf_classification(hypothesis_name, X, y):
    print(f"Trenowanie Random Forest Classifier dla {hypothesis_name}...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    metrics_df = pd.DataFrame([{
        "hipoteza": hypothesis_name,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "CV_mean_accuracy": cv_scores.mean(),
        "CV_std_accuracy": cv_scores.std(),
        "blad_calkowity": 1 - test_acc,
    }])

    save_dataframe(metrics_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_rf_metryki_klasyfikacja.csv"))

    importances_df = pd.DataFrame({
        "predyktor": X.columns,
        "waznosc": model.feature_importances_
    }).sort_values("waznosc", ascending=False)

    save_dataframe(importances_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_rf_waznosc_predyktorow.csv"))

    plot_feature_importance(
        importances_df,
        f"Random Forest - ważność predyktorów - {hypothesis_name}",
        os.path.join(PLOTS_DIR, f"{hypothesis_name}_rf_waznosc_predyktorow.png")
    )

    cm = confusion_matrix(y_test, y_pred_test)
    cm_df = pd.DataFrame(cm)
    save_dataframe(cm_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_rf_macierz_klasyfikacji.csv"))

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(PRIMARY_COLOR, as_cmap=True))
    plt.title(f"Random Forest - macierz klasyfikacji - {hypothesis_name}", pad=14)
    plt.xlabel("Predykcja")
    plt.ylabel("Klasa rzeczywista")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{hypothesis_name}_rf_macierz_klasyfikacji.png"), dpi=220, bbox_inches="tight")
    plt.close()

    report = classification_report(y_test, y_pred_test, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    save_dataframe(report_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_rf_classification_report.csv"))

    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred_test
    })
    save_dataframe(pred_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_rf_predykcje_test.csv"))

    return metrics_df, importances_df


# =========================
# MAIN
# =========================

def main():
    print("=== START V. Wybrany algorytm data mining ===")

    create_directories()
    set_plot_style()

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"FILE_PATH: {FILE_PATH}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    df = load_data(FILE_PATH)
    df = clean_data(df)

    summary_rows = []

    for hypothesis_name, config in HYPOTHESES.items():
        print("\n" + "=" * 60)
        print(f"Przetwarzanie: {hypothesis_name}")
        print("=" * 60)

        target = config["target"]
        features = config["features"]
        model_type = config["type"]

        X, y, encoders, data = prepare_dataset(df, target, features)

        if model_type == "regression":
            metrics_df, importances_df = run_rf_regression(hypothesis_name, X, y)

            summary_rows.append({
                "hipoteza": hypothesis_name,
                "typ": "regresja",
                "target": target,
                "test_R2": metrics_df.loc[0, "test_R2"],
                "test_RMSE": metrics_df.loc[0, "test_RMSE"],
                "CV_mean_R2": metrics_df.loc[0, "CV_mean_R2"],
                "najwazniejszy_predyktor": importances_df.iloc[0]["predyktor"],
            })

        else:
            metrics_df, importances_df = run_rf_classification(hypothesis_name, X, y)

            summary_rows.append({
                "hipoteza": hypothesis_name,
                "typ": "klasyfikacja",
                "target": target,
                "test_accuracy": metrics_df.loc[0, "test_accuracy"],
                "f1": metrics_df.loc[0, "f1"],
                "CV_mean_accuracy": metrics_df.loc[0, "CV_mean_accuracy"],
                "najwazniejszy_predyktor": importances_df.iloc[0]["predyktor"],
            })

    summary_df = pd.DataFrame(summary_rows)
    save_dataframe(summary_df, os.path.join(TABLES_DIR, "rf_podsumowanie.csv"))

    print("\n=== KONIEC ===")
    print(f"Wyniki zapisano w: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nWYSTĄPIŁ BŁĄD:")
        print(type(e).__name__, "-", e)