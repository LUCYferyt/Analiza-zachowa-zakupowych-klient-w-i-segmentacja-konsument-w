import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    plot_tree,
    _tree,
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data", "marketing_campaign.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_iii")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
RULES_DIR = os.path.join(OUTPUT_DIR, "rules")


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


def create_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RULES_DIR, exist_ok=True)


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
    print("Czyszczenie danych...")

    df = df.copy()
    before = df.shape[0]

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

    after = df.shape[0]
    print(f"Usunięto {before - after} duplikatów. Nowy rozmiar: {df.shape}")

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

    print("\nTypy danych po kodowaniu:")
    print(data.dtypes)

    X = data[features]
    y = data[target]

    print(f"Zbiór po przygotowaniu: X={X.shape}, y={y.shape}")
    return X, y, encoders, data


#wykresy i reguły

def plot_feature_importance(model, feature_names, title, filename):
    importances = pd.DataFrame({
        "predyktor": feature_names,
        "waznosc": model.feature_importances_
    }).sort_values("waznosc", ascending=False)

    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=importances, x="waznosc", y="predyktor", palette=COLOR_PALETTE)
    plt.title(title, pad=14)
    plt.xlabel("Ważność")
    plt.ylabel("Predyktor")
    plt.tight_layout()
    plt.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Zapisano wykres ważności: {filename}")
    return importances


def plot_decision_tree_chart(model, feature_names, title, filename, class_names=None):
    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Zapisano wykres drzewa: {filename}")


def extract_rules_from_tree(model, feature_names, model_type="regression", class_names=None):
    tree_ = model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            recurse(tree_.children_left[node], path + [f"({name} <= {threshold:.3f})"])
            recurse(tree_.children_right[node], path + [f"({name} > {threshold:.3f})"])
        else:
            n_node_samples = tree_.n_node_samples[node]

            if model_type == "classification":
                values = tree_.value[node][0]
                predicted_class = int(np.argmax(values))
                confidence = values[predicted_class] / np.sum(values)

                rules.append({
                    "warunki": " and ".join(path) if path else "TRUE",
                    "przewidywana_klasa": class_names[predicted_class] if class_names else predicted_class,
                    "pewnosc": confidence,
                    "wsparcie": n_node_samples,
                })
            else:
                prediction = tree_.value[node][0][0]
                impurity = tree_.impurity[node]

                rules.append({
                    "warunki": " and ".join(path) if path else "TRUE",
                    "przewidywana_wartosc": prediction,
                    "wariancja_w_lisciu": impurity,
                    "wsparcie": n_node_samples,
                })

    recurse(0, [])
    return pd.DataFrame(rules)


def select_top_rules(rules_df: pd.DataFrame, model_type="regression", top_n=5):
    if model_type == "classification":
        return rules_df.sort_values(
            by=["pewnosc", "wsparcie"],
            ascending=[False, False]
        ).head(top_n)
    else:
        return rules_df.sort_values(
            by=["wariancja_w_lisciu", "wsparcie"],
            ascending=[True, False]
        ).head(top_n)


def format_rules_txt(rules_df: pd.DataFrame, filename: str, model_type="regression"):
    with open(filename, "w", encoding="utf-8") as f:
        for number, (_, row) in enumerate(rules_df.iterrows(), start=1):
            f.write(f"Reguła {number}\n")
            f.write(f"Warunki: {row['warunki']}\n")

            if model_type == "classification":
                f.write(f"Przewidywana klasa: {row['przewidywana_klasa']}\n")
                f.write(f"Pewność: {row['pewnosc']:.4f}\n")
                f.write(f"Wsparcie: {int(row['wsparcie'])}\n")
            else:
                f.write(f"Przewidywana wartość: {row['przewidywana_wartosc']:.4f}\n")
                f.write(f"Wariancja w liściu: {row['wariancja_w_lisciu']:.4f}\n")
                f.write(f"Wsparcie: {int(row['wsparcie'])}\n")

            f.write("\n")

    print(f"Zapisano reguły: {filename}")


#drzewa regresyjne

def run_regression_tree(hypothesis_name, X, y):
    print(f"Budowa drzewa regresyjnego dla {hypothesis_name}...")

    model = DecisionTreeRegressor(
        max_depth=4,
        min_samples_leaf=20,
        random_state=RANDOM_STATE
    )

    model.fit(X, y)
    y_pred = model.predict(X)

    resub_mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2"
    )

    metrics_df = pd.DataFrame([{
        "hipoteza": hypothesis_name,
        "koszt_resubstytucji_MSE": resub_mse,
        "R2": r2,
        "SK_srednie_R2": cv_scores.mean(),
        "SK_std_R2": cv_scores.std(),
    }])

    save_dataframe(metrics_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_metryki_regresja.csv"))

    importance_df = plot_feature_importance(
        model,
        X.columns.tolist(),
        f"Ważność predyktorów - {hypothesis_name}",
        os.path.join(PLOTS_DIR, f"{hypothesis_name}_waznosc_predyktorow.png")
    )
    save_dataframe(importance_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_waznosc_predyktorow.csv"))

    plot_decision_tree_chart(
        model,
        X.columns.tolist(),
        f"Drzewo regresyjne - {hypothesis_name}",
        os.path.join(PLOTS_DIR, f"{hypothesis_name}_drzewo.png")
    )

    rules_df = extract_rules_from_tree(
        model,
        X.columns.tolist(),
        model_type="regression"
    )

    top_rules = select_top_rules(rules_df, model_type="regression", top_n=5)

    save_dataframe(rules_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_wszystkie_reguly.csv"))
    save_dataframe(top_rules, os.path.join(TABLES_DIR, f"{hypothesis_name}_top_reguly.csv"))
    format_rules_txt(
        top_rules,
        os.path.join(RULES_DIR, f"{hypothesis_name}_reguly.txt"),
        model_type="regression"
    )

    return metrics_df


#drzewo klasifikacji

def run_classification_tree(hypothesis_name, X, y):
    print(f"Budowa drzewa klasyfikacyjnego dla {hypothesis_name}...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=20,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    total_error = 1 - acc

    metrics_df = pd.DataFrame([{
        "hipoteza": hypothesis_name,
        "accuracy": acc,
        "blad_calkowity": total_error,
    }])
    save_dataframe(metrics_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_metryki_klasyfikacja.csv"))

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    save_dataframe(report_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_classification_report.csv"))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm)
    save_dataframe(cm_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_macierz_klasyfikacji.csv"))

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(PRIMARY_COLOR, as_cmap=True))
    plt.title(f"Macierz klasyfikacji - {hypothesis_name}", pad=14)
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywista klasa")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{hypothesis_name}_macierz_klasyfikacji.png"), dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Zapisano macierz klasyfikacji dla {hypothesis_name}")

    importance_df = plot_feature_importance(
        model,
        X.columns.tolist(),
        f"Ważność predyktorów - {hypothesis_name}",
        os.path.join(PLOTS_DIR, f"{hypothesis_name}_waznosc_predyktorow.png")
    )
    save_dataframe(importance_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_waznosc_predyktorow.csv"))

    class_names = [str(c) for c in sorted(y.unique())]

    plot_decision_tree_chart(
        model,
        X.columns.tolist(),
        f"Drzewo klasyfikacyjne - {hypothesis_name}",
        os.path.join(PLOTS_DIR, f"{hypothesis_name}_drzewo.png"),
        class_names=class_names
    )

    rules_df = extract_rules_from_tree(
        model,
        X.columns.tolist(),
        model_type="classification",
        class_names=class_names
    )

    top_rules = select_top_rules(rules_df, model_type="classification", top_n=5)

    save_dataframe(rules_df, os.path.join(TABLES_DIR, f"{hypothesis_name}_wszystkie_reguly.csv"))
    save_dataframe(top_rules, os.path.join(TABLES_DIR, f"{hypothesis_name}_top_reguly.csv"))
    format_rules_txt(
        top_rules,
        os.path.join(RULES_DIR, f"{hypothesis_name}_reguly.txt"),
        model_type="classification"
    )

    return metrics_df



def main():

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
            metrics_df = run_regression_tree(hypothesis_name, X, y)

            summary_rows.append({
                "hipoteza": hypothesis_name,
                "typ": "regresja",
                "target": target,
                "R2": metrics_df.loc[0, "R2"],
                "koszt_resubstytucji_MSE": metrics_df.loc[0, "koszt_resubstytucji_MSE"],
                "SK_srednie_R2": metrics_df.loc[0, "SK_srednie_R2"],
            })

        else:
            metrics_df = run_classification_tree(hypothesis_name, X, y)

            summary_rows.append({
                "hipoteza": hypothesis_name,
                "typ": "klasyfikacja",
                "target": target,
                "accuracy": metrics_df.loc[0, "accuracy"],
                "blad_calkowity": metrics_df.loc[0, "blad_calkowity"],
            })

    summary_df = pd.DataFrame(summary_rows)
    save_dataframe(summary_df, os.path.join(TABLES_DIR, "podsumowanie_drzew.csv"))

    print("\n=== KONIEC ===")
    print(f"Wyniki zapisano w: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nWYSTĄPIŁ BŁĄD:")
        print(type(e).__name__, "-", e)