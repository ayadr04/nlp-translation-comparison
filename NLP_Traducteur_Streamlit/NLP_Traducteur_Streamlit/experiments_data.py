"""Chargement centralisé des données de toutes les expériences du projet.

Chaque expérience est définie par :
  - id          : clé interne
  - label       : nom affiché dans l'UI
  - description : courte description pour l'utilisateur
  - history_path: chemin vers le JSON d'historique d'entraînement
  - metrics_path: chemin vers le JSON de métriques finales
  - bleu_metric : clé JSON du BLEU dans le fichier metrics
  - translations_path : chemin vers le CSV de traductions (optionnel)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Racine du projet (remonter depuis NLP_Traducteur_Streamlit/NLP_Traducteur_Streamlit/)
APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent.parent  # .../nlp-translation-comparison
RESULTS_ROOT = REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# Définition des expériences
# ---------------------------------------------------------------------------

EXPERIMENTS: list[dict] = [
    {
        "id": "exp4_lstm",
        "label": "Exp 4 · LSTM sans attention",
        "short": "LSTM sans attention",
        "color": "#6175BB",
        "description": (
            "Modèle LSTM seq2seq de base, sans mécanisme d'attention. "
            "5 epochs, teacher forcing complet. "
            "Sert de baseline pour mesurer le gain apporté par l'attention."
        ),
        "architecture_details": {
            "Encodeur": "LSTM (unidirectionnel)",
            "Décodeur": "LSTM",
            "Attention": "Aucune",
            "Teacher Forcing": "100 %",
            "Epochs": 5,
            "Paramètres": "5 920 528",
        },
        "history_path": RESULTS_ROOT / "lstm_history.json",
        "metrics_path": RESULTS_ROOT / "lstm_metrics.json",
        "bleu_metric": "bleu",
        "translations_path": RESULTS_ROOT / "lstm_test_translations.csv",
        "translations_cols": {"source": "source", "reference": "reference", "prediction": "prediction"},
    },
    {
        "id": "exp5a_bahdanau",
        "label": "Exp 5a · LSTM + Attention Bahdanau",
        "short": "LSTM + Bahdanau (TF 0 %)",
        "color": "#A855F7",
        "description": (
            "LSTM avec mécanisme d'attention de Bahdanau. "
            "Teacher forcing désactivé (ratio 0 %). "
            "Premier ajout d'attention pour guider le décodeur."
        ),
        "architecture_details": {
            "Encodeur": "LSTM (unidirectionnel)",
            "Décodeur": "LSTM + Attention Bahdanau",
            "Attention": "Bahdanau (additive)",
            "Teacher Forcing": "0 %",
            "Epochs": 5,
            "Paramètres": "8 874 000",
        },
        "history_path": RESULTS_ROOT / "experience_5_bahdanau" / "lstm_history.json",
        "metrics_path": RESULTS_ROOT / "experience_5_bahdanau" / "lstm_metrics.json",
        "bleu_metric": "bleu",
        "translations_path": RESULTS_ROOT / "experience_5_bahdanau" / "lstm_test_translations.csv",
        "translations_cols": {"source": "source", "reference": "reference", "prediction": "prediction"},
    },
    {
        "id": "exp5b_tf100",
        "label": "Exp 5b · LSTM + Attention (TF=100 %)",
        "short": "LSTM + Bahdanau (TF 100 %)",
        "color": "#F59E0B",
        "description": (
            "Même architecture que 5a mais avec teacher forcing à 100 %. "
            "Permet d'observer l'effet du teacher forcing sur la convergence "
            "et le BLEU final."
        ),
        "architecture_details": {
            "Encodeur": "LSTM (unidirectionnel)",
            "Décodeur": "LSTM + Attention Bahdanau",
            "Attention": "Bahdanau (additive)",
            "Teacher Forcing": "100 %",
            "Epochs": 5,
            "Paramètres": "8 874 000",
        },
        "history_path": RESULTS_ROOT / "experience_5_bahdanau_tf100" / "lstm_history.json",
        "metrics_path": None,
        "bleu_metric": None,
        "translations_path": RESULTS_ROOT / "experience_5_bahdanau_tf100" / "validation_translations.csv",
        "translations_cols": {"source": "source", "reference": "reference", "prediction": "prediction"},
    },
    {
        "id": "exp5_finale",
        "label": "Exp 5 Finale · No-repeat trigram ⭐",
        "short": "LSTM + Attention (No-repeat)",
        "color": "#10B981",
        "description": (
            "Modèle final : LSTM + Attention Bahdanau avec décodage greedy "
            "et contrainte no-repeat-trigram. Détokenisation Moses. "
            "Meilleur BLEU obtenu sur l'ensemble de test."
        ),
        "architecture_details": {
            "Encodeur": "LSTM (unidirectionnel)",
            "Décodeur": "LSTM + Attention Bahdanau",
            "Attention": "Bahdanau (additive)",
            "Teacher Forcing": "50 %",
            "Décodage": "Greedy + No-repeat trigram",
            "Détokenisation": "Moses (en)",
            "Epochs": 5,
        },
        "history_path": RESULTS_ROOT / "experience_5_bahdanau" / "lstm_history.json",
        "metrics_path": RESULTS_ROOT / "experience_5_finale" / "metriques_finales.json",
        "bleu_metric": "test_bleu",
        "translations_path": RESULTS_ROOT / "experience_5_finale" / "traductions_test_finales.csv",
        "translations_cols": {"source": "source", "reference": "reference", "prediction": "prediction"},
    },
]

# Dictionnaire indexé par id pour accès rapide
EXPERIMENTS_BY_ID: dict[str, dict] = {e["id"]: e for e in EXPERIMENTS}


# ---------------------------------------------------------------------------
# Fonctions de chargement
# ---------------------------------------------------------------------------

def load_history(exp: dict) -> pd.DataFrame | None:
    """Charge l'historique d'entraînement depuis le JSON."""
    path: Path | None = exp.get("history_path")
    if path is None or not path.exists():
        return None
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(records)
        col_map = {}
        for col in df.columns:
            low = col.lower()
            if low == "epoch":
                col_map[col] = "epoch"
            elif low in ("train_loss", "loss", "training_loss"):
                col_map[col] = "train_loss"
            elif low in ("validation_loss", "val_loss"):
                col_map[col] = "val_loss"
        df = df.rename(columns=col_map)
        for c in ("epoch", "train_loss", "val_loss"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception:
        return None


def load_metrics(exp: dict) -> dict[str, Any] | None:
    """Charge les métriques finales depuis le JSON."""
    path: Path | None = exp.get("metrics_path")
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        bleu_key = exp.get("bleu_metric")
        if bleu_key and bleu_key in data:
            data["_bleu_display"] = round(float(data[bleu_key]), 4)
        return data
    except Exception:
        return None


def load_translations(exp: dict, n: int = 5000) -> pd.DataFrame | None:
    """Charge les traductions CSV de l'expérience (premières n lignes)."""
    path: Path | None = exp.get("translations_path")
    if path is None or not path.exists():
        return None
    col_map: dict = exp.get("translations_cols", {})
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", nrows=n, on_bad_lines="skip")
        reverse_map = {v: k for k, v in col_map.items()}
        df = df.rename(columns=reverse_map)
        # Fallback positional mapping si colonnes pas trouvées
        cols = list(df.columns)
        for i, col in enumerate(("source", "reference", "prediction")):
            if col not in df.columns and len(cols) > i:
                df = df.rename(columns={cols[i]: col})
        df = df.dropna(subset=["source"])
        return df.reset_index(drop=True)
    except Exception:
        return None


def bleu_comparison_df() -> pd.DataFrame:
    """Retourne un DataFrame de comparaison BLEU pour toutes les expériences."""
    rows = []
    for exp in EXPERIMENTS:
        metrics = load_metrics(exp)
        bleu = metrics.get("_bleu_display") if metrics else None
        rows.append({
            "id": exp["id"],
            "Architecture": exp["short"],
            "Label": exp["label"],
            "BLEU": bleu if bleu is not None else float("nan"),
            "color": exp["color"],
        })
    return pd.DataFrame(rows)
