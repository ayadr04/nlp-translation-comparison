"""Chargement et normalisation des résultats du projet de traduction FR -> EN."""
from __future__ import annotations

import io
from pathlib import Path
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _key(name: object) -> str:
    s = unicodedata.normalize("NFKD", str(name).lower().strip())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return "_".join(s.replace("-", " ").split())


def read_csv(source) -> pd.DataFrame:
    """Lit les CSV anglais/français (séparateur , ou ;, UTF-8 ou latin-1)."""
    data = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    if isinstance(data, str):
        data = data.encode("utf-8")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Encodage CSV non pris en charge.")


PREDICTION_ALIASES = {
    "source": ("source", "fr", "francais", "french", "texte_fr", "phrase_fr", "input"),
    "reference": ("reference", "ref", "target", "anglais", "english", "en", "traduction_reference"),
    "lstm_sans_attention": (
        "lstm_sans_attention", "prediction_lstm_sans_attention", "prediction_sans_attention",
        "pred_sans_attention", "lstm_no_attention", "pred_lstm", "prediction_lstm",
    ),
    "lstm_avec_attention": (
        "lstm_avec_attention", "prediction_lstm_avec_attention", "prediction_avec_attention",
        "prediction_attention", "pred_attention", "lstm_attention", "attention",
    ),
    "rnn": ("rnn", "prediction_rnn", "pred_rnn"),
    "gru": ("gru", "prediction_gru", "pred_gru"),
    "index": ("index", "id", "test_id", "id_phrase"),
}

DISPLAY_MODELS = {
    "rnn": "RNN",
    "gru": "GRU",
    "lstm_sans_attention": "LSTM sans attention",
    "lstm_avec_attention": "LSTM avec attention",
}


def canonical_predictions(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {_key(c): c for c in df.columns}
    changes = {}
    for dest, aliases in PREDICTION_ALIASES.items():
        for alias in aliases:
            if _key(alias) in lookup:
                changes[lookup[_key(alias)]] = dest
                break
    df = df.rename(columns=changes).copy()
    if "source" not in df.columns or "reference" not in df.columns:
        raise ValueError("Colonnes obligatoires : source et reference.")
    for col in DISPLAY_MODELS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df.dropna(subset=["source"])
    return df.reset_index(drop=True)


def merge_experiment_csv() -> pd.DataFrame | None:
    """Accepte le format notebook : index,source,reference,prediction."""
    locations = {
        "lstm_sans_attention": (
            RESULTS / "experience_2" / "predictions_test.csv",
            RESULTS / "experience_2" / "predictions.csv",
            RESULTS / "predictions_lstm_sans_attention.csv",
        ),
        "lstm_avec_attention": (
            RESULTS / "experience_5" / "predictions_test.csv",
            RESULTS / "experience_5" / "predictions.csv",
            RESULTS / "predictions_lstm_avec_attention.csv",
        ),
    }
    merged = None
    for model, paths in locations.items():
        path = next((path for path in paths if path.exists()), None)
        if path is None:
            continue
        df = read_csv(path)
        lookup = {_key(c): c for c in df.columns}
        if not all(k in lookup for k in ("source", "reference", "prediction")):
            continue
        chosen = df.rename(columns={lookup["source"]: "source", lookup["reference"]: "reference", lookup["prediction"]: model})
        chosen = chosen[["source", "reference", model]].drop_duplicates(subset=["source", "reference"])
        # Rejoindre sur la paire exacte pour ne pas comparer des phrases différentes.
        merged = chosen if merged is None else merged.merge(chosen, how="outer", on=["source", "reference"])
    return canonical_predictions(merged) if merged is not None else None


def default_predictions() -> tuple[pd.DataFrame | None, str]:
    main = RESULTS / "predictions_comparees.csv"
    if main.exists():
        df = read_csv(main)
        if not df.empty:
            return canonical_predictions(df), "CSV comparatif du projet"
    experiments = merge_experiment_csv()
    if experiments is not None and not experiments.empty:
        return experiments, "CSV des expériences 2 et/ou 5"
    sample = RESULTS / "exemples_historiques.csv"
    if sample.exists():
        return canonical_predictions(read_csv(sample)), "Exemples historiques partiels"
    return None, "Aucun fichier de prédictions"


def canonical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {_key(c): c for c in df.columns}
    aliases = {
        "modele": ["modele", "model", "architecture"],
        "bleu": ["bleu", "score_bleu", "bleu_score"],
        "chrf": ["chrf", "chr_f", "score_chrf"],
        "ter": ["ter", "score_ter"],
    }
    remap = {}
    for key, opts in aliases.items():
        for candidate in opts:
            if _key(candidate) in lookup:
                remap[lookup[_key(candidate)]] = key
                break
    df = df.rename(columns=remap)
    if "modele" not in df.columns:
        raise ValueError("metrics.csv doit contenir la colonne modele.")
    mapping = {_key(label): label for label in DISPLAY_MODELS.values()}
    mapping.update({k: v for k, v in DISPLAY_MODELS.items()})
    df = df.copy()
    df["modele"] = df["modele"].apply(lambda x: mapping.get(_key(x), str(x)))
    for column in ("bleu", "chrf", "ter"):
        if column not in df:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column].astype("string").str.replace(",", ".", regex=False), errors="coerce")
    return df[["modele", "bleu", "chrf", "ter"]]


def canonical_losses(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {_key(c): c for c in df.columns}
    aliases = {
        "modele": ["modele", "model"],
        "epoch": ["epoch", "epoque", "epochs"],
        "train_loss": ["train_loss", "loss", "perte_entrainement", "training_loss"],
        "val_loss": ["val_loss", "validation_loss", "perte_validation"],
    }
    remap = {}
    for dest, opts in aliases.items():
        for opt in opts:
            if _key(opt) in lookup:
                remap[lookup[_key(opt)]] = dest
                break
    df = df.rename(columns=remap).copy()
    if "epoch" not in df or "train_loss" not in df:
        raise ValueError("losses.csv doit contenir epoch et train_loss (ou loss).")
    if "modele" not in df:
        df["modele"] = "LSTM avec attention"
    if "val_loss" not in df:
        df["val_loss"] = pd.NA
    for col in ("epoch", "train_loss", "val_loss"):
        df[col] = pd.to_numeric(df[col].astype("string").str.replace(",", ".", regex=False), errors="coerce")
    return df.dropna(subset=["epoch", "train_loss"])[["modele", "epoch", "train_loss", "val_loss"]]


def load_optional_csv(path: Path, converter):
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = read_csv(path)
    return converter(df) if not df.empty else None
