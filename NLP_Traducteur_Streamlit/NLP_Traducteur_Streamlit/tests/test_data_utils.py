from pathlib import Path
import pandas as pd
import pytest
import data_utils
from data_utils import canonical_predictions, canonical_metrics, canonical_losses, read_csv


def test_canonical_predictions_fills_missing_models():
    df = pd.DataFrame({"Source": ["Bonjour"], "REFERENCE": ["Hello"], "prediction_attention": ["Hello!"]})
    out = canonical_predictions(df)
    assert out.iloc[0]["lstm_avec_attention"] == "Hello!"
    assert pd.isna(out.iloc[0]["lstm_sans_attention"])


def test_canonical_predictions_requires_source_reference():
    with pytest.raises(ValueError, match="source et reference"):
        canonical_predictions(pd.DataFrame({"prediction": ["test"]}))


def test_metric_translation_and_french_decimal():
    out = canonical_metrics(pd.DataFrame({"modèle": ["lstm_avec_attention"], "bleu": ["2,25"], "TER": ["98,7"]}))
    assert out.iloc[0]["modele"] == "LSTM avec attention"
    assert out.iloc[0]["bleu"] == pytest.approx(2.25)
    assert out.iloc[0]["ter"] == pytest.approx(98.7)


def test_losses_support_english_keras_headers():
    df = pd.DataFrame({"epoch": [1, 2], "loss": [2.0, 1.8], "val_loss": [2.1, 1.9]})
    out = canonical_losses(df)
    assert len(out) == 2
    assert out.iloc[1]["train_loss"] == pytest.approx(1.8)


def test_read_semicolon_csv(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("source;reference\nbonjour;hello\n", encoding="utf-8")
    out = read_csv(file)
    assert out.columns.tolist() == ["source", "reference"]


def test_merge_experiment_files(tmp_path, monkeypatch):
    monkeypatch.setattr(data_utils, "RESULTS", tmp_path)
    (tmp_path / "experience_2").mkdir()
    (tmp_path / "experience_5").mkdir()
    (tmp_path / "experience_2" / "predictions_test.csv").write_text(
        "index,source,reference,prediction\n0,bonjour,hello,helloo\n1,merci,thanks,thanks\n", encoding="utf-8"
    )
    (tmp_path / "experience_5" / "predictions_test.csv").write_text(
        "index,source,reference,prediction\n0,bonjour,hello,hello\n1,autre,other,other\n", encoding="utf-8"
    )
    out = data_utils.merge_experiment_csv()
    assert len(out) == 3
    bonjour = out.loc[out["source"] == "bonjour"].iloc[0]
    assert bonjour["lstm_sans_attention"] == "helloo"
    assert bonjour["lstm_avec_attention"] == "hello"
