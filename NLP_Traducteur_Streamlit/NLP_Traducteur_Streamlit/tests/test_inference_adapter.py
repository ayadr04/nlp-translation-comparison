import inference_adapter
from inference_adapter import clean_text


def test_clean_text_none_preserves_case():
    assert clean_text("  Bonjour !  ", "none") == "Bonjour !"


def test_clean_text_lower_basic():
    assert clean_text(" Salut, toi! ", "lower_basic") == "salut , toi !"


def test_missing_files_reports_model_weights():
    files = inference_adapter.missing_files()
    assert any("lstm_attention.keras" in file for file in files)
