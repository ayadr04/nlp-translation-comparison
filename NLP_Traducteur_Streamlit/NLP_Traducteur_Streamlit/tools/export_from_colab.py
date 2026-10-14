"""À utiliser depuis une cellule du notebook (importer/coller cette fonction).

Exemple :
    export_keras_model(
        model, tokenizer_fr, tokenizer_en,
        out_dir='/content/export_app',
        max_source_length=30, max_target_length=30,
        target_start_token='<start>', target_end_token='<end>',
    )
Puis télécharger les quatre fichiers du dossier export_app et les
placer dans `models/` de l'application.

Ne modifie PAS tes prétraitements dans le notebook pour correspondre
à cette fonction : les paramètres ici doivent reproduire ton entraînement.
"""
from __future__ import annotations
import json
from pathlib import Path


def export_keras_model(
    model, tokenizer_fr, tokenizer_en, *, out_dir,
    max_source_length, max_target_length,
    target_start_token, target_end_token,
    normalization="none", source_start_token="", source_end_token="",
):
    """Exporte un modèle Keras à deux entrées et ses Tokenizer Keras."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if len(model.inputs) != 2:
        raise ValueError(
            "Le modèle n'a pas 2 entrées. Exporte plutôt un modèle d'inférence "
            "à deux entrées ou adapte le connecteur du projet."
        )
    if int(max_target_length) <= 1 or int(max_source_length) <= 0:
        raise ValueError("Longueurs d'entrée/sortie invalides.")
    if not hasattr(tokenizer_fr, "to_json") or not hasattr(tokenizer_en, "to_json"):
        raise TypeError("Il faut des Tokenizer Keras. Pour des vocabulaires dict/PyTorch, adapter le connecteur.")
    model.save(out / "lstm_attention.keras")
    (out / "tokenizer_fr.json").write_text(tokenizer_fr.to_json(), encoding="utf-8")
    (out / "tokenizer_en.json").write_text(tokenizer_en.to_json(), encoding="utf-8")
    cfg = dict(
        max_source_length=int(max_source_length),
        max_target_length=int(max_target_length),
        normalization=normalization,
        source_start_token=source_start_token,
        source_end_token=source_end_token,
        target_start_token=target_start_token,
        target_end_token=target_end_token,
    )
    (out / "inference_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Export terminé dans :", out)
    print("Important : si tu as utilisé des classes personnalisées, ajoute-les dans model_custom_objects.py")
    print("Important : comparer les prédictions de l'application à une phrase de test du notebook.")
