"""Connecteur pour un modèle TensorFlow/Keras seq2seq entraîné.

ATTENTION : il convient aux modèles teacher forcing à 2 entrées
[encodeur_ids, decodeur_ids] produisant (batch, longueur_cible, vocab_cible).
Si le notebook possède un décodeur pas à pas personnalisé, adapter
`translate` à la fonction de génération réellement utilisée dans ce notebook.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "lstm_attention.keras"
SRC_TOKENIZER_FILE = MODEL_DIR / "tokenizer_fr.json"
TGT_TOKENIZER_FILE = MODEL_DIR / "tokenizer_en.json"
CONFIG_FILE = MODEL_DIR / "inference_config.json"
NEEDED = (MODEL_FILE, SRC_TOKENIZER_FILE, TGT_TOKENIZER_FILE, CONFIG_FILE)


def missing_files() -> list[str]:
    return [str(path.relative_to(ROOT)) for path in NEEDED if not path.is_file()]


def clean_text(value: str, mode: str) -> str:
    value = value.strip()
    if mode == "none":
        return value
    if mode == "lowercase":
        return value.lower()
    if mode == "lower_basic":
        value = value.lower()
        value = unicodedata.normalize("NFKC", value)
        value = re.sub(r"([?.!,¿])", r" \1 ", value)
        return " ".join(value.split())
    raise ValueError(f"normalization invalide : {mode}")


class LiveTranslator:
    def __init__(self):
        missing = missing_files()
        if missing:
            raise FileNotFoundError("Fichiers manquants : " + ", ".join(missing))
        try:
            import numpy as np
            import tensorflow as tf
            from tensorflow.keras.preprocessing.text import tokenizer_from_json
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except ImportError as exc:
            raise RuntimeError("Installe TensorFlow pour activer la traduction directe : pip install tensorflow") from exc

        self.np = np
        self.pad_sequences = pad_sequences
        self.config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        self.max_src = int(self.config["max_source_length"])
        self.max_tgt = int(self.config["max_target_length"])
        self.mode = self.config.get("normalization", "none")
        if self.max_src <= 0 or self.max_tgt <= 1:
            raise ValueError("Longueurs source/cible invalides dans inference_config.json")
        self.source_tok = tokenizer_from_json(SRC_TOKENIZER_FILE.read_text(encoding="utf-8"))
        self.target_tok = tokenizer_from_json(TGT_TOKENIZER_FILE.read_text(encoding="utf-8"))
        self.start_token = self.config.get("target_start_token", "<start>")
        self.end_token = self.config.get("target_end_token", "<end>")
        self.start_id = self.target_tok.word_index.get(self.start_token)
        self.end_id = self.target_tok.word_index.get(self.end_token)
        if self.start_id is None or self.end_id is None:
            raise ValueError(
                "Tokens de début/fin introuvables dans le tokenizer cible. "
                "Modifie target_start_token / target_end_token dans la config."
            )
        # Si le notebook comporte des couches personnalisées, renseigner CUSTOM_OBJECTS.
        try:
            from model_custom_objects import CUSTOM_OBJECTS
        except ImportError:
            CUSTOM_OBJECTS = {}
        self.model = tf.keras.models.load_model(MODEL_FILE, custom_objects=CUSTOM_OBJECTS, compile=False)
        if len(self.model.inputs) != 2:
            raise ValueError("Ce connecteur nécessite un modèle Keras avec exactement 2 entrées. Adapte l'interface à ton décodeur.")
        self.reverse = {int(i): token for token, i in self.target_tok.word_index.items()}

    def translate(self, text: str) -> str:
        normalized = clean_text(text, self.mode)
        prefix = [self.config.get("source_start_token", "")]
        suffix = [self.config.get("source_end_token", "")]
        tokens = [token for token in prefix if token] + [normalized] + [token for token in suffix if token]
        ids = self.source_tok.texts_to_sequences([" ".join(tokens)])[0]
        if not ids:
            raise ValueError("La phrase ne contient aucun token reconnu par le vocabulaire source.")
        source = self.pad_sequences([ids], maxlen=self.max_src, padding="post", truncating="post")
        decoder = self.np.zeros((1, self.max_tgt), dtype="int32")
        decoder[0, 0] = self.start_id
        generated = []
        for step in range(self.max_tgt - 1):
            raw = self.model.predict([source, decoder], verbose=0)
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
            if getattr(raw, "ndim", None) != 3 or raw.shape[1] <= step:
                raise ValueError("Le modèle ne produit pas (batch, longueur_cible, vocabulaire). Vérifie sa fonction de décodage.")
            word_id = int(self.np.argmax(raw[0, step]))
            if word_id == self.end_id or word_id == 0:
                break
            token = self.reverse.get(word_id, "<UNK>")
            if token not in (self.start_token,):
                generated.append(token)
            decoder[0, step + 1] = word_id
        return " ".join(generated).strip() or "(aucun token généré)"
