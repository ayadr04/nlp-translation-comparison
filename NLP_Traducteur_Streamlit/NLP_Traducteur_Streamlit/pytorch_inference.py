"""Module d'inférence PyTorch pour le modèle LSTM + Attention de Bahdanau.
Charge le checkpoint exporté lstm_attention_final.pt et effectue la traduction en direct.
"""
from __future__ import annotations

from pathlib import Path
import re
import unicodedata
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3
TOKEN_PATTERN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+|[^\w\s]", flags=re.UNICODE)

# Emplacements possibles du fichier modèle
CURRENT_DIR = Path(__file__).resolve().parent
POSSIBLE_MODEL_PATHS = [
    CURRENT_DIR / "models" / "lstm_attention_final.pt",
    CURRENT_DIR.parent.parent / "models" / "lstm_attention_final.pt",
    CURRENT_DIR.parent / "models" / "lstm_attention_final.pt",
    CURRENT_DIR / "models" / "lstm_attention_final2.pt",
]


def find_model_path() -> Path | None:
    for p in POSSIBLE_MODEL_PATHS:
        if p.is_file():
            return p
    return None


def clean_text(text: str, lowercase: bool = True) -> str:
    if not isinstance(text, str):
        raise TypeError("Une phrase doit être une chaîne de caractères.")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'")
    text = " ".join(text.split())
    return text.lower() if lowercase else text


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    return TOKEN_PATTERN.findall(clean_text(text, lowercase=lowercase))


def encode_source(text: str, vocab_fr: dict, lowercase: bool = True) -> list[int]:
    tokens = tokenize(text, lowercase)
    return [vocab_fr.get(token, UNK_ID) for token in tokens] + [EOS_ID]


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:\'])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"'\s+([a-zA-Z])", r"'\1", text)
    text = re.sub(r"([a-zA-Z])\s+'", r"\1'", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor):
        embedded = self.embedding(src)
        packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=src.size(1))
        return encoder_outputs, hidden, cell


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.encoder_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_projection = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, encoder_outputs: torch.Tensor, src_mask: torch.Tensor):
        energy = torch.tanh(
            self.encoder_projection(encoder_outputs)
            + self.decoder_projection(query).unsqueeze(1)
        )
        scores = self.score_projection(energy).squeeze(-1)
        scores = scores.masked_fill(~src_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor, encoder_outputs: torch.Tensor, src_mask: torch.Tensor):
        embedded = self.embedding(token)
        context, weights = self.attention(hidden[-1], encoder_outputs, src_mask)
        lstm_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        logits = self.output_layer(torch.cat([output.squeeze(1), context], dim=-1))
        return logits, hidden, cell, weights


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder: EncoderLSTM, decoder: DecoderLSTM):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


class PyTorchLiveTranslator:
    def __init__(self, checkpoint_path: Path | str | None = None):
        if checkpoint_path is None:
            checkpoint_path = find_model_path()
        if checkpoint_path is None or not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Fichier de modèle introuvable. Emplacements testés : {POSSIBLE_MODEL_PATHS}")

        self.path = Path(checkpoint_path)
        data = torch.load(self.path, map_location="cpu", weights_only=False)

        self.vocab_fr = data["vocab_fr"]
        self.vocab_en = data["vocab_en"]
        self.id_to_token_en = {int(idx): tok for tok, idx in self.vocab_en.items()}
        self.config = data.get("config", {})
        self.decoding = data.get("decoding", {})

        # Architecture dimensions
        embedding_dim = int(self.config.get("embedding_dim", 256))
        hidden_dim = int(self.config.get("hidden_dim", 512))
        num_layers = int(self.config.get("num_layers", 1))

        encoder = EncoderLSTM(
            vocab_size=len(self.vocab_fr),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        decoder = DecoderLSTM(
            vocab_size=len(self.vocab_en),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.model = Seq2SeqLSTM(encoder, decoder)
        self.model.load_state_dict(data["model_state_dict"])
        self.model.eval()

        self.max_steps = int(self.decoding.get("max_generation_steps", 100))
        self.no_repeat_ngram_size = int(self.decoding.get("no_repeat_ngram_size", 3))

    @torch.no_grad()
    def translate(self, text: str) -> str:
        src_ids = encode_source(text, self.vocab_fr, lowercase=True)
        # S'il n'y a que le token EOS
        if len(src_ids) <= 1:
            return "(texte vide ou non reconnu)"

        src = torch.tensor([src_ids], dtype=torch.long)
        lengths = torch.tensor([len(src_ids)], dtype=torch.long)

        encoder_outputs, hidden, cell = self.model.encoder(src, lengths)
        mask = src.ne(PAD_ID)

        sequence = []
        token = torch.tensor([SOS_ID], dtype=torch.long)

        for _ in range(self.max_steps):
            logits, hidden, cell, _ = self.model.decoder(token, hidden, cell, encoder_outputs, mask)

            # Contrainte no-repeat trigram
            if len(sequence) >= 2 and self.no_repeat_ngram_size == 3:
                prefix = tuple(sequence[-2:])
                banned = {
                    sequence[j + 2]
                    for j in range(len(sequence) - 2)
                    if tuple(sequence[j:j + 2]) == prefix
                }
                banned.discard(EOS_ID)
                if banned:
                    logits[0, list(banned)] = float("-inf")

            next_id = int(logits.argmax(dim=-1).item())
            if next_id == EOS_ID:
                break

            sequence.append(next_id)
            token = torch.tensor([next_id], dtype=torch.long)

        words = [
            self.id_to_token_en.get(idx, "<UNK>")
            for idx in sequence
            if idx not in {PAD_ID, SOS_ID, EOS_ID}
        ]
        return detokenize(words)
