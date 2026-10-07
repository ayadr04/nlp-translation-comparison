
"""Préparation commune : importable sans téléchargement ni entraînement."""
import json
import re
import unicodedata
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3
TOKEN_PATTERN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+|[^\w\s]", flags=re.UNICODE)

def clean_text(text, lowercase=True):
    if not isinstance(text, str):
        raise TypeError("Une phrase doit être une chaîne de caractères.")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'")
    text = " ".join(text.split())
    return text.lower() if lowercase else text

def tokenize(text, lowercase=True):
    return TOKEN_PATTERN.findall(clean_text(text, lowercase=lowercase))

def prepare_pair(fr, en, lowercase=True):
    fr_clean = clean_text(fr, lowercase)
    en_clean = clean_text(en, lowercase)
    return {
        "fr_original": fr,
        "en_original": en,
        "fr": fr_clean,
        "en": en_clean,
        "src_tokens": tokenize(fr_clean, lowercase),
        "tgt_tokens": tokenize(en_clean, lowercase),
    }

def pair_key(row):
    return (tuple(row["src_tokens"]), tuple(row["tgt_tokens"]))

def encode_row(row, vocab_fr, vocab_en):
    item = dict(row)
    item["src_ids"] = [vocab_fr.get(t, UNK_ID) for t in row["src_tokens"]] + [EOS_ID]
    item["tgt_ids"] = [SOS_ID] + [vocab_en.get(t, UNK_ID) for t in row["tgt_tokens"]] + [EOS_ID]
    item["src_token_count"] = len(row["src_tokens"])
    item["src_length"] = len(item["src_ids"])
    return item

def decode_ids(ids, vocab, remove_special=True):
    reverse = {idx: token for token, idx in vocab.items()}
    skipped = {PAD_ID, SOS_ID, EOS_ID} if remove_special else set()
    return [reverse.get(int(idx), "<UNK>") for idx in ids if int(idx) not in skipped]

class TranslationDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return self.rows[index]

def collate_translation(rows):
    src_sequences = [torch.tensor(r["src_ids"], dtype=torch.long) for r in rows]
    input_sequences = [torch.tensor(r["tgt_ids"][:-1], dtype=torch.long) for r in rows]
    target_sequences = [torch.tensor(r["tgt_ids"][1:], dtype=torch.long) for r in rows]
    src = pad_sequence(src_sequences, batch_first=True, padding_value=PAD_ID)
    decoder_input = pad_sequence(input_sequences, batch_first=True, padding_value=PAD_ID)
    target = pad_sequence(target_sequences, batch_first=True, padding_value=PAD_ID)
    return {
        "src": src,
        "src_lengths": torch.tensor([len(s) for s in src_sequences], dtype=torch.long),
        "src_token_counts": torch.tensor([r["src_token_count"] for r in rows], dtype=torch.long),
        "src_mask": src.ne(PAD_ID),
        "decoder_input": decoder_input,
        "target": target,
        "target_mask": target.ne(PAD_ID),
        "source_texts": [r["fr"] for r in rows],
        "reference_texts": [r["en"] for r in rows],
    }

def make_loaders(prepared, config):
    generator = torch.Generator().manual_seed(config["seed"])
    return {
        name: DataLoader(TranslationDataset(rows), batch_size=config["batch_size"],
                         shuffle=(name == "train"), collate_fn=collate_translation,
                         num_workers=0, generator=generator if name == "train" else None)
        for name, rows in prepared.items()
    }

def load_prepared(project_dir):
    project_dir = Path(project_dir)
    data_dir = project_dir / "data" / "prepared"
    def load_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    config = load_json(project_dir / "config.json")
    vocab_fr = load_json(data_dir / "vocab_fr.json")
    vocab_en = load_json(data_dir / "vocab_en.json")
    prepared = {}
    for name in ["train", "validation", "test"]:
        with (data_dir / f"{name}.jsonl").open(encoding="utf-8") as f:
            prepared[name] = [json.loads(line) for line in f if line.strip()]
    return config, vocab_fr, vocab_en, prepared

def encode_source(text, vocab_fr, lowercase=True):
    tokens = tokenize(text, lowercase)
    return [vocab_fr.get(token, UNK_ID) for token in tokens] + [EOS_ID]
