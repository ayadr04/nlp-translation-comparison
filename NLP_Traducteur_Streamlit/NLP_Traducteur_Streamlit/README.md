# NLP Lab — comparateur de traduction FR → EN

Application Streamlit pour présenter un projet académique de traduction neuronale : RNN, GRU, LSTM sans attention et LSTM avec attention.

**Deux onglets** :

1. **Traduire une phrase** : traduction en direct à partir du **LSTM avec attention** quand les fichiers du modèle sont disponibles. Trois phrases d'essai ; avertissement sur `<UNK>` et sur les limites du modèle.
2. **Comprendre les résultats** : scores BLEU/chrF/TER, exploration des phrases de test, graphique des pertes et analyse de la longueur.

> Les métriques et prédictions ne sont **pas simulées**. Deux lignes historiques de l'expérience 2 et les métriques par longueur de l'expérience 4 sont présentes uniquement pour rendre l'interface explicable avant d'ajouter les fichiers complets.

## 1. Lancer en local (Windows / macOS / Linux)

Ouvrir le dossier dans VS Code. Dans le terminal :

```bash
python -m venv .venv
```

Activer l'environnement :

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Puis installer les bibliothèques et démarrer :

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ouvrir l'URL locale indiquée par Streamlit (généralement `http://localhost:8501`).

**Sans modèle**, l'onglet 2 fonctionne déjà ; la traduction directe affiche explicitement qu'elle n'est pas disponible. C'est voulu : l'application ne doit pas inventer de traductions.

## 2. Ajouter ses CSV (le plus facile)

### A. Prédictions du test

Copier un fichier `results/predictions_comparees.csv` de forme :

```csv
source,reference,rnn,gru,lstm_sans_attention,lstm_avec_attention
Bonjour,Hello,,,,
```

Remplacer les cellules vides par **les vraies prédictions** de vos notebooks. Les noms `source`, `reference` sont obligatoires ; les autres sont facultatifs.

**Autre option** (sans fusionner) : déposer les deux CSV déjà produits par les notebooks, si les noms sont exactement :

```
results/experience_2/predictions_test.csv
results/experience_5/predictions_test.csv
```

Chacun doit comporter `source`, `reference`, `prediction` et éventuellement `index`, comme dans l'exemple historique du notebook. L'application fusionne les prédictions d'après la paire source/référence (à condition que les phrases soient réellement les mêmes). Si une seule expérience est présente, l'autre colonne est vide.

Tu peux aussi **charger depuis l'onglet 2** un CSV comparatif directement, sans le copier dans le projet.

### B. Tableau des scores

Compléter `results/metrics.csv` avec des scores calculés **sur les mêmes données de test** :

```csv
modele,bleu,chrf,ter
RNN,,,
GRU,,,
LSTM sans attention,,,
LSTM avec attention,,,
```

**Ne remplace pas les cellules vides par des chiffres inventés.** Les scores doivent venir de tes résultats d'évaluation.

### C. Courbes d'entraînement

Compléter `results/losses.csv` avec les valeurs réelles pour chaque époque :

```csv
modele,epoch,train_loss,val_loss
LSTM avec attention,1,,
LSTM avec attention,2,,
```

Les lignes vides ne seront pas tracées. N'ajoute la courbe que lorsque le CSV contient des valeurs. Dans l'interface, tu peux aussi charger directement un CSV de pertes.

### D. Analyse des longueurs (déjà disponible)

`results/bleu_par_longueur.csv` contient les chiffres de l'expérience 4 fournis précédemment :

| Longueur | BLEU | chrF | TER |
|---|---:|---:|---:|
| Courte | 2,93 | 9,10 | 97,79 |
| Moyenne | 1,42 | 8,24 | 96,39 |
| Longue | 0,85 | 8,05 | 101,53 |

## 3. Activer le vrai modèle LSTM avec attention

Le modèle **n'était pas présent dans les fichiers remis avec la demande** ; on ne peut donc pas encore garantir une prédiction directe. La partie Streamlit est prête, et un connecteur pour un **cas courant** est inclus.

### Option Keras compatible avec le connecteur

Le connecteur `inference_adapter.py` suppose que :

- l'encodeur reçoit une séquence d'indices de tokens français ;
- le décodeur reçoit une séquence d'indices de tokens anglais commençant par un token de début ;
- `model.predict([encoder_input, decoder_input])` renvoie les distributions de mots pour chaque position (`batch, longueur, vocabulaire`) ;
- les deux tokenizers sont des `tf.keras.preprocessing.text.Tokenizer` ;
- la normalisation, le padding et les tokens spéciaux correspondent **exactement** au notebook d'entraînement.

Dans Colab, après avoir entraîné et récupéré ton vrai modèle, importe la fonction d'export de `tools/export_from_colab.py`, puis exécute (adapte les noms des variables aux tiens) :

```python
export_keras_model(
    model=modele_final,
    tokenizer_fr=tokenizer_fr,
    tokenizer_en=tokenizer_en,
    out_dir="/content/export_app",
    max_source_length=30,  # REMPLACER par la valeur du notebook
    max_target_length=30,  # REMPLACER par la valeur du notebook
    target_start_token="<start>",  # REMPLACER si différent
    target_end_token="<end>",      # REMPLACER si différent
    normalization="none",          # REMPLACER si différent
)
```

Télécharge le contenu du dossier `export_app` et mets dans `models/` :

```
models/
    lstm_attention.keras
    tokenizer_fr.json
    tokenizer_en.json
    inference_config.json
```

Ensuite :

```bash
pip install -r requirements-modele.txt
streamlit run app.py
```

Si ton notebook utilise une classe/couche Keras personnalisée, renseigne `CUSTOM_OBJECTS` dans `model_custom_objects.py` ; il faut y reprendre **la même définition** que pendant l'entraînement.

**Attention à l'architecture !** Certains notebooks LSTM avec attention construisent des modèles encodeur/décodeur séparés, utilisent un décodeur auto-régressif Python personnalisé, des vocabulaires `dict`, ou n'enregistrent que des poids `.weights.h5`. Dans ces cas, exporter le `.keras` seul ne suffit pas, et la fonction `translate` de `inference_adapter.py` doit être reliée à la vraie fonction de décodage du notebook. Les fichiers mentionnés ci-dessus ne sont pas interchangeables avec un checkpoint PyTorch ni avec des poids seuls. **Il faut alors fournir `05_lstm_avec_attention_tf100.ipynb` et les sauvegardes du modèle pour réaliser l'adaptation exacte.**

Avant la présentation, vérifie au moins 2 phrases test : les résultats Streamlit doivent être identiques aux sorties du décodage de ton notebook (mêmes règles de normalisation et longueur maximale).

## 4. Structure du projet

```text
NLP_Traducteur_Streamlit/
├── app.py                       # Interface des deux onglets
├── data_utils.py                # Import/normalisation CSV
├── inference_adapter.py         # Connexion réelle Keras (architecture compatible)
├── model_custom_objects.py      # Couches personnalisées éventuelles
├── tools/export_from_colab.py   # Export depuis le notebook
├── models/inference_config.json # Paramètres d'inférence (exemple, à remplacer)
├── results/
│   ├── bleu_par_longueur.csv     # Vraies valeurs de l'expérience 4
│   ├── exemples_historiques.csv  # 2 exemples de l'expérience 2
│   ├── metrics.csv              # À compléter
│   ├── losses.csv               # À compléter
│   └── predictions_comparees_modele.csv  # Gabarit CSV à renommer
├── tests/                       # Tests unitaires
├── .streamlit/config.toml
├── requirements.txt
├── requirements-modele.txt
└── README.md
```

## 5. Vérifier rapidement le projet

```bash
pip install pytest
pytest -q
```

### Ce qu'on peut dire à l'oral

« Notre application comporte deux parties : une interface de traduction pour le LSTM avec attention et un tableau d'analyse des expériences. Nous étudions les différences entre architectures, les erreurs de répétition et les mots inconnus. Les résultats de l'expérience 4 montrent déjà une diminution du BLEU pour les phrases longues. Le bénéfice de l'attention doit être confirmé par les scores comparables de l'expérience 5. »

## 6. Publication GitHub

Inclure les fichiers Python, `requirements.txt`, ce README et les CSV de résultats légers. Conserver les poids volumineux hors du dépôt ou utiliser Git LFS si nécessaire. Ne pas publier de données privées ni de fichiers dont la licence interdit la redistribution. Préparer un CSV `predictions_comparees.csv` avec des exemples vraiment évalués avant la démonstration.
