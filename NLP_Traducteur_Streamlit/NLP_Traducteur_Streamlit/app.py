"""NLP Lab — application Streamlit de démonstration traduction FR → EN.
Lancer : streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import streamlit as st

from data_utils import (
    DISPLAY_MODELS, RESULTS, canonical_losses, canonical_metrics,
    canonical_predictions, default_predictions, load_optional_csv, read_csv,
)
from inference_adapter import LiveTranslator, missing_files
from experiments_data import (
    EXPERIMENTS, load_history, load_metrics, load_translations, bleu_comparison_df,
)

st.set_page_config(
    page_title="NLP Lab · Traduction FR → EN",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS global
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.block-container {max-width: 1200px; padding-top: 1.5rem; padding-bottom: 4rem;}

/* Hero */
.hero {
  border-radius: 20px; padding: 28px 32px; margin: 0 0 24px;
  background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1a0533 100%);
  border: 1px solid rgba(99,102,241,0.4);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.eyebrow {color: #a5b4fc; text-transform: uppercase; letter-spacing: .18em; font-size: .72rem; font-weight: 700;}
.hero h1 {font-size: clamp(1.8rem, 3.5vw, 2.6rem); margin: 8px 0 6px; font-weight: 800;
          letter-spacing: -.03em;
          background: linear-gradient(135deg, #e0e7ff, #c4b5fd, #a5f3fc);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero p {color: #c7d2fe; font-size: 1rem; line-height: 1.6; margin-bottom: 0;}
.hero-tags {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px;}
.tag {
  display: inline-block; padding: 4px 12px; border-radius: 100px;
  font-size: .75rem; font-weight: 600; letter-spacing: .04em;
  border: 1px solid;
}

/* Architecture card dans la sidebar */
.arch-card {
  border-radius: 12px; padding: 14px 16px; margin-bottom: 8px;
  background: rgba(30,27,75,0.6); border: 1px solid rgba(99,102,241,0.3);
}
.arch-title {font-weight: 700; font-size: .92rem; color: #e0e7ff; margin-bottom: 4px;}
.arch-desc {color: #94a3b8; font-size: .8rem; line-height: 1.5;}

/* Metric cards */
.metric-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0;}
.metric-card {
  border-radius: 14px; padding: 16px 18px;
  background: linear-gradient(135deg, #1e1b4b, #0f172a);
  border: 1px solid rgba(99,102,241,0.35);
  box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.metric-label {font-size: .72rem; text-transform: uppercase; letter-spacing: .12em;
               color: #94a3b8; font-weight: 600; margin-bottom: 6px;}
.metric-value {font-size: 1.6rem; font-weight: 800; color: #e0e7ff; line-height: 1.1;}
.metric-sub {font-size: .76rem; color: #64748b; margin-top: 4px;}

/* Section titles */
.sec-title {font-size: 1.25rem; font-weight: 700; color: #e0e7ff; margin: 16px 0 6px;}
.soft-copy {color: #94a3b8; font-size: .92rem; margin-bottom: 16px;}

/* Result card (translation output) */
.result-card {
  border: 1px solid rgba(99,102,241,0.4); border-radius: 14px;
  background: linear-gradient(135deg, rgba(30,27,75,0.8), rgba(15,23,42,0.8));
  padding: 18px 22px; margin: 14px 0;
}
.result-label {font-size: .72rem; color: #a5b4fc; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 8px;}
.result-value {font-size: 1.2rem; font-weight: 600; color: #f1f5f9; overflow-wrap: anywhere;}

/* Experiment selector pills */
.exp-badge {
  display: inline-block; padding: 3px 10px; border-radius: 100px;
  font-size: .72rem; font-weight: 700; margin-left: 8px;
}

/* Comparison bar */
.bleu-bar-wrap {display: flex; align-items: center; gap: 12px; margin: 6px 0;}
.bleu-bar-bg {flex: 1; height: 10px; border-radius: 100px; background: rgba(30,27,75,0.6); border: 1px solid rgba(99,102,241,0.2); overflow: hidden;}
.bleu-bar-fill {height: 100%; border-radius: 100px; transition: width .6s cubic-bezier(.4,0,.2,1);}
.bleu-label {font-size: .8rem; font-weight: 600; color: #e0e7ff; min-width: 42px; text-align: right;}

[data-testid="stTabs"] [role="tab"] {font-weight: 650; font-size: .95rem;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — sélecteur d'architecture
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Choisir une architecture")
    st.markdown('<p style="color:#94a3b8; font-size:.85rem; margin-bottom:12px;">Sélectionne une expérience pour explorer ses résultats dans l\'onglet <b>Expérience</b>.</p>', unsafe_allow_html=True)

    exp_labels = [e["label"] for e in EXPERIMENTS]
    exp_ids    = [e["id"]    for e in EXPERIMENTS]

    selected_label = st.radio(
        "Architecture",
        options=exp_labels,
        index=3,           # Exp finale par défaut
        label_visibility="collapsed",
    )
    selected_exp = EXPERIMENTS[exp_labels.index(selected_label)]

    st.divider()

    # Détails de l'architecture sélectionnée
    color = selected_exp["color"]
    st.markdown(f'<div style="width:100%; height:4px; border-radius:2px; background:{color}; margin-bottom:12px;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="arch-title" style="color:{color};">{selected_exp["short"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="arch-desc">{selected_exp["description"]}</div>', unsafe_allow_html=True)

    details = selected_exp.get("architecture_details", {})
    if details:
        st.markdown("**Détails :**")
        for k, v in details.items():
            st.markdown(f'<span style="color:#64748b; font-size:.8rem;">{k}</span> : <span style="color:#e0e7ff; font-size:.82rem; font-weight:600;">{v}</span><br>', unsafe_allow_html=True)

    st.divider()
    st.caption("NLP Lab · Projet académique FR → EN")

# ──────────────────────────────────────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<section class="hero">
  <div class="eyebrow">Projet académique · NLP · Français / Anglais</div>
  <h1>Du texte à la traduction.</h1>
  <p>Exploration comparative de 4 architectures : LSTM sans attention, LSTM + Bahdanau,<br>
     et le modèle final avec décodage no-repeat-trigram.</p>
  <div class="hero-tags">
    <span class="tag" style="color:#6175BB; border-color:#6175BB; background:rgba(97,117,187,0.1);">LSTM</span>
    <span class="tag" style="color:#A855F7; border-color:#A855F7; background:rgba(168,85,247,0.1);">Attention Bahdanau</span>
    <span class="tag" style="color:#F59E0B; border-color:#F59E0B; background:rgba(245,158,11,0.1);">Teacher Forcing</span>
    <span class="tag" style="color:#10B981; border-color:#10B981; background:rgba(16,185,129,0.1);">No-repeat Trigram ⭐</span>
    <span class="tag" style="color:#94a3b8; border-color:#334155; background:rgba(30,41,59,0.5);">Prototype de recherche</span>
  </div>
</section>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌐  Traduire en direct",
    "🔬  Expérience sélectionnée",
    "📊  Comparer toutes les architectures",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Traduction en direct
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-title">Traduction en direct</div>', unsafe_allow_html=True)
    st.markdown('<p class="soft-copy">Teste le modèle final (LSTM avec attention) sur une phrase de ton choix.</p>', unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Chargement du modèle LSTM avec attention (PyTorch)…")
    def load_pytorch_translator():
        try:
            from pytorch_inference import PyTorchLiveTranslator, find_model_path
            model_p = find_model_path()
            if model_p is not None:
                return PyTorchLiveTranslator(model_p)
        except Exception as exc:
            return exc
        return None

    pytorch_res = load_pytorch_translator()
    translator = None
    model_type = None

    if isinstance(pytorch_res, Exception):
        st.error(f"Fichier PyTorch détecté mais erreur au chargement : {pytorch_res}")
        with st.expander("Détails techniques de l'erreur"):
            st.exception(pytorch_res)
    elif pytorch_res is not None:
        translator = pytorch_res
        model_type = "pytorch"
        st.success("🟢 **Modèle PyTorch connecté avec succès** (`lstm_attention_final.pt`) · L'inférence neuronale en direct est active sur n'importe quel texte !")
    else:
        # Essai Keras si disponible
        missing = missing_files()
        if not missing:
            try:
                translator = LiveTranslator()
                model_type = "keras"
                st.success("🟢 **Modèle Keras chargé en direct**")
            except Exception as exc:
                st.error("Modèle Keras détecté mais erreur au chargement.")
        else:
            st.info(
                "ℹ️ **Mode consultation actif.** Le modèle PyTorch (`models/lstm_attention_final.pt`) est en cours d'initialisation. "
                "Vous pouvez traduire les phrases du jeu de test ci-dessous ou explorer les autres onglets."
            )

    # Exemples réels issus du jeu de test de l'Expérience Finale
    examples = [
        "qui va parler au vendeur de voitures ? big freddy.",
        "peter et moi étions dans un accident de voiture...",
        "je suis content de t'avoir vu à la télé.",
    ]
    st.markdown("**Essayer une phrase d'exemple :**")
    example_cols = st.columns(3)
    if "fr_input" not in st.session_state:
        st.session_state.fr_input = examples[0]
    for col, phrase in zip(example_cols, examples):
        col.button(phrase, key="example_" + str(examples.index(phrase)), use_container_width=True,
                   on_click=lambda p=phrase: st.session_state.update(fr_input=p))

    st.text_area(
        "Votre phrase en français", key="fr_input", height=110,
        placeholder="Écris ou colle n'importe quelle phrase en français (ex : je suis étudiante)…",
        max_chars=1000,
    )

    do_translate = st.button("🌐  Traduire en anglais", type="primary", use_container_width=True)

    if do_translate:
        text_query = st.session_state.fr_input.strip()
        if not text_query:
            st.warning("Saisis d'abord une phrase française.")
        elif translator is not None:
            # Mode inférence directe en temps réel (avec le modèle PyTorch)
            try:
                output = translator.translate(text_query)
                st.session_state.translation = output
                st.session_state.translation_ref = None
                st.session_state.translation_origin = "Modèle PyTorch en direct (lstm_attention_final.pt)" if model_type == "pytorch" else "Modèle en direct"
                st.session_state.translated_source = text_query
            except Exception as exc:
                st.error("Le modèle n'a pas pu générer de traduction.")
                with st.expander("Erreur technique"):
                    st.exception(exc)
                st.session_state.pop("translation", None)
        else:
            # Mode consultation des résultats réels d'expériences
            exp_df = load_translations(selected_exp, n=5000)
            match_row = None
            if exp_df is not None and not exp_df.empty:
                # 1. Correspondance exacte ou minuscule
                clean_q = text_query.lower()
                src_series = exp_df["source"].astype(str).str.lower()
                exact = exp_df[src_series == clean_q]
                if not exact.empty:
                    match_row = exact.iloc[0]
                else:
                    # 2. Correspondance par sous-chaîne
                    sub = exp_df[src_series.str.contains(re.escape(clean_q[:30]), na=False)]
                    if not sub.empty:
                        match_row = sub.iloc[0]

            if match_row is not None:
                st.session_state.translation = str(match_row.get("prediction", "—"))
                st.session_state.translation_ref = str(match_row.get("reference", "—"))
                st.session_state.translation_origin = f"Résultat réel de l'expérience : {selected_exp['short']}"
                st.session_state.translated_source = text_query
            else:
                st.session_state.pop("translation", None)
                st.warning(
                    f"⚠️ Phrase non trouvée dans les résultats de test précalculés de **{selected_exp['short']}**.\n\n"
                    "Comme les poids du modèle (`.keras` / `.pt`) ne sont pas dans `models/`, le traducteur utilise les **phrases réelles évaluées lors de vos entraînements**.\n\n"
                    "👉 Cliquez sur l'un des boutons d'exemples ci-dessus, ou rendez-vous dans l'onglet **🔬 Expérience sélectionnée** pour explorer les 1 000+ phrases de test !"
                )

    if "translation" in st.session_state and st.session_state.get("translated_source") == st.session_state.fr_input.strip():
        origin_label = st.session_state.get("translation_origin", "LSTM avec attention")
        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Traduction · {origin_label}</div>
          <div class="result-value">{st.session_state.translation}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("translation_ref"):
            st.caption(f"📖 **Référence humaine (vérité terrain) :** {st.session_state['translation_ref']}")

        if "<UNK>" in st.session_state.translation or "<unk>" in st.session_state.translation:
            st.info("`<UNK>` représente un mot absent du vocabulaire appris.")

    st.caption("Le système est un prototype académique, pas un traducteur professionnel.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Expérience sélectionnée
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    exp = selected_exp
    color = exp["color"]

    st.markdown(f'<div class="sec-title" style="color:{color};">{exp["label"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="soft-copy">{exp["description"]}</p>', unsafe_allow_html=True)

    # ── Métriques ──────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _load_metrics_cached(exp_id: str):
        e = next(x for x in EXPERIMENTS if x["id"] == exp_id)
        return load_metrics(e)

    @st.cache_data(show_spinner=False)
    def _load_history_cached(exp_id: str):
        e = next(x for x in EXPERIMENTS if x["id"] == exp_id)
        return load_history(e)

    @st.cache_data(show_spinner=False)
    def _load_translations_cached(exp_id: str):
        e = next(x for x in EXPERIMENTS if x["id"] == exp_id)
        return load_translations(e, n=2000)

    metrics = _load_metrics_cached(exp["id"])
    history = _load_history_cached(exp["id"])
    translations = _load_translations_cached(exp["id"])

    # KPIs
    bleu_val = metrics.get("_bleu_display") if metrics else None
    val_loss = (metrics.get("validation_loss") or metrics.get("val_loss")) if metrics else None
    best_ep  = metrics.get("best_epoch") if metrics else None
    n_params = metrics.get("num_parameters") if metrics else None
    train_min = metrics.get("training_minutes") if metrics else None

    kpi_items = [
        ("BLEU ↑", f"{bleu_val:.4f}" if bleu_val is not None else "—", "Score sur ensemble de test"),
        ("Val Loss ↓", f"{val_loss:.4f}" if val_loss is not None else "—", "Meilleure loss de validation"),
        ("Best Epoch", str(best_ep) if best_ep is not None else "—", "Epoch de la meilleure validation"),
        ("Paramètres", f"{int(n_params):,}".replace(",", " ") if n_params else "—", "Poids entraînables"),
        ("Durée entraîn.", f"{train_min:.1f} min" if train_min else "—", "Temps total d'entraînement"),
    ]

    cards_html = '<div class="metric-grid">'
    for label, val, sub in kpi_items:
        cards_html += f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color};">{val}</div>
          <div class="metric-sub">{sub}</div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Courbe d'entraînement ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Évolution de la loss")
    if history is not None and "epoch" in history.columns and "train_loss" in history.columns:
        chart_df = history.set_index("epoch")[["train_loss", "val_loss"]].dropna(how="all")
        chart_df.columns = ["Train Loss", "Val Loss"]
        st.line_chart(chart_df, height=260, use_container_width=True)
        st.caption("Évolution de la loss d'entraînement et de validation par epoch.")
    else:
        st.info("Historique d'entraînement non disponible pour cette expérience.")

    # ── Explorer les traductions ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💬 Explorer les traductions")
    if translations is not None and not translations.empty:
        st.caption(f"{len(translations)} phrases disponibles dans le CSV de test.")
        # Barre de recherche
        search = st.text_input("🔍 Rechercher dans les phrases sources", placeholder="Ex : voiture, merci, …", key=f"search_{exp['id']}")
        if search.strip():
            mask = translations["source"].str.contains(search.strip(), case=False, na=False)
            filtered = translations[mask]
        else:
            filtered = translations

        if filtered.empty:
            st.warning("Aucune phrase trouvée pour cette recherche.")
        else:
            choose = st.selectbox(
                "Choisir une phrase",
                options=list(range(min(len(filtered), 500))),
                format_func=lambda i: f"{i+1}. {str(filtered.iloc[i]['source'])[:100]}",
                key=f"sel_{exp['id']}",
            )
            row = filtered.iloc[choose]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🇫🇷 Source (Français)**")
                st.info(str(row.get("source", "—")))
            with c2:
                st.markdown("**📖 Référence (Anglais)**")
                st.success(str(row.get("reference", "—")))

            pred_val = str(row.get("prediction", "—"))
            st.markdown(f"**🤖 Traduction générée — {exp['short']}**")
            st.markdown(f"""
            <div class="result-card">
              <div class="result-value">{pred_val}</div>
            </div>
            """, unsafe_allow_html=True)

            if "<UNK>" in pred_val or "<unk>" in pred_val:
                st.caption("⚠️ `<UNK>` = token absent du vocabulaire appris.")
    else:
        st.info("CSV de traductions non disponible pour cette expérience.")

    # ── Détails architecture ───────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🔧 Détails de l'architecture"):
        details = exp.get("architecture_details", {})
        rows_detail = [{"Paramètre": k, "Valeur": str(v)} for k, v in details.items()]
        st.dataframe(pd.DataFrame(rows_detail), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Comparaison globale
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">Comparaison de toutes les architectures</div>', unsafe_allow_html=True)
    st.markdown('<p class="soft-copy">Vue d\'ensemble des métriques, courbes d\'entraînement et exemples de traductions côte à côte.</p>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def _bleu_df_cached():
        return bleu_comparison_df()

    bleu_df = _bleu_df_cached()

    # ── Tableau comparatif BLEU ────────────────────────────────────────────
    st.markdown("#### 🏆 Scores BLEU comparés")
    bleu_max = bleu_df["BLEU"].max(skipna=True)

    # Barres horizontales stylisées
    for _, row_b in bleu_df.iterrows():
        val = row_b["BLEU"]
        pct = (val / bleu_max * 100) if (bleu_max and not pd.isna(val)) else 0
        val_str = f"{val:.4f}" if not pd.isna(val) else "N/A"
        badge_color = row_b["color"]
        st.markdown(f"""
        <div style="margin: 10px 0;">
          <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
            <span style="font-size:.85rem; font-weight:600; color:#e0e7ff;">{row_b['Architecture']}</span>
            <span style="font-size:.85rem; font-weight:700; color:{badge_color};">{val_str}</span>
          </div>
          <div class="bleu-bar-bg">
            <div class="bleu-bar-fill" style="width:{pct:.1f}%; background:{badge_color};"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if not bleu_df["BLEU"].isna().all():
        best = bleu_df.loc[bleu_df["BLEU"].idxmax()]
        st.success(f"🏆 Meilleur BLEU : **{best['Architecture']}** avec **{best['BLEU']:.4f}**")

    # ── Tableau de métriques détaillées ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Tableau détaillé des métriques")

    @st.cache_data(show_spinner=False)
    def _all_metrics_cached():
        rows = []
        for exp_item in EXPERIMENTS:
            m = load_metrics(exp_item)
            bleu_val_m = m.get("_bleu_display") if m else None
            val_loss_m = (m.get("validation_loss") or m.get("val_loss") or m.get("test_loss")) if m else None
            best_ep_m = m.get("best_epoch") if m else None
            n_params_m = m.get("num_parameters") if m else None
            train_min_m = m.get("training_minutes") if m else None
            rows.append({
                "Architecture": exp_item["short"],
                "BLEU ↑": f"{bleu_val_m:.4f}" if bleu_val_m else "—",
                "Val Loss ↓": f"{val_loss_m:.4f}" if val_loss_m else "—",
                "Best Epoch": str(best_ep_m) if best_ep_m else "—",
                "Paramètres": f"{int(n_params_m):,}".replace(",", " ") if n_params_m else "—",
                "Durée (min)": f"{train_min_m:.1f}" if train_min_m else "—",
            })
        return pd.DataFrame(rows)

    metrics_table = _all_metrics_cached()
    st.dataframe(metrics_table, hide_index=True, use_container_width=True)

    # ── Courbes d'entraînement superposées ───────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Courbes de validation loss comparées")

    @st.cache_data(show_spinner=False)
    def _all_histories_cached():
        frames = []
        seen_paths = set()
        for exp_item in EXPERIMENTS:
            hp = exp_item.get("history_path")
            if hp is None or str(hp) in seen_paths:
                continue
            seen_paths.add(str(hp))
            h = load_history(exp_item)
            if h is not None and "epoch" in h.columns and "val_loss" in h.columns:
                h = h.copy()
                h["modele"] = exp_item["short"]
                frames.append(h[["epoch", "val_loss", "modele"]])
        return frames

    all_histories = _all_histories_cached()
    if all_histories:
        pivot = None
        for frame in all_histories:
            sub = frame.set_index("epoch")[["val_loss"]].rename(columns={"val_loss": frame["modele"].iloc[0]})
            pivot = sub if pivot is None else pivot.join(sub, how="outer")
        if pivot is not None:
            st.line_chart(pivot, height=300, use_container_width=True)
            st.caption("Validation loss par epoch. Les courbes avec le même historique partagent les mêmes points.")
    else:
        st.info("Historiques non disponibles.")

    # ── Comparaison phrase à phrase ────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💬 Comparer les traductions sur une même phrase")
    st.caption("Sélectionne une phrase source pour voir la traduction de chaque architecture côte à côte.")

    @st.cache_data(show_spinner=False)
    def _all_translations_cached():
        return {exp_item["id"]: load_translations(exp_item, n=500) for exp_item in EXPERIMENTS}

    all_trans = _all_translations_cached()

    first_df = next((df for df in all_trans.values() if df is not None and not df.empty), None)
    if first_df is not None:
        search_comp = st.text_input("🔍 Rechercher une phrase source", key="search_comp")
        if search_comp.strip():
            mask_c = first_df["source"].str.contains(search_comp.strip(), case=False, na=False)
            filtered_c = first_df[mask_c]
        else:
            filtered_c = first_df

        if filtered_c.empty:
            st.warning("Aucune phrase trouvée.")
        else:
            pick = st.selectbox(
                "Choisir une phrase",
                options=list(range(min(len(filtered_c), 200))),
                format_func=lambda i: f"{i+1}. {str(filtered_c.iloc[i]['source'])[:100]}",
                key="comp_pick",
            )
            src_text = str(filtered_c.iloc[pick]["source"])
            ref_text = str(filtered_c.iloc[pick].get("reference", "—"))

            st.markdown(f"**🇫🇷 Source :** {src_text}")
            st.markdown(f"**📖 Référence :** {ref_text}")
            st.markdown("")

            cols_comp = st.columns(2)
            for ci, exp_item in enumerate(EXPERIMENTS):
                df_trans = all_trans.get(exp_item["id"])
                if df_trans is None or df_trans.empty:
                    pred_text = "—"
                else:
                    match = df_trans[df_trans["source"] == src_text]
                    if match.empty:
                        idx = filtered_c.index[pick]
                        pred_text = str(df_trans.iloc[idx].get("prediction", "—")) if idx < len(df_trans) else "—"
                    else:
                        pred_text = str(match.iloc[0].get("prediction", "—"))

                with cols_comp[ci % 2]:
                    c_color = exp_item["color"]
                    st.markdown(f"""
                    <div style="border-left: 3px solid {c_color}; padding: 10px 14px; margin: 6px 0;
                                background: rgba(15,23,42,0.5); border-radius: 0 10px 10px 0;">
                      <div style="font-size:.75rem; font-weight:700; color:{c_color}; text-transform:uppercase;
                                  letter-spacing:.08em; margin-bottom:6px;">{exp_item['short']}</div>
                      <div style="color:#e0e7ff; font-size:.9rem; line-height:1.5;">{pred_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Aucun CSV de traductions disponible pour la comparaison côte à côte.")

st.divider()
st.caption("NLP Lab · Projet de traduction neuronale FR → EN · RNN / GRU / LSTM / Attention · Démonstration académique")

