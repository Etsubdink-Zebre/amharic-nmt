"""Streamlit web app for English → Amharic translation.

Run locally:  streamlit run streamlit_app.py
Deployed on Streamlit Community Cloud from this repository.
"""
import html

import streamlit as st
import torch

from src.translator import SCOPE_NOTE, Translator

MODELS = {
    "Attention-LSTM": "bahdanau",
    "Seq2Seq-LSTM (baseline)": "seq2seq",
}
EXAMPLES = ["She is Ethiopian and was born there.", "How much does this book cost?",
            "The farmers are waiting for the rain.", "Where is the hospital?",
            "Our school has many students and teachers.", "Thank you very much."]

st.set_page_config(page_title="English → Amharic Translator", page_icon="🌍", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def get_translator(name):
    # CPU is plenty for single sentences and is what the cloud host provides.
    torch.set_num_threads(2)
    return Translator(name, torch.device("cpu"))


def heatmap_html(r):
    """Attention matrix as an HTML table, so the browser renders the Ethiopic script."""
    clean = lambda t: html.escape(t.replace("▁", "") or "␣")
    head = "".join(f"<th style='writing-mode:vertical-rl;transform:rotate(180deg);font-weight:400;"
                   f"padding:2px;height:70px;text-align:left'>{clean(t)}</th>" for t in r["src_tokens"])
    rows = ""
    for tok, weights in zip(r["tgt_tokens"], r["attention"]):
        cells = "".join(f"<td title='{w:.2f}' style='width:26px;height:24px;border:1px solid rgba(128,128,128,.25);"
                        f"background:rgba(31,111,235,{w:.3f})'></td>" for w in weights)
        rows += f"<tr><th style='text-align:right;font-weight:400;padding-right:6px'>{clean(tok)}</th>{cells}</tr>"
    return f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:14px'><tr><th></th>{head}</tr>{rows}</table></div>"


st.title("English → Amharic Translator")
st.caption("LSTM encoder–decoder models trained from scratch on 692k English–Amharic sentence pairs.")
st.info("**What this translator handles.** " + SCOPE_NOTE, icon="ℹ️")

if "text" not in st.session_state:
    st.session_state.text = EXAMPLES[0]

st.write("Try an example:")
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, use_container_width=True):
        st.session_state.text = ex

text = st.text_area("English sentence", key="text", max_chars=500, height=100)
c1, c2 = st.columns([3, 1])
model_label = c1.selectbox("Model", list(MODELS))
beam = c2.selectbox("Beam size", [1, 3, 5, 8], index=2)

if st.button("Translate", type="primary") or text:
    if text.strip():
        r = get_translator(MODELS[model_label]).translate(text.strip(), beam=beam)
        st.markdown(f"<p style='font-size:2rem;line-height:1.5;margin:.5rem 0'>{html.escape(r['translation'])}</p>",
                    unsafe_allow_html=True)
        st.caption(f"{model_label} · beam {beam} · {r['latency_ms']} ms")
        for w in r["warnings"]:
            st.warning(w, icon="⚠️")
        if "attention" in r:
            with st.expander("Attention heatmap (rows = Amharic output, columns = English input)", expanded=True):
                st.markdown(heatmap_html(r), unsafe_allow_html=True)
        else:
            st.info("The baseline Seq2Seq model has no attention, so there is no heatmap.")
