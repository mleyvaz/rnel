"""RNEL demo: paste a claim and its evidence; see the RNEL tuple, the SL opinion and a suggested action.

    pip install -e .[nn] transformers streamlit
    streamlit run examples/app.py
"""
import pandas as pd
import streamlit as st

from rnel.decide import Policy
from rnel.text import NLIReader, rnel_from_text, sl_from_text

st.set_page_config(page_title="RNEL demo", layout="wide")
st.title("Refined Neutrosophic Evidential Logic: demo")
st.caption("Each piece of evidence is read by a natural-language-inference model (no training). "
           "RNEL keeps the contradiction between pieces that Subjective Logic fusion erases (Definition 8.4, "
           "Proposition 8.3 of Smarandache and Leyva-Vázquez).")


@st.cache_resource
def reader():
    return NLIReader()


EXAMPLES = {
    "Sources disagree": ("Coffee consumption increases the risk of heart disease.",
                         "A 2021 cohort of 500,000 adults found higher coffee intake linked to more heart disease.\n"
                         "A large meta-analysis found that moderate coffee drinking lowers the risk of heart disease."),
    "Sources agree": ("The Great Wall of China is in Asia.",
                      "The Great Wall stretches across northern China.\nChina is a country in East Asia."),
    "No relevant evidence": ("The new vaccine prevents 90% of infections.",
                             "The clinical trial is still recruiting participants.\nResults are expected next year."),
}
choice = st.selectbox("Example", ["(write your own)"] + list(EXAMPLES))
claim0, ev0 = EXAMPLES.get(choice, ("", ""))
claim = st.text_input("Claim", claim0)
evidence = [e.strip() for e in st.text_area("Evidence, one piece per line", ev0, height=150).splitlines() if e.strip()]
weight = st.slider("Evidence units per piece", 1.0, 20.0, 5.0)
st.info("NLI models often label unrelated text as 'contradiction'. Give only evidence that is about the claim.")

if claim and evidence:
    t, readings = rnel_from_text(reader(), claim, evidence, weight=weight)
    sl = sl_from_text(readings, weight=weight)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("RNEL tuple")
        d = t.as_dict()
        st.bar_chart(pd.DataFrame({"value": [d[k] for k in ("T", "F", "C", "U", "N", "G")]},
                                  index=["T true", "F false", "C contradiction", "U undetermined",
                                         "N neither", "G ignorance"]))
        comp, action = Policy().decide(t)
        st.success(f"Suggested action ({comp}): {action}")
    with c2:
        st.subheader("Subjective Logic (cumulative fusion)")
        st.write(f"belief {sl.b:.3f}, disbelief {sl.d:.3f}, uncertainty {sl.u:.3f}, "
                 f"projected probability {sl.projected:.3f}")
        st.caption("SL gives one uncertainty mass and cannot tell disagreement from balanced evidence.")
    st.subheader("Reading of each piece")
    st.dataframe(pd.DataFrame([{"evidence": r.evidence, "entailment": round(r.entail, 3),
                                "contradiction": round(r.contradict, 3), "neutral": round(r.neutral, 3)}
                               for r in readings]), use_container_width=True)
