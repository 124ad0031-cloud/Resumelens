"""
app.py  -  Streamlit interface of ResumeLens (AI resume screening assistant).
Run with:  streamlit run app.py
All NLP work is done by matcher.py.
"""

import html
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

import matcher
from samples import SAMPLE_JD, SAMPLE_RESUMES

st.set_page_config(page_title="ResumeLens", page_icon="📄", layout="wide")

st.markdown("""
<style>
.hero {background: linear-gradient(120deg,#1B2A41 0%,#0F8B8D 100%); color:white;
       padding: 1.3rem 1.8rem; border-radius: 14px; margin-bottom: 1rem;}
.hero h1 {margin:0; font-size: 2rem; color:white;}
.hero p  {margin:.3rem 0 0 0; opacity:.92;}
.card {padding:.6rem .85rem; border-radius:8px; margin:.35rem 0; color:#1B2A41; line-height:1.45;}
.Strong {background:#D8F0E1; border-left:6px solid #2E8B57;}
.Partial {background:#FBF0CF; border-left:6px solid #E0A800;}
.Missing {background:#F8DADF; border-left:6px solid #D1495B;}
.small {font-size:.85rem; opacity:.85;}
.chip {display:inline-block; padding:2px 10px; border-radius:12px; margin:2px; font-size:.85rem; color:#1B2A41;}
.chip.ok {background:#D8F0E1;} .chip.no {background:#F8DADF;} .chip.extra {background:#E3E7EC;}
</style>
<div class="hero">
  <h1>ResumeLens</h1>
  <p>AI resume screening assistant: ranks candidates by what they have actually done, not by the exact
     words they used. Powered by a pretrained sentence Transformer that runs on your laptop.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Loading the Transformer (first run downloads about 90 MB)...")
def get_model():
    return matcher.load_model()

model, model_error = get_model()


@st.cache_data(show_spinner="Analysing candidates...")
def run_analysis(jd_text, resume_items, strong, partial, weight, blind, _model):
    """Analyse every resume against the job description. resume_items = tuple of (name, text)."""
    requirements = matcher.extract_requirements(jd_text)
    jd_skills = matcher.find_skills(jd_text)
    results = {}
    for name, text in resume_items:
        text = matcher.anonymize(text) if blind else text
        outcome = matcher.analyze_candidate(requirements, jd_skills, text, _model, strong, partial, weight)
        if outcome is not None:
            results[name] = outcome
    return requirements, sorted(jd_skills), results


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Engine")
    if model is not None:
        st.success("Transformer active\n\nall-MiniLM-L6-v2 (6 layers, 384-dim embeddings)")
    else:
        st.error("Transformer could not be loaded: using the keyword fallback (TF-IDF). "
                 "Connect to the internet once to download the model.")
        with st.expander("Error details"):
            st.code(model_error)

    st.header("Data")
    source = st.radio("Candidates", ["Demo data", "Upload my own"], label_visibility="collapsed")
    blind = st.toggle("Blind screening", value=False,
                      help="Masks names, e-mails, phone numbers and links before scoring to reduce bias.")

    st.header("Scoring")
    weight = st.slider("Weight of requirement matching", 0.0, 1.0, 0.6, 0.05,
                       help="Final score = weight x requirement coverage + (1 - weight) x skill-keyword coverage")
    default_strong, default_partial = (0.50, 0.35) if model is not None else (0.30, 0.15)
    strong = st.slider("Strong-match threshold", 0.10, 0.90, default_strong, 0.01, key=f"s{model is not None}")
    partial = st.slider("Partial-match threshold", 0.05, 0.80, default_partial, 0.01, key=f"p{model is not None}")
    if partial >= strong:
        st.warning("Partial threshold should be lower than the strong threshold.")

# ------------------------------------------------------------------ inputs
left, right = st.columns([1, 1])
with left:
    st.subheader("1. Job description")
    jd_file = st.file_uploader("Upload job description (.txt/.pdf), or edit the text below",
                               type=["txt", "pdf"], key="jd_file")
    if jd_file is not None:
        try:
            st.session_state["jd"] = matcher.read_uploaded_file(jd_file)
        except Exception as error:
            st.warning(f"Could not read that file: {error}")
    if "jd" not in st.session_state:
        st.session_state["jd"] = SAMPLE_JD
    jd_text = st.text_area("Job description", key="jd", height=300, label_visibility="collapsed")

with right:
    st.subheader("2. Candidate resumes")
    resumes = {}
    if source == "Demo data":
        resumes = dict(SAMPLE_RESUMES)
        st.caption("4 fictional candidates with deliberately different backgrounds and wording.")
        for name, text in resumes.items():
            with st.expander(name):
                st.text(text)
    else:
        files = st.file_uploader("Upload resumes (.txt or .pdf)", type=["txt", "pdf"], accept_multiple_files=True)
        for uploaded in files or []:
            try:
                text = matcher.read_uploaded_file(uploaded)
                if text.strip():
                    resumes[uploaded.name.rsplit(".", 1)[0]] = text
                else:
                    st.warning(f"{uploaded.name}: no text found (scanned PDFs are not supported).")
            except Exception as error:
                st.warning(f"Could not read {uploaded.name}: {error}")

requirements = matcher.extract_requirements(jd_text)
if len(requirements) == 0:
    st.info("Add a job description with at least one requirement line (3 or more words).")
    st.stop()
if len(resumes) == 0:
    st.info("Add at least one resume.")
    st.stop()

requirements, jd_skills, results = run_analysis(jd_text, tuple(resumes.items()), strong, partial,
                                                weight, blind, model)
if not results:
    st.warning("No resume contained analysable text.")
    st.stop()

# candidate labels (anonymous in blind mode)
names = list(results.keys())
label = {name: (f"Candidate {chr(65 + i)}" if blind else name) for i, name in enumerate(names)}
ranked = sorted(names, key=lambda n: results[n]["score"], reverse=True)

st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["Ranking dashboard", "Candidate deep-dive",
                                  "Transformer vs keywords", "How it works"])

# ------------------------------------------------------------------ TAB 1: ranking
with tab1:
    best = ranked[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates screened", len(results))
    c2.metric("Requirements extracted", len(requirements))
    c3.metric("Top candidate", label[best], f"{results[best]['score']}")
    c4.metric("Matches only the Transformer found", sum(r["transformer_only"] for r in results.values()),
              help="Requirements matched by meaning although the resume sentence shares almost no words with them.")
    if blind:
        st.info("Blind screening is ON: names and contact details were masked before scoring.")

    table = []
    for rank, name in enumerate(ranked, start=1):
        r = results[name]
        counts = pd.Series([row["status"] for row in r["evidence"]]).value_counts()
        table.append({
            "Rank": rank, "Candidate": label[name], "Score (0-100)": r["score"], "Verdict": matcher.verdict(r["score"]),
            "Strong": int(counts.get("Strong", 0)), "Partial": int(counts.get("Partial", 0)),
            "Missing": int(counts.get("Missing", 0)),
            "Skills matched": f"{len(r['matched'])}/{len(jd_skills)}",
        })
    table_df = pd.DataFrame(table)

    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.subheader("Ranking")
        st.bar_chart(table_df.set_index("Candidate")["Score (0-100)"])
    with col_b:
        st.subheader("Requirement coverage matrix")
        matrix = np.array([[row["semantic"] for row in results[name]["evidence"]] for name in ranked])
        fig = Figure(figsize=(7, 0.6 * len(ranked) + 1.6))
        ax = fig.subplots()
        image = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=max(0.8, float(matrix.max())), aspect="auto")
        ax.set_xticks(range(len(requirements)))
        ax.set_xticklabels([f"R{i + 1}" for i in range(len(requirements))])
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels([label[n] for n in ranked])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.03)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Green = the resume contains a sentence similar in meaning to that requirement.")

    st.dataframe(table_df, hide_index=True, width="stretch")
    st.download_button("Download ranking (.csv)", table_df.to_csv(index=False), "resumelens_ranking.csv")
    with st.expander("Requirements extracted from the job description (R1, R2, ...)"):
        for i, requirement in enumerate(requirements):
            st.write(f"**R{i + 1}.** {requirement}")
        st.write("**Skills detected in the job description:** " + (", ".join(jd_skills) or "none"))

# ------------------------------------------------------------------ TAB 2: deep-dive
with tab2:
    chosen = st.selectbox("Candidate", ranked, format_func=lambda n: label[n])
    r = results[chosen]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Match score", r["score"], matcher.verdict(r["score"]), delta_color="off")
    m2.metric("Requirement coverage", f"{r['requirement_score']}%")
    m3.metric("Skill coverage", "n/a" if r["skill_score"] is None else f"{r['skill_score']}%")
    m4.metric("Transformer-only matches", r["transformer_only"])

    st.subheader("Evidence for each requirement")
    for i, row in enumerate(r["evidence"]):
        evidence_text = html.escape(row["evidence"]) if row["status"] != "Missing" else \
            "<i>No convincing evidence found. Closest sentence: </i>" + html.escape(row["evidence"])
        st.markdown(
            f'<div class="card {row["status"]}"><b>R{i + 1} - {row["status"]}</b> '
            f'<span class="small">(meaning similarity {row["semantic"]:.2f})</span><br>'
            f'<b>Requires:</b> {html.escape(row["requirement"])}<br><b>Resume says:</b> {evidence_text}</div>',
            unsafe_allow_html=True)

    st.subheader("Skills")
    chips = "".join(f'<span class="chip ok">{html.escape(s)}</span>' for s in r["matched"])
    chips += "".join(f'<span class="chip no">missing: {html.escape(s)}</span>' for s in r["missing"])
    st.markdown(chips or "No skills detected in the job description.", unsafe_allow_html=True)
    if r["extra"]:
        st.markdown("Additional skills: " + "".join(f'<span class="chip extra">{html.escape(s)}</span>'
                                                    for s in r["extra"]), unsafe_allow_html=True)
    st.caption("The skill list is a simple dictionary, so a resume that describes a skill in other words "
               "may show it as missing. The requirement evidence above is the semantic check.")

    st.subheader("Recruiter assistant: suggested interview questions")
    if r["questions"]:
        for question in r["questions"]:
            st.write("- " + question)
    else:
        st.write("No gaps found: focus the interview on depth and past impact.")

# ------------------------------------------------------------------ TAB 3: Transformer vs keywords
with tab3:
    st.subheader("Why a Transformer? Meaning vs matching words")
    st.caption("The keyword method (TF-IDF) only counts shared words. The Transformer compares meaning.")
    s1, s2 = st.columns(2)
    sentence_a = s1.text_input("Sentence A", "Created web services that other teams can call.")
    sentence_b = s2.text_input("Sentence B", "Build REST APIs for other teams.")
    if sentence_a.strip() and sentence_b.strip():
        meaning = float(matcher.semantic_similarity([sentence_a], [sentence_b], model)[0, 0])
        words = float(matcher.keyword_similarity([sentence_a], [sentence_b])[0, 0])
        d1, d2 = st.columns(2)
        d1.metric("Keyword similarity (TF-IDF)", f"{words:.2f}")
        d2.metric("Meaning similarity (Transformer)" if model is not None else "Meaning similarity (fallback = keywords)",
                  f"{meaning:.2f}")

    st.subheader("Candidate comparison")
    compare = pd.DataFrame({
        "Candidate": [label[n] for n in names],
        "Avg best-match similarity: keywords": [results[n]["avg_keyword"] for n in names],
        "Avg best-match similarity: Transformer": [results[n]["avg_semantic"] for n in names],
    })
    compare["Rank by keywords"] = compare["Avg best-match similarity: keywords"].rank(ascending=False, method="min").astype(int)
    compare["Rank by Transformer"] = compare["Avg best-match similarity: Transformer"].rank(ascending=False, method="min").astype(int)
    st.dataframe(compare, hide_index=True, width="stretch")
    st.bar_chart(compare.set_index("Candidate")[["Avg best-match similarity: keywords",
                                                 "Avg best-match similarity: Transformer"]])

    st.subheader("Requirement-level view")
    who = st.selectbox("Candidate ", ranked, format_func=lambda n: label[n], key="who")
    detail = pd.DataFrame([{"Requirement": f"R{i + 1}", "Keyword sim": row["keyword"],
                            "Transformer sim": row["semantic"], "Status": row["status"],
                            "Found only by Transformer": row["status"] != "Missing" and row["keyword"] < 0.10}
                           for i, row in enumerate(results[who]["evidence"])])
    st.dataframe(detail, hide_index=True, width="stretch")

# ------------------------------------------------------------------ TAB 4: how it works
with tab4:
    st.markdown(f"""
### Problem
A recruiter may receive hundreds of resumes for one role. Keyword filters reject strong candidates who describe
the same skill in different words ("built web services" instead of "REST APIs") and reward resumes that simply
copy the job description.

### Method
1. **Parse**: the job description is split into *requirements*; each resume into *sentences*.
2. **Embed**: a pretrained sentence Transformer (`{matcher.MODEL_NAME}`) converts every sentence into a
   384-number vector; sentences with similar meaning get similar vectors.
3. **Match**: for each requirement the most similar resume sentence (cosine similarity) is its *evidence*.
   Similarity at or above the strong threshold = **Strong**, at or above the partial threshold = **Partial**, otherwise **Missing**.
4. **Score**: `score = w x requirement coverage + (1 - w) x skill coverage`; Strong counts 1, Partial 0.5, Missing 0.
   Skill coverage is the share of job-description skills (dictionary lookup) found in the resume.
5. **Explain**: the app shows the evidence sentence, skill gaps and interview questions, so every score can be checked by a human.
6. **Blind mode**: masks names and contact details before scoring.

### Limitations
- The thresholds are heuristics and can be adjusted in the sidebar; they are not learned from labelled hiring data.
- The skill dictionary is small. The model reads English only and truncates very long sentences.
- Scores support, and must not replace, human decisions. Blind mode is a simple rule-based mask and does not remove every identifying detail.
- Scanned (image-only) PDFs are not supported.
""")
