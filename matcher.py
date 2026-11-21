"""
matcher.py  -  all NLP logic of ResumeLens (no UI code).

Pipeline for one resume:
  1. Split the job description into REQUIREMENTS and the resume into SENTENCES.
  2. Turn every requirement and every resume sentence into a vector with a pretrained
     sentence Transformer (all-MiniLM-L6-v2).
  3. For each requirement find the resume sentence with the highest cosine similarity (the "evidence").
  4. Combine requirement coverage with a skill-keyword check into one match score.
  5. Compare with a plain keyword (TF-IDF) baseline to show what the Transformer adds.
"""

import io
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skills import SKILLS

MODEL_NAME = os.environ.get("RESUMELENS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Loading the Transformer
# ---------------------------------------------------------------------------
def load_model():
    """Load the sentence Transformer. Returns (model, error_message). model is None if loading failed."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME), ""
    except Exception as error:                           # no internet / model missing
        return None, str(error)


# ---------------------------------------------------------------------------
# Reading and splitting text
# ---------------------------------------------------------------------------
def read_uploaded_file(uploaded):
    """Return the text of an uploaded .txt or .pdf file (Streamlit UploadedFile)."""
    data = uploaded.read()
    if uploaded.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def clean_line(line):
    """Remove bullet symbols and extra spaces."""
    return re.sub(r"^[\s\-\*\u2022\u25cf\u25aa\d\.\)]+", "", line).strip()


def split_units(text):
    """Split text into lines and then sentences; keep units with at least 3 words."""
    units = []
    for line in text.splitlines():
        for part in re.split(r"(?<=[.!?])\s+", clean_line(line)):
            part = part.strip()
            if len(part.split()) >= 3:
                units.append(part)
    return units


def extract_requirements(jd_text):
    """
    Requirements = lines after a heading such as 'Requirements' / 'Responsibilities' / 'Qualifications'.
    If no such heading exists, every sentence of the job description is used.
    """
    lines = jd_text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if re.match(r"^\s*(key\s+)?(requirements|responsibilities|qualifications|what you)", line.strip(), re.I):
            start = i + 1
            break
    return split_units("\n".join(lines[start:]))


def anonymize(text):
    """Blind screening: mask e-mails, phone numbers, links and a name-like first line."""
    text = re.sub(r"[\w\.\-+]+@[\w\.\-]+\.\w+", "[EMAIL]", text)
    text = re.sub(r"(https?://\S+|www\.\S+|github\.com/\S+|linkedin\.com/\S+)", "[LINK]", text)
    text = re.sub(r"\+?\d[\d\s\-()]{8,}\d", "[PHONE]", text)
    lines = text.strip().splitlines()
    if lines and len(lines[0].split()) <= 4 and not re.search(r"\d", lines[0]):
        lines[0] = "[NAME]"                              # first short line is almost always the name
    return "\n".join(lines)


def find_skills(text):
    """Return the set of canonical skills mentioned in the text (whole-word matching)."""
    lowered = text.lower()
    found = set()
    for skill, aliases in SKILLS.items():
        for alias in aliases:
            if re.search(r"(?<![a-z0-9+#])" + re.escape(alias) + r"(?![a-z0-9+#])", lowered):
                found.add(skill)
                break
    return found


# ---------------------------------------------------------------------------
# Similarity: Transformer vs keyword baseline
# ---------------------------------------------------------------------------
def semantic_similarity(texts_a, texts_b, model):
    """
    Cosine similarity matrix (len(a) x len(b)) using Transformer sentence embeddings.
    Each text becomes ONE 384-number vector; normalised vectors make the dot product equal the cosine.
    Falls back to the keyword method if the model could not be loaded.
    """
    if model is None:
        return keyword_similarity(texts_a, texts_b)
    vectors_a = model.encode(texts_a, normalize_embeddings=True, show_progress_bar=False)
    vectors_b = model.encode(texts_b, normalize_embeddings=True, show_progress_bar=False)
    return np.clip(np.asarray(vectors_a) @ np.asarray(vectors_b).T, 0, 1)   # negative similarity -> 0


def keyword_similarity(texts_a, texts_b):
    """Baseline: TF-IDF cosine similarity. It only sees shared WORDS, not meaning."""
    try:
        vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        vectorizer.fit(list(texts_a) + list(texts_b))
        return cosine_similarity(vectorizer.transform(texts_a), vectorizer.transform(texts_b))
    except ValueError:                                   # only stop-words
        return np.zeros((len(texts_a), len(texts_b)))


def status_of(similarity, strong, partial):
    if similarity >= strong:
        return "Strong"
    if similarity >= partial:
        return "Partial"
    return "Missing"


# ---------------------------------------------------------------------------
# Scoring one candidate
# ---------------------------------------------------------------------------
def analyze_candidate(requirements, jd_skills, resume_text, model, strong, partial, semantic_weight):
    """
    Returns a dict with the final score, per-requirement evidence, skill gaps and interview questions.
    final score = semantic_weight * requirement_coverage + (1 - semantic_weight) * skill_coverage
    """
    sentences = split_units(resume_text)
    if not sentences or not requirements:
        return None

    semantic = semantic_similarity(requirements, sentences, model)       # (requirements x sentences)
    keyword = keyword_similarity(requirements, sentences)

    credit = {"Strong": 1.0, "Partial": 0.5, "Missing": 0.0}
    evidence, transformer_only = [], 0
    for i, requirement in enumerate(requirements):
        best = int(semantic[i].argmax())                                 # most similar resume sentence
        sim, kw_sim = float(semantic[i, best]), float(keyword[i].max())
        status = status_of(sim, strong, partial)
        if status != "Missing" and kw_sim < 0.10:                        # matched with almost no shared words
            transformer_only += 1
        evidence.append({"requirement": requirement, "evidence": sentences[best],
                         "semantic": round(sim, 3), "keyword": round(kw_sim, 3), "status": status})
    requirement_score = float(np.mean([credit[row["status"]] for row in evidence]))

    resume_skills = find_skills(resume_text)
    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)
    skill_score = len(matched) / len(jd_skills) if jd_skills else None
    if skill_score is None:
        final = requirement_score
    else:
        final = semantic_weight * requirement_score + (1 - semantic_weight) * skill_score

    questions = [f"Describe a project where you used {skill}." for skill in missing[:3]]
    questions += [f"Tell us about your experience with: {row['requirement']}"
                  for row in evidence if row["status"] == "Missing"][:2]

    whole_semantic = float(semantic_similarity([" ".join(requirements)], [resume_text], model)[0, 0])
    whole_keyword = float(keyword_similarity([" ".join(requirements)], [resume_text])[0, 0])
    return {
        "score": round(100 * final, 1),
        "requirement_score": round(100 * requirement_score, 1),
        "skill_score": None if skill_score is None else round(100 * skill_score, 1),
        "evidence": evidence, "matched": matched, "missing": missing, "extra": extra,
        "questions": questions, "transformer_only": transformer_only,
        "avg_semantic": round(float(semantic.max(axis=1).mean()), 3),
        "avg_keyword": round(float(keyword.max(axis=1).mean()), 3),
        "whole_semantic": round(whole_semantic, 3), "whole_keyword": round(whole_keyword, 3),
    }


def verdict(score):
    if score >= 70:
        return "Strong match"
    if score >= 45:
        return "Possible match"
    return "Weak match"
