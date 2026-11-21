"""
evaluate.py  -  compares keyword (TF-IDF), Transformer and hybrid ranking on the 15 labelled pairs.

  python evaluate.py                # needs the Transformer (first run downloads it)
  python evaluate.py --keyword-only # keyword baseline only, no model needed

Metrics (per job description, then averaged):
  Top-1 : does the top-ranked candidate have the highest label? (1 or 0)
  Spearman : rank correlation between the method's scores and the labels (-1 to 1)
Writes results/metrics.csv and results/per_job.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matcher
from eval_data import EVAL_SET, RESUMES

STRONG, PARTIAL, WEIGHT = 0.50, 0.35, 0.6                # same defaults as the app


def spearman(scores, labels):
    """Rank correlation (average ranks for ties). Returns None if either list is constant."""
    ranks_a = pd.Series(scores).rank().to_numpy()
    ranks_b = pd.Series(labels).rank().to_numpy()
    if ranks_a.std() == 0 or ranks_b.std() == 0:
        return None
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def main(keyword_only):
    model = None
    if not keyword_only:
        model, error = matcher.load_model()
        if model is None:
            sys.exit(f"Model could not be loaded: {error}\nRun with --keyword-only or fix the download.")
    methods = ["Keyword (TF-IDF)"] if keyword_only else ["Keyword (TF-IDF)", "Semantic (Transformer)", "Hybrid (ResumeLens)"]
    rows = []
    for job, (jd_text, labels) in EVAL_SET.items():
        requirements = matcher.extract_requirements(jd_text)
        jd_skills = matcher.find_skills(jd_text)
        names = list(labels)
        outcomes = {n: matcher.analyze_candidate(requirements, jd_skills, RESUMES[n], model, STRONG, PARTIAL, WEIGHT)
                    for n in names}
        score_lists = {"Keyword (TF-IDF)": [outcomes[n]["avg_keyword"] for n in names]}
        if not keyword_only:
            score_lists["Semantic (Transformer)"] = [outcomes[n]["avg_semantic"] for n in names]
            score_lists["Hybrid (ResumeLens)"] = [outcomes[n]["score"] for n in names]
        truth = [labels[n] for n in names]
        for method in methods:
            scores = score_lists[method]
            top = names[int(np.argmax(scores))]
            rows.append({"Job": job, "Method": method, "Top candidate": top,
                         "Top-1": int(labels[top] == max(truth)), "Spearman": spearman(scores, truth)})
    per_job = pd.DataFrame(rows)
    summary = per_job.groupby("Method", sort=False).agg(Top1=("Top-1", "mean"), Spearman=("Spearman", "mean")).round(3)
    os.makedirs("results", exist_ok=True)
    per_job.to_csv("results/per_job.csv", index=False)
    summary.to_csv("results/metrics.csv")
    print(per_job.to_string(index=False))
    print("\nSummary (mean over the 3 job descriptions):")
    print(summary.to_string())


if __name__ == "__main__":
    main("--keyword-only" in sys.argv)
