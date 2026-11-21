"""
Evaluation set written for this project (fictional). 3 job descriptions x 5 resumes = 15 labelled pairs.
Labels were assigned by the author: 2 = good fit, 1 = partial fit, 0 = poor fit.
Resumes are reused from samples.py plus one extra candidate (Meera Iyer).
"""
from samples import SAMPLE_JD, SAMPLE_RESUMES

MEERA = """Meera Iyer
meera.iyer@example.com | +91 97000 22334
Summary
Software engineer moving into language technology.
Experience
- Built a tool that sorts incoming support emails into categories using a pretrained BERT model from Hugging Face.
- Wrote Python services that expose predictions to the support team through REST APIs.
- Used PostgreSQL and SQL to analyse ticket data and reported trends to non-technical managers.
- Deployed the application with Docker on Google Cloud and used Git in an agile team."""

RESUMES = dict(SAMPLE_RESUMES)
RESUMES["Meera Iyer"] = MEERA

JD_ANALYST = """Data Analyst - Northwind Retail

Requirements
- Write SQL queries to analyse large sales datasets.
- Build dashboards in Tableau or Power BI for business teams.
- Clean and transform data with Python and pandas.
- Run statistical analysis and A/B tests to measure campaign impact.
- Present insights clearly to managers and non-technical stakeholders.
- Use Excel for quick reporting and checks.
"""

JD_FRONTEND = """Front-End Developer - Bright Interfaces

Requirements
- Build responsive web interfaces with React and JavaScript.
- Turn design mockups into reusable interface components.
- Collaborate with designers and product managers in an agile team.
- Write unit tests for user interface components.
- Improve page loading speed and performance.
- Use Git for version control.
"""

# job title -> (job description, {candidate: label})
EVAL_SET = {
    "ML Engineer (NLP)": (SAMPLE_JD, {"Aisha Khan": 2, "Meera Iyer": 2, "Daniel Roy": 1, "Rahul Mehta": 1, "Priya Nair": 0}),
    "Data Analyst": (JD_ANALYST, {"Rahul Mehta": 2, "Aisha Khan": 1, "Meera Iyer": 1, "Daniel Roy": 1, "Priya Nair": 0}),
    "Front-End Developer": (JD_FRONTEND, {"Priya Nair": 2, "Aisha Khan": 0, "Meera Iyer": 0, "Daniel Roy": 0, "Rahul Mehta": 0}),
}
