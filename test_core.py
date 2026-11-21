"""Self-check. Run:  python test_core.py   (first run downloads the model, about 90 MB)"""
import numpy as np
import matcher
from samples import SAMPLE_JD, SAMPLE_RESUMES

model, error = matcher.load_model()
assert model is not None, f"Model did not load: {error}"
dim = model.get_sentence_embedding_dimension()
print("model loaded, embedding size:", dim, "(all-MiniLM-L6-v2 should print 384)")

# 1. structure checks
vectors = model.encode(["hello world", "machine learning"], normalize_embeddings=True)
assert vectors.shape == (2, dim) and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
requirements = matcher.extract_requirements(SAMPLE_JD)
assert len(requirements) == 8
assert matcher.status_of(0.6, 0.5, 0.35) == "Strong" and matcher.status_of(0.4, 0.5, 0.35) == "Partial"
assert matcher.status_of(0.1, 0.5, 0.35) == "Missing"
assert "[EMAIL]" in matcher.anonymize("Jo Doe\njo@x.com")
assert matcher.find_skills("I use Python and PyTorch") == {"python", "pytorch"}
assert matcher.analyze_candidate([], set(), "text here now", model, 0.5, 0.35, 0.6) is None

# 2. meaning check (printed, not asserted: it depends on the real pretrained weights)
a = "Created web services that other teams can call."
related = float(matcher.semantic_similarity([a], ["Build REST APIs for other teams."], model)[0, 0])
unrelated = float(matcher.semantic_similarity([a], ["I enjoy hiking on weekends."], model)[0, 0])
keyword = float(matcher.keyword_similarity([a], ["Build REST APIs for other teams."])[0, 0])
print(f"related pair: Transformer {related:.2f} vs keywords {keyword:.2f} | unrelated pair: Transformer {unrelated:.2f}")
print("CHECK related > unrelated:", "OK" if related > unrelated else "UNEXPECTED, tell your helper")

# 3. ranking of the demo candidates (READ THIS BEFORE YOUR DEMO)
skills = matcher.find_skills(SAMPLE_JD)
print("\nDemo ranking (strong>=0.50, partial>=0.35, weight 0.6):")
scores = {}
for name, text in SAMPLE_RESUMES.items():
    r = matcher.analyze_candidate(requirements, skills, text, model, 0.50, 0.35, 0.6)
    scores[name] = r["score"]
    print(f"  {name:12} score {r['score']:5}  requirements {r['requirement_score']:5}  skills {r['skill_score']}")
print("ALL STRUCTURE CHECKS PASSED")
