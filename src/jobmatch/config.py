"""App settings in one place. TAU/ALPHA come from tuning on the dev split (scripts/eval_retrieval.py)."""

INDEX_REPO = "srish327/tech-jobs-index"
INDEX_VERSION = "v1"
SKILL_MODEL = "srish327/deberta-v3-skillspan-crf"
# Chosen on the dev split (scripts/eval_retrieval.py -> results/l1_retrieval.json): base retrieval mode for the
# app, and the re-rank settings tuned for it.
MODE = "hybrid"
TAU = 0.75  # soft skill match: cosine >= TAU between skill-phrase embeddings
ALPHA = 0.5  # final score = ALPHA * retrieval + (1 - ALPHA) * skill coverage

SAMPLE_RESUME = """Machine learning engineer with 2 years of experience. Fine-tuned transformer models (BERT, DeBERTa)
for named entity recognition in PyTorch and Hugging Face, beating a published baseline. Built FastAPI services and
packaged them with Docker; set up CI/CD with GitHub Actions and experiment tracking. Comfortable with Python, SQL and
pandas for data analysis. Deployed models with ONNX for in-browser inference. Mentored two interns and presented
results to non-technical stakeholders. B.Tech in Computer Science."""
