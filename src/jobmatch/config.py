"""App settings in one place. TAU/ALPHA come from tuning on the dev split (scripts/eval_retrieval.py)."""

INDEX_REPO = "srish327/tech-jobs-index"
INDEX_VERSION = "v1"
SKILL_MODEL = "srish327/deberta-v3-skillspan-crf"
# Chosen on the dev split, never on test: hybrid + skill re-rank had the best dev L2 nDCG@10 (LLM judge, "are the top 10
# good matches?"); plain hybrid tied it (-0.002), and the tie went to re-rank, which won L1 dev MRR by +0.10.
# bm25 + re-rank wins L1 (recover the one source posting) but loses L2. See README "Results".
MODE = "hybrid"
TAU = 0.85  # soft skill match: cosine >= TAU between skill-phrase embeddings
# final score = alpha * retrieval + (1 - alpha) * skill coverage, tuned on dev per base mode (dense: not tuned, uses hybrid's)
ALPHA = {"bm25": 0.2, "hybrid": 0.3, "dense": 0.3}
EXPLAIN_TOP = 3  # LLM explanations for the top 3 only: ~4k tokens/request vs Groq free tier's 8k tokens/minute

SAMPLE_RESUME = """Machine learning engineer with 2 years of experience. Fine-tuned transformer models (BERT, DeBERTa)
for named entity recognition in PyTorch and Hugging Face, beating a published baseline. Built FastAPI services and
packaged them with Docker; set up CI/CD with GitHub Actions and experiment tracking. Comfortable with Python, SQL and
pandas for data analysis. Deployed models with ONNX for in-browser inference. Mentored two interns and presented
results to non-technical stakeholders. B.Tech in Computer Science."""
