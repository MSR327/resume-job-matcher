# Deploy (Streamlit Community Cloud)

1. https://share.streamlit.io → **Create app** → **Deploy a public app from GitHub**.
2. Repository `MSR327/resume-job-matcher`, branch `main`, main file `streamlit_app.py`.
3. **Advanced settings:** Python **3.12**. Secrets:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
   (Paste it yourself. Without it the app still works, without LLM explanations.)
4. Deploy. The first start downloads ~600 MB (ONNX models + index) and takes a few minutes; later visits are fast.
5. Check: paste the sample resume, expect 5 jobs with explanations in ~3–8 s; the **Stats** page shows the request.

Memory: ~970 MB measured locally (limits per the docs: 690 MB minimum, ~2.7 GB maximum). If the app shows
"over its resource limits", reboot it from **Manage app**.

Rotating the key: Groq console → API keys → revoke, create a new one, update the app's Secrets, and update `.env`.
