
# Etsy Bridge (Enterprise+) — Streamlit Cloud Ready

Adds:
- **Exact Phrase Coverage**: checks each listing against target phrases.
- **Title/Tag A/B Generator**: two optimized variants for CTR testing.
- **One‑Click Client Report (PDF)**: bundles Opportunities + Coverage + UTM plan.

Other features retained:
- Live **GSC** (service account), **Google Trends**, **Pinterest CSV/paste**.
- **Opportunity Score** merge (GSC × Trends).
- **Bulk UTM** builder + **Content Briefs**.
- Exports via **fpdf2** (Cloud‑safe).

## Deploy
1) Push to a **public GitHub** repo.
2) Streamlit Cloud → Create app → repo, branch `main`, file `app.py`.
3) Add **GSC service account** JSON to **Secrets** under `[gcp_service_account]`.
