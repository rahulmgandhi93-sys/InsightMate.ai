# InsightMate.ai
# InsightMate — Privacy-first Data Analytics Assistant

**Tagline:** Ask your data anything — private, local, and free.

InsightMate converts natural-language questions into pandas code, executes safely on your dataset, and returns charts + concise natural-language summaries. This repo contains the prototype, demo notebook, and a Streamlit front-end for local or cloud deployment.

## Features
- Upload CSV (<= 5MB)
- Natural language → pandas code (Flan-T5)
- Safe execution sandbox
- Charts + NL summaries + export (PNG / code / slide)
- Runs locally (no paid APIs required)

## Quick start (local)
1. Clone:
```bash
git clone https://github.com/YOUR_USERNAME/insightmate.git
cd insightmate
