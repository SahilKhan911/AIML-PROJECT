# Deployment Guide

## Option 1: Streamlit Community Cloud (Recommended)
1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Select the repo and set `app.py` as entry point.
4. Deploy.

## Option 2: Render
1. Push this repository to GitHub.
2. In Render, create a new Web Service from repo.
3. Render will detect `render.yaml` automatically.
4. Deploy.

## Local run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
