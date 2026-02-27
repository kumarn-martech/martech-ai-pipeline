<<<<<<< HEAD
# MarTech AI Pipeline (v1): Segmentation → Activation

A beginner-friendly project that simulates a CDP-style customer dataset, builds features, segments users using K-Means, and generates an activation file that can be used to trigger journeys (SFMC-style).

## What it does
- Creates basic behavioral features (email opens, web visits, purchases)
- Builds an engagement score
- Segments customers (K-Means)
- Generates an activation output with a “next best action”

## How to run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/pipeline_v1.py

## Architecture (simple)
CDP-style data (CSV) → Feature Engineering → Segmentation (K-Means) → Activation Output (CSV)
