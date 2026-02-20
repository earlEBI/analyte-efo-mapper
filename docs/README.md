# GitHub Pages for pQTL Mapper

This folder is a static GitHub Pages site.

## For New Users

The website is documentation and templates only.  
Actual mapping runs locally with Python.

Quick local setup:

```bash
git clone https://github.com/earlEBI/protein-efo-mapper.git
cd protein-efo-mapper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
cp docs/input_template.tsv data/analytes.tsv
```

Run mapping:

```bash
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py map \
  --input data/analytes.tsv \
  --output final_output/analytes_efo.tsv \
  --review-output final_output/review_queue.tsv \
  --index skills/pqtl-measurement-mapper/references/measurement_index.json \
  --workers 8 \
  --parallel-mode process \
  --progress
```

## Publish

1. Push this repository to GitHub.
2. In GitHub, open `Settings -> Pages`.
3. Ensure GitHub Pages is enabled.
4. The workflow `.github/workflows/pages.yml` will deploy this `docs/` folder.

After deployment, your page URL will be:

`https://<your-user-or-org>.github.io/<repo-name>/`

## Note

GitHub Pages is static hosting only.  
The actual mapping runs in:

- `skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py`
- or your local/API-backed web app.
