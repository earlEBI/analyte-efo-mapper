# GitHub Pages for pQTL Mapper

This folder is a static GitHub Pages site.

## For New Users

The website is documentation and templates only.  
Actual mapping runs locally with Python.

Quick local setup:

```bash
git clone https://github.com/earlEBI/analyte-efo-mapper.git
cd analyte-efo-mapper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
cp docs/input_template.tsv data/analytes.tsv
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py setup-bundled-caches
```

Bundled offline caches used by setup:
- `skills/pqtl-measurement-mapper/references/uniprot_aliases.tsv` (protein alias cache)
- `skills/pqtl-measurement-mapper/references/metabolite_aliases.tsv` (metabolite HMDB-derived alias cache)
- `skills/pqtl-measurement-mapper/references/trait_mapping_cache.tsv` (disease/phenotype trait cache)

These are local repository files; setup does not download these caches from the internet.

Quick Start (5 lines):

```bash
cp docs/input_template.tsv data/analytes.tsv
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py map \
  --input data/analytes.tsv \
  --output final_output/analytes_efo.tsv \
  --review-output final_output/withheld_for_review.tsv
```

This writes your main results to `final_output/analytes_efo.tsv` and withheld items to `final_output/withheld_for_review.tsv`.
`setup-bundled-caches` is offline/local only and validates the bundled protein, metabolite, and trait caches.

Disease/phenotype trait mapping mode (optional):

```bash
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py trait-map \
  --input data/traits.tsv \
  --output final_output/traits_mapped.tsv \
  --trait-cache skills/pqtl-measurement-mapper/references/trait_mapping_cache.tsv \
  --efo-obo skills/pqtl-measurement-mapper/references/efo.obo \
  --min-score 0.82 \
  --review-output final_output/traits_review.tsv \
  --progress
```

Notes:
- This mode is cache-first (curated trait cache), then falls back to `efo.obo`.
- Input can include free-text trait, ICD10, and/or PheCode columns.
- Output includes a per-row `provenance` field suitable for curation notes tabs.

How mapping works (accession vs name input):

- Accession input (`UniProt`, `HMDB`, `ChEBI`, `KEGG`) is treated as highest-confidence identity input.
- For UniProt accessions, the mapper canonicalizes the accession (for example `P12345-2` to `P12345`), resolves aliases from local UniProt alias resources, then searches EFO/OBA labels and synonyms.
- For gene symbol / gene ID / UniProt mnemonic inputs, the mapper first tries to resolve to a stable UniProt accession; if successful, it uses the same accession-based identity checks.
- For free-text protein or metabolite names, the mapper uses synonym/lexical retrieval, then stricter validation gates to avoid family-member drift.
- EFO/OBA matching uses normalized exact phrase matches first, then token retrieval + lexical reranking as fallback.
- Token retrieval means candidate generation by shared informative words/tokens between your query aliases and EFO/OBA labels/synonyms.
- Lexical reranking means scoring those token-retrieved candidates (string similarity + token overlap + synonym evidence, plus matrix/subject adjustments) and ordering best-to-worst before validation.

Validation and context filters:

- Identity validation: accession/concept-aware subject checks reduce wrong mappings from near-name collisions (for example numeric family mismatches like protein 1 vs protein 4).
- Context validation: `--measurement-context blood` (default) allows blood/plasma/unlabeled and excludes serum unless `--additional-contexts serum` is set.
- Keyword context add-on: `--additional-context-keywords` can allow extra free-text contexts (for example `aorta`).
- Ratio safeguard: ratio/quotient traits are blocked unless the query explicitly requests a ratio.
- Auto-validation gate: lower-confidence token-only hits are withheld unless they pass stricter criteria; withheld rows go to `--review-output` when enabled.

Practical expectation:

- Accession-based input is usually most reliable.
- Gene/symbol input can be reliable when alias resolution is unique.
- Free-text name input is more variable and should be reviewed via `withheld_for_review.tsv` and optionally `review_queue.tsv`.

Run mapping (mixed protein + metabolite inputs):

```bash
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py map \
  --input data/analytes.tsv \
  --output final_output/analytes_efo.tsv \
  --index skills/pqtl-measurement-mapper/references/measurement_index.json \
  --entity-type auto \
  --workers 8 \
  --parallel-mode process \
  --progress
```

You can run this directly after clone/install.  
No local HMDB file is required for normal mapping because the repo ships a metabolite alias cache.

Optional (recommended for metabolite mapping): pin your HMDB file version, then build cache once:

```bash
mkdir -p skills/pqtl-measurement-mapper/references/hmdb_source
cp /path/to/your/structures.sdf skills/pqtl-measurement-mapper/references/hmdb_source/structures.sdf
```

Then build aliases (auto-detects local HMDB file):

```bash
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py metabolite-alias-build \
  --output skills/pqtl-measurement-mapper/references/metabolite_aliases.tsv \
  --no-merge-existing
```

If you already downloaded HMDB locally (for example `structures.sdf`), build from local HMDB only:

```bash
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py metabolite-alias-build \
  --output skills/pqtl-measurement-mapper/references/metabolite_aliases.tsv \
  --hmdb-xml /path/to/structures.sdf \
  --no-merge-existing
```

HMDB download/cache notes:
- The command auto-detects HMDB files in `skills/pqtl-measurement-mapper/references/hmdb_source/` and `skills/pqtl-measurement-mapper/references/metabolite_downloads/`.
- If no local HMDB file is found, it auto-downloads from `--hmdb-url` (default: this repo's GitHub release asset).
- You can point `--hmdb-url` to your own private/public GitHub release bundle.
- Maintainer setup: upload your pinned `structures.sdf.gz` as a GitHub Release asset so users can fetch the exact same HMDB version.
- Use `--no-merge-existing` to regenerate a cleaner alias cache (normalized IDs and fewer noisy aliases).
- Practical model:
  - End users: no HMDB step needed unless rebuilding aliases.
  - Rebuild step: pulls your pinned bundle from GitHub release automatically when local HMDB is absent.
  - Fully offline installs: include `structures.sdf` or `structures.sdf.gz` in `references/hmdb_source/`.

Example with explicit pinned bundle URL:

```bash
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py metabolite-alias-build \
  --output skills/pqtl-measurement-mapper/references/metabolite_aliases.tsv \
  --hmdb-url https://github.com/<owner>/<repo>/releases/download/<tag>/structures.sdf.gz \
  --no-merge-existing
```

Optional (recommended after upgrading): refresh EFO measurement cache so metabolite xrefs
(CHEBI/HMDB/KEGG) are available for direct ID-to-term matching:

```bash
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py refresh-efo-cache \
  --term-cache skills/pqtl-measurement-mapper/references/efo_measurement_terms_cache.tsv \
  --index skills/pqtl-measurement-mapper/references/measurement_index.json
```

Then map with it:

```bash
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py map \
  --input data/analytes.tsv \
  --output final_output/analytes_efo.tsv \
  --index skills/pqtl-measurement-mapper/references/measurement_index.json \
  --metabolite-aliases skills/pqtl-measurement-mapper/references/metabolite_aliases.tsv \
  --entity-type auto
```

Optional withheld-for-review output:

```bash
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py map \
  --input data/analytes.tsv \
  --output final_output/analytes_efo.tsv \
  --review-output final_output/withheld_for_review.tsv \
  --index skills/pqtl-measurement-mapper/references/measurement_index.json \
  --entity-type auto \
  --workers 8 \
  --parallel-mode process \
  --progress
```

`final_output/withheld_for_review.tsv` contains only rows where the top candidate was withheld from auto-validation (`reason_code=manual_validation_required`).

Optional broader review queue output:

```bash
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py map \
  --input data/analytes.tsv \
  --output final_output/analytes_efo.tsv \
  --review-queue-output final_output/review_queue.tsv \
  --index skills/pqtl-measurement-mapper/references/measurement_index.json \
  --entity-type auto \
  --workers 8 \
  --parallel-mode process \
  --progress
```

`final_output/review_queue.tsv` is for manual review of closest unresolved candidates; it is useful but can contain probable mismatches.

Advanced optional triage for withheld candidates in the same run:

```bash
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py map \
  --input data/analytes.tsv \
  --output final_output/analytes_efo.tsv \
  --review-output final_output/withheld_for_review.tsv \
  --withheld-triage-output final_output/withheld_for_review_triage.tsv \
  --index skills/pqtl-measurement-mapper/references/measurement_index.json \
  --entity-type auto \
  --workers 8 \
  --parallel-mode process \
  --progress
```

`final_output/withheld_for_review_triage.tsv` classifies top withheld candidates as:

- `reject_high`: likely wrong target.
- `review_needed`: unresolved identity, send to manual review.
- `accept_medium`: accession mismatch but same primary symbol (often alias/isoform-equivalent).
- `accept_high`: candidate subject resolves to same accession.
- Includes `query_primary_label` and `suggested_subject` so you can QC full-name alignment directly.

Refresh EFO/OBA measurement cache and rebuild index:

```bash
.venv/bin/python skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py refresh-efo-cache \
  --download-url https://github.com/EBISPOT/efo/releases/latest/download/efo.obo \
  --term-cache skills/pqtl-measurement-mapper/references/efo_measurement_terms_cache.tsv \
  --index skills/pqtl-measurement-mapper/references/measurement_index.json \
  --analyte-cache skills/pqtl-measurement-mapper/references/analyte_to_efo_cache.tsv \
  --uniprot-aliases skills/pqtl-measurement-mapper/references/uniprot_aliases.tsv \
  --metabolite-aliases skills/pqtl-measurement-mapper/references/metabolite_aliases.tsv
```

Notes:

- You do not need a local OBO file for this command.
- Index rebuild is enabled by default and should usually be kept on.

Defaults and options:

- Default `--measurement-context blood` excludes serum-specific terms.
- Add serum explicitly with `--additional-contexts serum`.
- Output IDs are written with underscores (for example `EFO_0802947`).

Measurement context usage:

- Blood default (blood + plasma + unlabeled, excludes serum):
  - `--measurement-context blood`
- Plasma-only:
  - `--measurement-context plasma`
- Blood plus serum:
  - `--measurement-context blood --additional-contexts serum`
- CSF-only:
  - `--measurement-context cerebrospinal_fluid`
- Free-text context add-on:
  - `--additional-context-keywords aorta`
  - `--additional-context-keywords "adipose tissue"`
- Free-text only filter mode:
  - `--measurement-context auto --additional-context-keywords aorta`

Useful optional arguments:

- `--top-k` number of candidates to keep per input.
- `--min-score` minimum score threshold.
- `--workers` parallel workers.
- `--name-mode strict|fuzzy` handling for name-like inputs.
- `--entity-type auto|protein|metabolite` routing/validation mode.
- `--auto-enrich-uniprot` checks UniProt for accession-like queries missing from your local alias file, appends returned aliases to `uniprot_aliases.tsv`, and rebuilds index before mapping.
  - It is incremental for current input queries, not a full UniProt rebuild.
- `--metabolite-aliases` local metabolite concept aliases for HMDB/ChEBI/KEGG/name resolution.
- `--unmapped-output` write unresolved rows to a separate TSV.
- `--review-output` write withheld-from-auto-validation rows (`withheld_for_review.tsv`).
- `--review-queue-output` write optional broader manual-review suggestions (`review_queue.tsv`).
- `--withheld-triage-output` triage top withheld candidates into accept/reject/review buckets (requires `--review-output`).

The published site URL is:

`https://earlebi.github.io/analyte-efo-mapper/`
