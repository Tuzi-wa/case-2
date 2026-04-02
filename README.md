# SearchMatch Case 2

Candidate-job matching project for ranking, vector retrieval, and PostgreSQL + pgvector search.

## Overview

This project builds a small matching engine that links candidate CVs to job posts and ranks the best opportunities for each candidate.

The repository includes three retrieval strategies:

- `Matching engine`
  - hybrid ranking with keyword overlap, TF-IDF similarity, title similarity, and semantic embeddings
- `Local vector retrieval`
  - nearest-neighbor search in a local embedding space
- `PostgreSQL vector search`
  - database-backed retrieval using `pgvector`

The goal is not only to compute a score, but to produce a useful ranking and explain why a candidate is aligned with a job.

## Repository structure

- `code/`
  - `matching_app.py`: main ranking pipeline
  - `matching_eval.py`: holdout evaluation script
  - `vector_db_matcher.py`: local vector retrieval pipeline
  - `postgres_vector_setup.py`: PostgreSQL + pgvector setup and loading
  - `postgres_vector_search.py`: PostgreSQL vector search and export
- `data/`
  - `Matching_training.xlsx`
  - `Matching_validation.xlsx`
- `results/`
  - `Matching_validation_scored.xlsx`
  - `Matching_validation_vector_db.xlsx`
  - `Matching_validation_postgres_vector.xlsx`
  - `vector_db_store/`
  - `vector_db_store_pg/`
- `demo/`
  - `streamlit_app.py`
- `docs/`
  - theory material and project notes

## Data flow

1. Read CVs and job posts from the training and validation Excel files.
2. Clean and normalize text.
3. Build text representations and semantic embeddings.
4. Score or retrieve candidate-job pairs with different strategies.
5. Export ranked results to Excel.
6. Present the results in a local Streamlit demo.

## Main deliverables

### 1. Matching engine results

File:

- `results/Matching_validation_scored.xlsx`

Contains:

- full candidate-job ranking output
- top 20 jobs per candidate
- top 5 jobs per candidate
- top 5 matches sorted by job title
- match explanations

### 2. Local vector retrieval results

Files:

- `results/Matching_validation_vector_db.xlsx`
- `results/vector_db_store/`

Contains:

- local vector retrieval results
- persisted vector snapshots and metadata

### 3. PostgreSQL + pgvector results

Files:

- `results/Matching_validation_postgres_vector.xlsx`
- `results/vector_db_store_pg/`

Contains:

- PostgreSQL vector-search results
- database-oriented vector snapshots and metadata

## Requirements

Recommended environment:

- Python 3.9+
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `openpyxl`
- `streamlit`

Install Python dependencies if needed:

```bash
pip install pandas numpy scikit-learn scipy openpyxl streamlit
```

For the PostgreSQL part:

- PostgreSQL running locally
- `pgvector` installed
- `psql` and `createdb` available in the shell

## How to run

### Run the main matching engine

From `code/`:

```bash
python matching_app.py \
  --train "/Users/mi/Desktop/case 2/data/Matching_training.xlsx" \
  --validation "/Users/mi/Desktop/case 2/data/Matching_validation.xlsx" \
  --output "/Users/mi/Desktop/case 2/results/Matching_validation_scored.xlsx"
```

### Run the evaluation script

From `code/`:

```bash
python matching_eval.py \
  --train "/Users/mi/Desktop/case 2/data/Matching_training.xlsx"
```

This prints ranking-oriented metrics such as:

- `Recall@1`
- `Recall@5`
- `Recall@10`
- `MRR`

### Run the local vector retrieval pipeline

From `code/`:

```bash
python vector_db_matcher.py \
  --train "/Users/mi/Desktop/case 2/data/Matching_training.xlsx" \
  --validation "/Users/mi/Desktop/case 2/data/Matching_validation.xlsx" \
  --store-dir "/Users/mi/Desktop/case 2/results/vector_db_store" \
  --output "/Users/mi/Desktop/case 2/results/Matching_validation_vector_db.xlsx" \
  --top-n 5
```

### Set up PostgreSQL + pgvector

From `code/`:

```bash
python postgres_vector_setup.py \
  --train "/Users/mi/Desktop/case 2/data/Matching_training.xlsx" \
  --validation "/Users/mi/Desktop/case 2/data/Matching_validation.xlsx" \
  --db-name "searchmatch_case2" \
  --store-dir "/Users/mi/Desktop/case 2/results/vector_db_store_pg" \
  --reset
```

This step:

- creates the target database if needed
- enables `pgvector`
- generates embeddings
- loads candidate and job vectors into PostgreSQL
- creates cosine vector indexes

### Run PostgreSQL vector search

From `code/`:

```bash
python postgres_vector_search.py \
  --db-name "searchmatch_case2" \
  --top-n 5 \
  --output "/Users/mi/Desktop/case 2/results/Matching_validation_postgres_vector.xlsx"
```

## Demo

Launch the local Streamlit demo:

```bash
streamlit run "/Users/mi/Desktop/case 2/demo/streamlit_app.py"
```

The demo compares all three approaches:

- matching engine
- local vector retrieval
- PostgreSQL vector search

Main views:

- `Candidate Top Matches`
- `Job Title View`
- `Vector Retrieval`
- `Postgres Vector`
- `Compare All`

## Interpreting the results

The scores are relative ranking scores, not absolute truth values.

This means:

- higher scores suggest stronger alignment between the CV and the job post
- the most important output is the ranking order
- different methods may return different top jobs because they focus on different signals
- `match_reason` explains the main overlap between the CV and the job requirements

The comparison view is mainly for method comparison, not for claiming final accuracy by itself.

## Notes

- This repository is structured for local execution on the current machine paths.
- Large result files are tracked with Git LFS where needed.
- If file paths change, update them either in the CLI commands or in `demo/streamlit_app.py`.
