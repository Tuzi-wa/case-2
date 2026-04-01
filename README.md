# SearchMatch Case 2

This folder contains the current matching project for candidate-job ranking, vector retrieval, and local demo presentation.

## Folder structure

- `code/`
  - `matching_app.py`: main ranking pipeline
  - `matching_eval.py`: holdout evaluation script
  - `vector_db_matcher.py`: local vector-database style retrieval pipeline
  - `postgres_vector_setup.py`: PostgreSQL + pgvector setup and data loading
  - `postgres_vector_search.py`: PostgreSQL vector-search querying and export
- `data/`
  - `Matching_training.xlsx`
  - `Matching_validation.xlsx`
- `results/`
  - `Matching_validation_scored.xlsx`
  - `Matching_validation_vector_db.xlsx`
  - `vector_db_store/`
- `demo/`
  - `streamlit_app.py`
- `docs/`
  - project PDF / theory notes

## What the project does

The system reads candidate CVs and job posts, converts them into text features and semantic vectors, scores every candidate-job pair, and ranks the best matches.

There are two result views:

- Matching engine result: ranking-oriented scoring output
- Vector retrieval result: local vector database style nearest-neighbor retrieval
- PostgreSQL vector result: pgvector-backed similarity search

## Main outputs

- `results/Matching_validation_scored.xlsx`
  - full matching output
  - top 20 jobs per candidate
  - top 5 jobs per candidate
  - top 5 matches sorted by job title
- `results/Matching_validation_vector_db.xlsx`
  - vector-based top jobs per candidate
  - vector-based results sorted by job title
- `results/Matching_validation_postgres_vector.xlsx`
  - PostgreSQL + pgvector search results
- `results/vector_db_store/`
  - persisted local vector store files

## Requirements

Recommended Python environment:

- Python 3.9+
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `openpyxl`
- `streamlit`

If needed, install:

```bash
pip install pandas numpy scikit-learn scipy openpyxl streamlit
```

## Run the matching engine

From the `code` folder:

```bash
python matching_app.py \
  --train "/Users/mi/Desktop/case 2/data/Matching_training.xlsx" \
  --validation "/Users/mi/Desktop/case 2/data/Matching_validation.xlsx" \
  --output "/Users/mi/Desktop/case 2/results/Matching_validation_scored.xlsx"
```

## Run the evaluation script

From the `code` folder:

```bash
python matching_eval.py \
  --train "/Users/mi/Desktop/case 2/data/Matching_training.xlsx"
```

This prints ranking metrics such as:

- `Recall@1`
- `Recall@5`
- `Recall@10`
- `MRR`

## Run the vector database pipeline

From the `code` folder:

```bash
python vector_db_matcher.py \
  --train "/Users/mi/Desktop/case 2/data/Matching_training.xlsx" \
  --validation "/Users/mi/Desktop/case 2/data/Matching_validation.xlsx" \
  --store-dir "/Users/mi/Desktop/case 2/results/vector_db_store" \
  --output "/Users/mi/Desktop/case 2/results/Matching_validation_vector_db.xlsx" \
  --top-n 5
```

## Run the PostgreSQL + pgvector setup

Prerequisites:

- PostgreSQL is running locally
- `pgvector` is installed for the local PostgreSQL server
- `psql` and `createdb` are available in the shell

From the `code` folder:

```bash
python postgres_vector_setup.py \
  --train "/Users/mi/Desktop/case 2/data/Matching_training.xlsx" \
  --validation "/Users/mi/Desktop/case 2/data/Matching_validation.xlsx" \
  --db-name "searchmatch_case2" \
  --store-dir "/Users/mi/Desktop/case 2/results/vector_db_store_pg" \
  --reset
```

What it does:

- creates the target PostgreSQL database if needed
- checks and enables `pgvector`
- generates candidate/job embeddings
- loads them into `candidate_vectors` and `job_vectors`
- creates cosine vector indexes

## Run PostgreSQL vector search

From the `code` folder:

```bash
python postgres_vector_search.py \
  --db-name "searchmatch_case2" \
  --top-n 5 \
  --output "/Users/mi/Desktop/case 2/results/Matching_validation_postgres_vector.xlsx"
```

This creates a PostgreSQL-backed result workbook with:

- top jobs per candidate from SQL vector search
- results sorted by job title
- metadata exported from the database

## Run the local demo

From any location:

```bash
streamlit run "/Users/mi/Desktop/case 2/demo/streamlit_app.py"
```

The demo reads by default:

- `/Users/mi/Desktop/case 2/results/Matching_validation_scored.xlsx`
- `/Users/mi/Desktop/case 2/results/Matching_validation_vector_db.xlsx`

## Demo views

- `Candidate Top Matches`
  - shows the best jobs for a selected candidate
  - includes score, CV preview, job requirements preview, and match reason
- `Job Title View`
  - shows top candidate matches grouped by job title
- `Vector Retrieval`
  - shows vector-database style retrieval results
- `Matching vs Vector`
  - compares classical matching and vector retrieval

## Meaning of the results

The score is a relative matching score, not an absolute truth value.

That means:

- a higher score means the system considers the candidate more suitable for that job than lower-ranked jobs
- the most important output is the ranking order
- `match_reason` explains the main overlap between CV and job requirements

## Notes

- The project is designed for local use on this machine structure.
- If file paths change, update them either in the command line or in `demo/streamlit_app.py`.
