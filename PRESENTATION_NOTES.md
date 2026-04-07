# SearchMatch Case 2 Presentation Notes

## 5-7 Minute Presentation Script

### 1. Project goal

Our project goal was to build a candidate-job matching system.

The task was not simply to decide whether a candidate matches a job or not, but to rank job offers for each candidate and identify the most suitable opportunities.

For the validation task, the final requirement was to provide, for each candidate, the top 5 matching job offers in a template file with 3 columns:

- `CANDIDATE_ID`
- `RANK`
- `OFFER_ID`

### 2. Data used

We worked with two main Excel files:

- `Matching_training.xlsx`
- `Matching_validation.xlsx`

The training file contains:

- candidate CVs
- job posts
- historical match scores

The validation file contains:

- candidate CVs
- job posts

but no final matching labels, so we had to generate the rankings ourselves.

### 3. Core logic

We built a matching pipeline with several stages.

First, we cleaned and normalized the CV and job-post text.

Second, we transformed the text into numerical representations using:

- keyword-based and TF-IDF features
- title similarity
- semantic embeddings

Third, we combined these features into a ranking-based matching engine that scores every candidate-job pair.

The main idea is that a higher score means stronger alignment between a candidate profile and a job offer.

### 4. Additional methods

To go beyond the basic ranking engine, we also developed two vector-search versions.

The first is a local vector retrieval pipeline, where candidate and job embeddings are compared directly in a local vector space.

The second is a PostgreSQL + pgvector version, where embeddings are stored in a database and retrieved using vector similarity search.

This was important because the latest project requirement explicitly asked us to work on:

- PostgreSQL setup with vectors
- vector search

### 5. Outputs we produced

We generated several result files:

- `Matching_validation_scored.xlsx`
- `Matching_validation_vector_db.xlsx`
- `Matching_validation_postgres_vector.xlsx`

We also created a local Streamlit demo to compare:

- the matching engine
- local vector retrieval
- PostgreSQL vector search

### 6. Final validation file

For the final validation submission, we used the main matching engine results and extracted the top 5 offers for each candidate.

We then filled the teacher's template file:

- `Template results 2nd Work.xlsx`

The final submission contains:

- `6485` rows
- `1297` candidates
- ranked offers from `1` to `5`

This matches the expected maximum size described in the instructions:

- `1297 x 5 = 6485`

### 7. How to interpret the result

The ranking score is not an absolute truth value.

Instead, it is a relative measure of how strongly a candidate appears to fit a job compared with the other available jobs.

So the most important output is the ranking order:

- which jobs appear in the top 5
- and in what order

### 8. Conclusion

In summary, we built a full matching workflow that:

- reads CVs and job posts
- computes candidate-job similarity
- ranks offers
- supports vector retrieval
- supports PostgreSQL vector search
- and produces the final required validation template for submission

The final filled template is ready for delivery, and we also prepared an interactive demo to explain the project results.

## Shorter 1-Minute Backup Version

We built a candidate-job matching system that ranks job offers for each candidate rather than making a simple yes/no decision. Using the training data, we created a ranking engine based on text similarity, title similarity, and semantic embeddings. We also extended the system with local vector retrieval and PostgreSQL + pgvector search. For the final submission, we used the main matching engine to generate the top 5 offers for each of the 1297 validation candidates and filled the required template file with `CANDIDATE_ID`, `RANK`, and `OFFER_ID`. The final file contains 6485 rows and is ready to submit.
