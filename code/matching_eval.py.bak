import argparse
import os

import numpy as np

import matching_app as m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the matching ranker on a training holdout split.")
    parser.add_argument("--train", required=True, help="Path to Matching_training.xlsx")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_path = os.path.abspath(args.train)

    training = m.load_training_dataset(train_path)
    train_candidates = m.prepare_candidates(training.candidates)
    train_jobs = m.prepare_jobs(training.jobs)

    word_vectorizer, char_vectorizer, title_vectorizer = m.fit_vectorizers(
        all_candidate_docs=list(train_candidates["cv_clean"]),
        all_job_docs=list(train_jobs["job_document"]),
        all_titles=list(train_jobs["job_title_clean"]),
    )

    train_candidate_word = m.transform_texts(word_vectorizer, train_candidates["cv_clean"])
    train_candidate_char = m.transform_texts(char_vectorizer, train_candidates["cv_clean"])
    train_candidate_title = m.transform_texts(title_vectorizer, train_candidates["cv_clean"])
    train_job_word = m.transform_texts(word_vectorizer, train_jobs["job_document"])
    train_job_char = m.transform_texts(char_vectorizer, train_jobs["job_document"])
    train_job_title = m.transform_texts(title_vectorizer, train_jobs["job_title_clean"])

    svd_main, svd_title = m.fit_svd_embeddings(
        train_candidate_word=train_candidate_word,
        train_job_word=train_job_word,
        train_candidate_title=train_candidate_title,
        train_job_title=train_job_title,
    )
    train_candidate_main_emb = m.transform_embeddings(svd_main, train_candidate_word)
    train_job_main_emb = m.transform_embeddings(svd_main, train_job_word)
    train_candidate_title_emb = m.transform_embeddings(svd_title, train_candidate_title)
    train_job_title_emb = m.transform_embeddings(svd_title, train_job_title)

    train_feature_matrices = [
        m.linear_kernel(train_candidate_word, train_job_word).astype(np.float32),
        m.linear_kernel(train_candidate_char, train_job_char).astype(np.float32),
        m.linear_kernel(train_candidate_title, train_job_title).astype(np.float32),
        np.clip(train_candidate_main_emb @ train_job_main_emb.T, 0.0, 1.0).astype(np.float32),
        np.clip(train_candidate_title_emb @ train_job_title_emb.T, 0.0, 1.0).astype(np.float32),
    ]

    weights = m.tune_rank_weights(
        matches=training.matches,
        candidate_ids=list(train_candidates["CANDIDATE_ID"].astype(int)),
        job_ids=list(train_jobs["Offer_ID"].astype(str)),
        feature_matrices=train_feature_matrices,
    )

    score_matrix = sum(weight * feature for weight, feature in zip(weights, train_feature_matrices))
    score_matrix = score_matrix / max(1.0, float(weights.sum()))

    ranked_matches = training.matches.sort_values(["CANDIDATE_ID", "match_score"], ascending=[True, False])
    holdout = ranked_matches.groupby("CANDIDATE_ID", group_keys=False).head(1)

    candidate_index = {
        int(candidate_id): idx for idx, candidate_id in enumerate(train_candidates["CANDIDATE_ID"].astype(int))
    }
    job_ids = train_jobs["Offer_ID"].astype(str).to_numpy()

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    reciprocal_rank = 0.0

    for row in holdout.itertuples(index=False):
        candidate_idx = candidate_index[int(row.CANDIDATE_ID)]
        true_offer = str(row.Offer_ID)
        ranking = np.argsort(-score_matrix[candidate_idx])
        ranked_offer_ids = job_ids[ranking]
        true_position = int(np.where(ranked_offer_ids == true_offer)[0][0]) + 1
        recall_1 += int(true_position <= 1)
        recall_5 += int(true_position <= 5)
        recall_10 += int(true_position <= 10)
        reciprocal_rank += 1.0 / true_position

    total = max(1, len(holdout))
    print(f"Holdout candidates: {total}")
    print(
        "Tuned weights: "
        f"word={weights[0]:.4f}, char={weights[1]:.4f}, title={weights[2]:.4f}, "
        f"main_emb={weights[3]:.4f}, title_emb={weights[4]:.4f}"
    )
    print(f"Recall@1: {recall_1 / total:.4f}")
    print(f"Recall@5: {recall_5 / total:.4f}")
    print(f"Recall@10: {recall_10 / total:.4f}")
    print(f"MRR: {reciprocal_rank / total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
