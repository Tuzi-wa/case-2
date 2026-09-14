"""在 holdout 上评测匹配排序器。

相对旧版的三处修正：

1. 旧版把权重调参和评测放在同一批候选人上（`tune_rank_weights` 用全量
   matches 选最优权重，然后又在其中的 top-1 上报成绩），属于数据泄漏。
   现在先按候选人切成 tune / eval 两份，权重只在 tune 份上选。

2. 旧版只认 match_score 最高的那一个职位。候选人平均有 3.5 个标注匹配，
   这个口径过严，所以同时报「严格」和「宽松（命中任一标注）」两套。

3. 1,804 个职位里找 1 个，绝对值天然很低，单看没有意义。
   现在一并输出随机基线和提升倍数。

可选：--embeddings sentence 用真正的句向量模型替代 TF-IDF+SVD(LSA)。
"""
import argparse
import os
import random
from collections import defaultdict

import numpy as np

import matching_app as m

SEED = 20240517


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the matching ranker with a leakage-free split.")
    p.add_argument("--train", required=True, help="Path to Matching_training.xlsx")
    p.add_argument("--embeddings", choices=["svd", "sentence", "both"], default="svd",
                   help="svd = 原来的 TF-IDF+TruncatedSVD；sentence = 句向量模型；both = 两者都作为特征")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--tune-frac", type=float, default=0.5, help="用于调权重的候选人比例")
    return p.parse_args()


def build_features(tc, tj, mode, model_name):
    """返回 [特征矩阵]，每个形状为 (候选人数, 职位数)，取值已归一到 [0,1] 量级。"""
    feats, names = [], []

    if mode in ("svd", "both"):
        wv, cvv, tv = m.fit_vectorizers(
            all_candidate_docs=list(tc["cv_clean"]),
            all_job_docs=list(tj["job_document"]),
            all_titles=list(tj["job_title_clean"]),
        )
        c_w = m.transform_texts(wv, tc["cv_clean"])
        c_c = m.transform_texts(cvv, tc["cv_clean"])
        c_t = m.transform_texts(tv, tc["cv_clean"])
        j_w = m.transform_texts(wv, tj["job_document"])
        j_c = m.transform_texts(cvv, tj["job_document"])
        j_t = m.transform_texts(tv, tj["job_title_clean"])

        svd_main, svd_title = m.fit_svd_embeddings(c_w, j_w, c_t, j_t)
        c_me, j_me = m.transform_embeddings(svd_main, c_w), m.transform_embeddings(svd_main, j_w)
        c_te, j_te = m.transform_embeddings(svd_title, c_t), m.transform_embeddings(svd_title, j_t)

        feats += [
            m.linear_kernel(c_w, j_w).astype(np.float32),
            m.linear_kernel(c_c, j_c).astype(np.float32),
            m.linear_kernel(c_t, j_t).astype(np.float32),
            np.clip(c_me @ j_me.T, 0.0, 1.0).astype(np.float32),
            np.clip(c_te @ j_te.T, 0.0, 1.0).astype(np.float32),
        ]
        names += ["word", "char", "title", "svd_main", "svd_title"]

    if mode in ("sentence", "both"):
        from sentence_transformers import SentenceTransformer

        print(f"   >> 加载句向量模型 {model_name} ...")
        st = SentenceTransformer(model_name)
        cand_vec = st.encode(list(tc["cv_clean"]), batch_size=64, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=True)
        job_vec = st.encode(list(tj["job_document"]), batch_size=64, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=True)
        title_vec = st.encode(list(tj["job_title_clean"]), batch_size=64, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=True)
        feats += [
            np.clip(cand_vec @ job_vec.T, 0.0, 1.0).astype(np.float32),
            np.clip(cand_vec @ title_vec.T, 0.0, 1.0).astype(np.float32),
        ]
        names += ["sent_body", "sent_title"]

    return feats, names


def tune_weights(feats, rows, gold, n_trials=64):
    """在 tune 份候选人上挑权重，目标是 MRR。"""
    rng = random.Random(SEED)
    trials = [tuple([1.0] * len(feats))]
    trials += [tuple(round(rng.uniform(0.0, 3.0), 2) for _ in feats) for _ in range(n_trials)]

    best_w, best_mrr = np.ones(len(feats), np.float32), -1.0
    for w in trials:
        wv = np.array(w, np.float32)
        if wv.sum() <= 0:
            continue
        S = sum(a * f for a, f in zip(wv, feats)) / wv.sum()
        mrr = 0.0
        for r in rows:
            order = np.argsort(-S[r])
            hit = np.where(np.isin(order, list(gold[r])))[0]
            if len(hit):
                mrr += 1.0 / (hit[0] + 1)
        mrr /= max(1, len(rows))
        if mrr > best_mrr:
            best_mrr, best_w = mrr, wv
    return best_w, best_mrr


def score(feats, weights, rows, gold, n_jobs, label):
    S = sum(a * f for a, f in zip(weights, feats)) / weights.sum()
    r1 = r5 = r10 = 0
    mrr = 0.0
    n = 0
    for r in rows:
        if not gold[r]:
            continue
        order = np.argsort(-S[r])
        hit = np.where(np.isin(order, list(gold[r])))[0]
        if not len(hit):
            continue
        pos = hit[0] + 1
        n += 1
        r1 += pos <= 1
        r5 += pos <= 5
        r10 += pos <= 10
        mrr += 1.0 / pos
    k = np.mean([len(gold[r]) for r in rows if gold[r]])
    base1, base10 = k / n_jobs, min(1.0, 10 * k / n_jobs)
    print(f"\n  【{label}】 n={n}  (平均每人 {k:.2f} 个正确职位)")
    print(f"    Recall@1 {r1/n:.4f}   Recall@5 {r5/n:.4f}   Recall@10 {r10/n:.4f}   MRR {mrr/n:.4f}")
    print(f"    随机基线 R@1 {base1:.5f}  R@10 {base10:.5f}"
          f"   →  提升 R@1 {(r1/n)/base1:.1f}x  R@10 {(r10/n)/base10:.1f}x")
    return dict(r1=r1 / n, r5=r5 / n, r10=r10 / n, mrr=mrr / n)


def main() -> int:
    args = parse_args()
    training = m.load_training_dataset(os.path.abspath(args.train))
    tc = m.prepare_candidates(training.candidates)
    tj = m.prepare_jobs(training.jobs)

    print(f"候选人 {len(tc)} | 职位 {len(tj)} | 标注匹配 {len(training.matches)}")
    print(f"[Step 1] 构建特征 (embeddings={args.embeddings}) ...")
    feats, names = build_features(tc, tj, args.embeddings, args.model)
    print(f"   >> 特征: {', '.join(names)}")

    cidx = {int(c): i for i, c in enumerate(tc["CANDIDATE_ID"].astype(int))}
    jidx = {str(j): i for i, j in enumerate(tj["Offer_ID"].astype(str))}
    n_jobs = len(jidx)

    # 每个候选人的全部标注匹配 / 仅最高分那一个（都转成行列下标）
    rel = defaultdict(set)
    for r in training.matches.itertuples(index=False):
        if int(r.CANDIDATE_ID) in cidx and str(r.Offer_ID) in jidx:
            rel[cidx[int(r.CANDIDATE_ID)]].add(jidx[str(r.Offer_ID)])
    top1 = (training.matches.sort_values(["CANDIDATE_ID", "match_score"], ascending=[True, False])
            .groupby("CANDIDATE_ID", group_keys=False).head(1))
    strict = defaultdict(set)
    for r in top1.itertuples(index=False):
        if int(r.CANDIDATE_ID) in cidx and str(r.Offer_ID) in jidx:
            strict[cidx[int(r.CANDIDATE_ID)]].add(jidx[str(r.Offer_ID)])

    # ⚠️ 关键修正：按候选人切分，权重只在 tune 份上选，评测只在 eval 份上做
    rows = sorted(rel.keys())
    random.Random(SEED).shuffle(rows)
    cut = int(len(rows) * args.tune_frac)
    tune_rows, eval_rows = rows[:cut], rows[cut:]
    print(f"\n[Step 2] 调参 {len(tune_rows)} 人 / 评测 {len(eval_rows)} 人（互不重叠）")

    weights, tune_mrr = tune_weights(feats, tune_rows, strict)
    print("   >> 选中权重: " + ", ".join(f"{n}={w:.2f}" for n, w in zip(names, weights)))
    print(f"   >> tune 份 MRR {tune_mrr:.4f}（仅供参考，不是成绩）")

    print(f"\n[Step 3] 在 {len(eval_rows)} 个未参与调参的候选人上评测")
    score(feats, weights, eval_rows, strict, n_jobs, "严格：只认 match_score 最高的那一个")
    score(feats, weights, eval_rows, rel, n_jobs, "宽松：命中任一标注匹配")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
