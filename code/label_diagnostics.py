"""诊断：标注(match_score / Matches)和 CV-职位文本相似度到底有多大关系。

结论（见 README 的 Evaluation 一节）：
  - match_score 与文本相似度几乎不相关（Pearson -0.01, Spearman +0.05），
    所以「预测 match_score 最高的那个职位」这个任务用文本基本做不了；
  - 但标注职位确实比随机职位更相似（1.61x），中位数排在全部职位的前 16.5%，
    说明文本相似度适合做粗筛，不适合做精排。
"""
import numpy as np
from scipy.stats import spearmanr
import matching_app as m

tr = m.load_training_dataset("../data/Matching_training.xlsx")
tc, tj = m.prepare_candidates(tr.candidates), m.prepare_jobs(tr.jobs)

wv, _, _ = m.fit_vectorizers(list(tc["cv_clean"]), list(tj["job_document"]), list(tj["job_title_clean"]))
S = m.linear_kernel(m.transform_texts(wv, tc["cv_clean"]),
                    m.transform_texts(wv, tj["job_document"])).astype(np.float32)

cidx = {int(c): i for i, c in enumerate(tc["CANDIDATE_ID"].astype(int))}
jidx = {str(j): i for i, j in enumerate(tj["Offer_ID"].astype(str))}
pairs = [(cidx[int(r.CANDIDATE_ID)], jidx[str(r.Offer_ID)], float(r.match_score))
         for r in tr.matches.itertuples(index=False)
         if int(r.CANDIDATE_ID) in cidx and str(r.Offer_ID) in jidx]

lab = np.array([S[i, j] for i, j, _ in pairs])
rng = np.random.default_rng(0)
rnd = S[rng.integers(0, S.shape[0], 20000), rng.integers(0, S.shape[1], 20000)]
sc = np.array([s for _, _, s in pairs])
ranks = np.array([(S[i] > S[i, j]).mean() for i, j, _ in pairs])

print(f"标注对相似度 {lab.mean():.4f} vs 随机 {rnd.mean():.4f}  ({lab.mean()/rnd.mean():.2f}x)")
print(f"match_score 相关性  Pearson {np.corrcoef(sc, lab)[0,1]:+.4f}  Spearman {spearmanr(sc, lab).statistic:+.4f}")
print(f"标注职位排序百分位  mean {ranks.mean():.3f}  median {np.median(ranks):.3f}  (随机 0.5)")
