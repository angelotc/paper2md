# Papers summary (auto-generated)

_Generated: 2026-01-04 10:46_

This file summarizes the PDFs in `papers/` for use as context in this codebase.

## DealSeek: current personalization context
_Source: `docs/PERSONALIZATION_PLAN.md` (summary of what the codebase currently implements)._

- **High-level flow**: `/deals?userId=...` runs **regular trending feed** and **personalized candidate fetch** in parallel, then blends/interleaves results (target: **1 personalized : 2 regular**).
- **User preference representation**: time-weighted centroids per **(user, category)** stored in Milvus (`user_preferences`).
- **Candidate retrieval**: Milvus vector search using top categories + deal embeddings from Milvus (`deals`), followed by SQL fetch of details with **quality gating** pushed down (e.g. `deal_score >= 25`).
- **Diversity**: multi-centroid blending (top 3 categories) uses softmax allocation + a floor to avoid single-category domination.
- **Latency**: parallelized via `errgroup`, plus **RAM preference cache** (`TTLCache`, 1h TTL) to avoid repeated Milvus calls.
- **Fallbacks**: no `userId`, no prefs, Milvus failure, or all personalized candidates gated → standard ranking/trending only.
- **Key implementation files**: `backend/go/internal/handlers/deals.go`, `backend/go/internal/service/personalization.go`, `backend/go/internal/milvus/preferences.go`, `backend/go/internal/service/deals_service.go`.

## How to use this
- Use each paper's **Practical takeaways for DealSeek** section for prompting/feature ideation.
- If you set `OPENAI_API_KEY`, re-run the generator to get deeper method + results summaries.

## Index
- [Beyond Immediate Click: Engagement-Aware and MoE-Enhanced Transformers for Sequential Movie Recommendation](#beyond-immediate-click-engagement-aware-and-moe-enhanced-transformers-for-sequential-movie-recommendation)
- [Beyond Top-1: Addressing Inconsistencies in Evaluating Counterfactual Explanations for Recommender Systems](#beyond-top-1-addressing-inconsistencies-in-evaluating-counterfactual-explanations-for-recommender-systems)
- [Coarse-to-fine Dynamic Uplift Modeling for Real-time Video Recommendation](#coarse-to-fine-dynamic-uplift-modeling-for-real-time-video-recommendation)
- [Exploring Scaling Laws of CTR Model for Online Performance Improvement](#exploring-scaling-laws-of-ctr-model-for-online-performance-improvement)
- [LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders](#longer-scaling-up-long-sequence-modeling-in-industrial-recommenders)
- [Modeling Long-term User Behaviors with Diffusion-driven Multi-interest Network for CTR Prediction](#modeling-long-term-user-behaviors-with-diffusion-driven-multi-interest-network-for-ctr-prediction)
- [Non-parametric Graph Convolution for Re-ranking in Recommendation Systems](#non-parametric-graph-convolution-for-re-ranking-in-recommendation-systems)
- [Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation](#prompt-to-slate-diffusion-models-for-prompt-conditioned-slate-generation)
- [Test-Time Alignment with State Space Model for Tracking User Interest Shis in Sequential Recommendation](#test-time-alignment-with-state-space-model-for-tracking-user-interest-shi-s-in-sequential-recommendation)
- [Unified Embedding Based Personalized Retrieval in Siddharth Subramaniyam†*](#unified-embedding-based-personalized-retrieval-in-siddharth-subramaniyam)
- [You Don't Bring Me Flowers: Mitigating Unwanted Recommendations Through Conformal Risk Control](#you-don-t-bring-me-flowers-mitigating-unwanted-recommendations-through-conformal-risk-control)

## Summaries

## Beyond Immediate Click: Engagement-Aware and MoE-Enhanced Transformers for Sequential Movie Recommendation

- **Source PDF**: `papers/Beyond Immediate Click Engagement-Aware and MoE-Enhanced Transformers for Sequential Movie Recommendation.pdf`
- **DOI**: `https://doi.org/10.1145/3705328.3748076`

### TL;DR (3 bullets)
- Problem: Sequential recommenders over-optimize for immediate clicks and use weak negatives / fixed history lengths, producing high-click but low-engagement suggestions.
- Approach: Combine personalized hard-negative sampling (PHNS), an adaptive context-aware Mixture-of-Experts transformer (S-MoE), engagement-aware multi-task losses (CTR + contrastive ranking + optional completion regression) and soft-label next-K training (K=3, labels 1.0/0.6/0.3).
- Result: On a Prime Video dataset (1M users, ~7.24M sequences) the best model (S-MoE-BST + PHNS + MTL + personalized loss) improves NDCG@1 by up to +3.52% vs BST; gains are statistically significant (paired t-test p << 1e-8).

### Problem
Sequential next-item recommenders:
- Are trained to maximize immediate clicks (CTR) and tend to neglect downstream engagement (watch/completion).
- Use weak negative sampling and fixed-length histories, which hurts learning of long(er)-term patterns.
- As a result, they can recommend items that attract clicks but not sustained engagement (partial watches, abandoned viewing).

### Approach
Key components implemented and tested:
- Personalized Hard Negative-aware Sampling (PHNS)
  - Three negative pools: user-specific partial-watch “hard” negatives, globally trending, globally tail.
  - Sampling proportions conditioned on user completion rates (completion = seconds viewed / runtime).
  - Low completion threshold used: 0.05 to define hard negatives.
- Adaptive Context-aware Mixture-of-Experts Transformer (S-MoE)
  - Separate transformer experts specialized for short, mid (and optional long) term context windows.
  - Adaptive gating MLP routes sequence to experts using softmax with temperature and Gaussian noise.
  - Entropy regularization with target entropy (TE) ≈ 0.52 to avoid expert collapse while preserving specialization.
- Multi-task learning with Engagement-aware Personalized Loss
  - Jointly optimize CTR (BCE), contrastive ranking, and optional completion-rate regression.
  - Scale training losses per <user,item> by completion rate using per-user weighting to prioritize high-engagement signals.
  - Regression head included during training but discarded at inference.
- Soft-label Multi-K training (next-K prediction)
  - Extend next-item prediction to K=3 with soft positives (example schedule: 1.0, 0.6, 0.3) so model focuses on near-term accuracy while learning farther horizons.

Noted design/hyperparameter choices: K=3, soft-labels (1.0,0.6,0.3), completion threshold 0.05, TE ≈ 0.52.

### Results (include numbers)
- Dataset: Prime Video — ~1M users, ~7.24M sequences.
- Best-model improvement: S-MoE-BST with PHNS + MTL + personalized loss achieves up to +3.52% NDCG@1 over baseline BST.
- Statistical significance: improvements remain highly significant (paired t-test p << 1e-8).
- Robustness: Balanced/medium-strength PHNS sampling generalizes best across PHNS-enabled and standard test conditions.
- Ablations/claims:
  - S-MoE outperforms vanilla MoE and single-task models (exact deltas for these comparisons not provided in chunks).
  - Soft-label Multi-K (K=3) improves both immediate and multi-step accuracy versus equal-weight K-training and single-step training (exact metric deltas beyond NDCG@1 not provided).
- Other metrics (NDCG@5, Recall@5, MRR@5) and online/ business-impact numbers: Unclear from text.

### Practical takeaways for DealSeek (deals/product recommendations)
- Treat engagement, not only clicks, as a primary optimization target:
  - Map "completion rate" to e-commerce engagement proxies (e.g., time-on-page / scroll depth, add-to-cart-to-purchase fraction, session time after click, repeat purchases).
  - Compute per-item "runtime" analogs (e.g., typical product page dwell, typical time-to-purchase) if feasible, or use normalized engagement fractions.
- Hard-negative construction (PHNS analog):
  - Use user-specific partial-interaction events as hard negatives (e.g., users who viewed product page but bounced, abandoned carts, short product video watches).
  - Include global trending and tail negative pools to avoid overfitting to user-specific quirks.
  - Start with a low "completion" threshold equivalent (e.g., 0.05) to identify partial interactions as hard negatives; tune sampler balance with validation.
- Sequence modeling and experts:
  - Adopt an MoE-like approach with experts specialized by context window (short-term session signals vs. mid-term behavior vs. long-term preferences).
  - Use a gating network with entropy regularization to avoid collapse; aim for a target entropy around 0.5 (≈0.52) as a starting point.
- Multi-task and weighted losses:
  - Jointly train CTR/next-item ranking with engagement/regression objectives; weigh losses by per-user engagement to prioritize high-quality signals.
  - Keep an engagement-regression head during training and consider removing it at inference to reduce serving cost (as done in the paper).
- Next-K soft labels:
  - Train for next-K (K≈3) with a soft-label schedule (1.0, 0.6, 0.3) to improve immediate accuracy while learning longer horizons.
- Evaluation and offline setup:
  - Use harder negatives in validation/test (PHNS-style) to better predict real-world behavior; evaluate NDCG@1 and test robustness to negative pools.
- Operational considerations:
  - Expect increased model complexity and compute for multiple transformers and gating — plan infrastructure (training GPU memory, inference latency) and consider pruning/disabling regression head at inference.
  - Always validate with online A/B testing focusing on engagement KPIs (e.g., dwell, conversion, retention) rather than clicks alone.

### Limitations / open questions
- Required telemetry: Method assumes reliable watch-time / completion and per-item runtime; in e-commerce this maps to time-based engagement signals which may be noisy or unavailable.
- Compute and complexity: S-MoE + gating + entropy regularization increases training and inference cost; exact resource/latency overheads are Unclear from text.
- Evaluation caveats: Experiments use sampled negatives and offline ranking metrics; how offline gains translate to business metrics (revenue, conversion lift) is Unclear from text.
- Hyperparameter sensitivity: PHNS sampling ratios, low completion threshold (0.05), target entropy (≈0.52), and soft-label schedule require careful tuning; wrong settings can degrade performance.
- Generalization: Paper assumes per-item runtimes and quality telemetry; whether gains transfer across domains with different user behavior patterns (e.g., short impulse purchases vs. long-consideration products) is Unclear from text.
- Regression head usage: Training uses a completion regression head that is discarded at inference — the trade-off between training benefits and added training complexity/overhead needs empirical validation per use case.
- Online deployment details and long-term effects (user satisfaction over weeks/months): Unclear from text.

## Beyond Top-1: Addressing Inconsistencies in Evaluating Counterfactual Explanations for Recommender Systems

- **Source PDF**: `papers/Beyond Top-1 Addressing Inconsistencies in Evaluating Counterfactual Explanations for Recommender Systems.pdf`
- **DOI**: `https://doi.org/10.1145/3705328.3748028`

### TL;DR (3 bullets)
- Evaluating counterfactual explanations (CEs) only on the top-1 recommendation is unstable: CE method rankings vary substantially with recommender quality.  
- Use list-wise, perturbation-based metrics POS-P@T and NEG-P@T computed over top-k (k=1..5). Expanding k (often k≥3, sometimes k=5) yields far more consistent CE rankings across recommender checkpoints.  
- NEG-P@T is more robust to recommender quality (often stabilizes at k=2); POS-P@T is more sensitive and needs larger k to stabilize. Evaluate explainers across multiple recommender performance checkpoints (25/50/75/100% training).

### Problem
Current practice evaluates counterfactual explanations for recommenders by measuring effects only on the top-1 recommended item and using single recommender snapshots. That yields inconsistent and misleading comparisons of CE methods because rankings depend heavily on the underlying recommender’s quality and on whether list-wise effects are considered.

### Approach
- Introduce two perturbation-based, list-wise metrics computed over the top-k recommendation list:
  - POS-P@T: remove increasingly important user-history items (according to the explainer) and measure how quickly the target items drop beyond rank threshold T.
  - NEG-P@T: remove least-important history items and measure how robust target items are to staying within rank threshold T.
- Evaluate these metrics for thresholds T ∈ {5, 10, 20} and k ∈ {1..5} (i.e., list-wise top-k, k up to 5).
- Experimental design: 3 datasets (MovieLens-1M, Yahoo! Music, Pinterest), 2 recommenders (Matrix Factorization (MF), VAE), 6 CE methods (Jaccard, Cosine, LIME-RS, SHAP, ACCENT, LXR), and four recommender checkpoints (25%, 50%, 75%, 100% of training). Repeated experiments; code released.

### Results (include numbers)
- Datasets: ML-1M, Yahoo! Music, Pinterest. Recommenders: MF and VAE. CE methods: 6 named above. Checkpoints: 25/50/75/100% training.
- Top-1 evaluations show large fluctuations in CE method rankings as recommender training level changes (no single stable ordering).
- Moving to top-k:
  - Generally, k ≥ 3 yields substantially more consistent CE rankings across recommender checkpoints.
  - For many settings, k = 5 further improves stability.
  - MF (matrix factorization) is relatively stable; small k suffices to stabilize rankings.
  - VAE requires larger k to reach similar stability.
- Metric-specific:
  - NEG-P@T stabilizes earlier (often at k = 2), i.e., rankings under NEG are less sensitive to recommender quality.
  - POS-P@T is highly sensitive to recommender quality and often needs k ≥ 3 (sometimes k = 5) for stable rankings.
- Thresholds used for rank checks: T ∈ {5, 10, 20}.

### Practical takeaways for DealSeek (deals/product recommendations)
- Don’t judge explainers using only top-1: measure effects over the top-k list (run k = 1..5) to avoid misleading conclusions about explanation quality.
- Use NEG-P@T for a stable, robust ordering of explainers (expect stability often by k = 2). Use POS-P@T if you specifically want sensitivity to what breaks a target’s rank (but expect more dependence on recommender quality).
- Evaluate explainers across multiple recommender checkpoints (e.g., 25/50/75/100% of training or production A/B model versions) because CE rankings change with recommender performance.
- Tune k and the metric per model family:
  - For simpler CF models like MF, small k (2–3) may be sufficient.
  - For more expressive models (e.g., VAE-like architectures used in DealSeek if applicable), evaluate up to k = 5.
- Use rank thresholds T relevant to your UX: the paper tested T ∈ {5, 10, 20} — for DealSeek, choose T that maps to visible positions in your UI (e.g., T = 5 if only the top 5 deals are prominent).
- Operational checklist: evaluate 6+ CE candidates, compute NEG-P@T and POS-P@T over k ∈ {1..5} and T ∈ {5,10,20}, compare rankings across at least 3 recommender checkpoints, prefer methods that are consistently strong under NEG-P@T.

### Limitations / open questions
- Scope limitations: study focuses on implicit-feedback collaborative filtering and perturbations by removing user-history items.
- Architectural scope: only two recommenders tested (MF and VAE). Results may not generalize to session-based recommenders, content-aware models, or other modern architectures.
- k range: experiments limited to k ≤ 5. Behavior for larger k is Unclear from text.
- Perturbation schemes: only history-item removal was considered; other perturbations (e.g., feature-level, counterfactual content changes) were not tested.
- Explanation paradigms: only certain CE families (6 methods) were evaluated; other explanation approaches may behave differently.
- Composite evaluation: authors suggest—but did not provide—a composite metric that jointly considers recommender performance and explanation quality.
- Reproducibility: code released, but generalization to production DealSeek datasets and business metrics (CTR, conversion, revenue) is Unclear from text.

## Coarse-to-fine Dynamic Uplift Modeling for Real-time Video Recommendation

- **Source PDF**: `papers/Coarse-to-fine Dynamic Uplift Modeling for Real-time Video Recommendation.pdf`

### TL;DR (3 bullets)
- Problem framed as uplift estimation for video-duration interventions: who benefits from which exposure-duration bucket, but needs to handle multi-treatment choices and fast request-level interest shifts.  
- Method (CDUM): two-stage coarse-to-fine system — CPM (offline, per-user long-term multi-treatment uplift predictions) + FIC (online, per-request candidate-level real-time refinement) — with a simple decision rule that enables a treatment when real-time interest exceeds a threshold.  
- In production A/B on Kuaishou (18 days, 20% treatment vs 20% baseline) CDUM produced small but statistically significant uplifts (Enter LT7 +0.048%, Slide LT7 +0.041%); removing the online FIC substantially reduced gains.

### Problem
- Many pipeline modules (e.g., exposure-adjustment by duration) are non-personalized and can harm some users while helping others.  
- Goal: estimate individualized uplift (which users benefit from which treatment) in a multi-treatment setting (K duration buckets) and make real-time per-request decisions to match fast-changing interests.  
- Practical constraints: treatments are actionable interventions in the serving pipeline (here: exposure duration buckets); need low-latency online decisions and instrumentation to produce per-request candidate exposure logs.

### Approach
- High-level design: Coarse-to-Fine Dynamic Uplift Modeling (CDUM) with two stages:
  1. CPM (Coarse-grained Preference Modeling) — offline, daily-level features:
     - Predicts long-term expected outcome ˆy_k for every treatment k (multi-treatment model).
     - Architecture highlights: encodes user and treatment features; decomposes treatment embedding into a guidance component (filters user features) and an indicator component (helps generalization); uses a per-treatment mixture-of-experts + gated towers; trained with Huber loss.
  2. FIC (Fine-grained Interest Capture) — online, per-request:
     - Encodes recent request-level candidate features and computes per-treatment real-time interest scores ˆr_k using an MMOE-like multi-task model.
     - Labels for FIC are request-level ratios (analogue: long-play / short-play within candidate exposures).
     - Decision rule: enable a treatment at a request when ˆr_k > ξ (threshold), combining offline ˆy_k and online ˆr_k to make dynamic uplift decisions.
- Training/validation: uses randomized logs / RCTs in industrial data to support unbiased uplift learning; ablations show both guidance and indicator embeddings matter.

### Results (include numbers)
- Offline: CPM outperforms a range of uplift baselines (meta-learners, tree-based, neural uplift models) on CRITEO, LAZADA public benchmarks and an industrial Kuaishou multi-treatment dataset. Exact numeric gains on those benchmarks: Unclear from text.
- Production online A/B (Kuaishou):
  - Experiment length: 18 days; traffic split: 20% treatment vs 20% baseline.
  - Reported statistically significant absolute improvements (small magnitudes):
    - Enter LT7: +0.048% (absolute)
    - Slide LT7: +0.041% (absolute)
  - Also improved app usage / watch-time metrics (exact numbers: Unclear from text).
  - Ablation: removing the FIC real-time module materially reduced online gains, indicating the importance of request-level adjustment.
- Ablation offline: both guidance and indicator parts of the treatment embedding contributed to performance gains (numeric contribution: Unclear from text).

### Practical takeaways for DealSeek (deals/product recommendations)
- Map the intervention: treat actionable pipeline knobs as treatments (e.g., exposure prominence/duration, placement frequency, discount highlight level, price rounding). Define K discrete treatment buckets that reflect business trade-offs (e.g., conversion vs engagement).
- Two-stage deployment pattern:
  - Build an offline CPM to learn long-term per-user expected outcomes for each treatment (e.g., retention, purchase rate, LTV). Use a multi-treatment architecture (treatment-aware embeddings, MoE per treatment) so you can score all K options cheaply for each user.
  - Run an online FIC that uses request/session/candidate signals (immediate intent, recent clicks, scroll depth, candidate attributes) to adjust offline scores and decide whether to enable a treatment at this request. Only enable when real-time interest ˆr_k exceeds a tuned threshold ξ.
- Labels and instrumentation:
  - Log per-request candidate exposures and short/long interactions (or analogous signals for deals: immediate click vs sustained engagement/purchase) to construct per-request labels (ratios analogous to long-play/short-play).
  - Prefer RCT/randomized treatments during data collection to get less biased uplift estimates.
- Operational points:
  - Precompute CPM ˆy_k per user and cache; serve FIC in low-latency path to compute ˆr_k on the candidate set and apply ξ rule.
  - Tune ξ on holdout/RCT data to trade off precision of enabling treatments vs scale.
  - Expect small absolute improvements on high-scale traffic; even small % absolute lifts (e.g., ~0.04–0.05% absolute in retention-related metrics) were deemed meaningful in production.
- If translating to deals: a likely profitable treatment is increasing visibility for certain deals (longer prominence/duration) for users predicted to have high real-time purchase intent for that deal; avoid global non-personalized boosts.

### Limitations / open questions
- Specialized evaluation: method is demonstrated for duration-based exposure adjustments; generalization to other treatment types (e.g., ranking-level interventions, different modalities) is suggested but not demonstrated.
- Data requirements: relies on per-request candidate exposure logs and per-request labels (instrumentation required). Training used RCT logs in the industrial setting — without randomization, unbiased uplift estimates need careful bias correction.  
- Counterfactuals: as with uplift methods generally, counterfactual outcomes are unobserved and validity depends on the quality of randomization / deconfounding.
- Numbers and details missing: exact offline numeric improvements and many online metric values beyond the two reported absolute lifts are Unclear from text. Also: model sizes, latency figures, and compute/serving cost are not specified.  
- Engineering constraints: needing low-latency FIC inference, threshold ξ tuning, and maintaining CPM cache freshness are operational burdens.  
- Open questions: how well the approach adapts to treatments with continuous parameters (not discrete buckets), to interventions earlier in the pipeline (ranking), and to settings with weaker or no randomized data.

## Exploring Scaling Laws of CTR Model for Online Performance Improvement

- **Source PDF**: `papers/Exploring Scaling Laws of CTR Model for Online Performance Improvement.pdf`
- **DOI**: `https://doi.org/10.1145/3705328.3748046`

### TL;DR (3 bullets)
- SUAN: a transformer-style Stacked Unified Attention Network models very long user sequences (target-aware input + concatenated behavior features) and shows predictable power-law scaling of AUC with model grade (size + sequence length) and data size.
- LightSUAN: a deployment-aware, sparsely-attentive variant with parallel inference is distilled from a high-grade SUAN to meet latency constraints while preserving/improving online CTR.
- Production impact: SUAN gave +2.67% CTR (+1.63% CPM) at 48 ms latency (was 33 ms baseline); distilled LightSUAN gave +2.81% CTR (+1.69% CPM) with ~43 ms average inference time.

### Problem
- Click-through-rate (CTR) prediction needs maximal accuracy from very long user behavior histories, but production systems have strict latency/memory constraints.
- Offline AUC gains are incremental and unpredictable; authors ask whether "scaling laws" (predictable accuracy vs model/data scale, as in LLMs) hold for CTR models and whether those gains can be transferred into deployable, low-latency models.

### Approach
- Model family: SUAN (Stacked Unified Attention Network)
  - Built from repeated Unified Attention Blocks (UABs): RMSNorm + self-attention (learned relative time/position bias), cross-attention (profile → sequence), and dual-alignment attention gating to fuse sequential and non-sequential features; SwiGLU FFN.
  - Input encoding: target-aware sequence (candidate appended into behavior sequence) and concatenated behavior features to mitigate ID sparsity and focus encoder on candidate-relevant behaviors.
- Deployment-aware variant: LightSUAN
  - Sparse self-attention (local window + dilated patterns) to reduce attention complexity.
  - Parallel-inference strategy: reuse cached behavior representations across batches of candidates (parameter m2 controls parallelism).
- Knowledge transfer / training:
  - Train a high-grade SUAN (teacher) to exploit scaling improvements.
  - Online distillation: train LightSUAN (student) with binary cross-entropy on labels plus softened logits from teacher (temperature t) with λ scaled to compensate gradients from temperature.
- Engineering/ops considerations:
  - Normalization and activation choices (RMSNorm, pre-norm, SwiGLU) are critical for stable scaling behavior.
  - Sparse attention hyperparameters (window k, dilation rate r) and parallel-batch size (m2) trade off compute vs accuracy.

### Results (include numbers)
- Online production metrics:
  - SUAN: +2.67% CTR, +1.63% CPM; latency increased from 33 ms → 48 ms.
  - Distilled LightSUAN: +2.81% CTR, +1.69% CPM; average inference time ~43 ms.
- Offline & scaling:
  - SUAN shows strong offline AUC gains vs eight baselines on 2 public + 1 industrial dataset — numeric AUC deltas Unclear from text.
  - AUC follows power-law scaling with model grade and data size; fits observed across sequence lengths up to 1000 and sample counts spanning ~3 orders of magnitude with high R^2 (numeric R^2 Unclear from text).
  - Distilled LightSUAN empirically outperforms a SUAN of one grade higher (magnitude of improvement Unclear from text).
- Training regime: experiments often report single-epoch training; training cost and teacher compute are significant but exact costs Unclear from text.

### Practical takeaways for DealSeek (deals/product recommendations)
- Architecture / features
  - Use target-aware sequences (append candidate to user behavior timeline) and concatenate behavior-level features to reduce ID-sparsity issues and focus the encoder on relevant behaviors.
  - Adopt attention + cross-attention blocks (profile→sequence) with gating to merge sequential and non-sequential signals.
  - Use stable normalization and activations (RMSNorm/pre-norm, SwiGLU) — these materially affect scalability and training stability.
- Scaling & modeling strategy
  - Consider a teacher-student workflow: train a high-grade offline model to exploit scaling laws, then distill to a deployment model to meet latency constraints.
  - Expect predictable returns from increasing model grade and data size (power-law), but validate on your data scale — results rely on large logged histories.
- Deployment / latency engineering
  - Implement sparse attention (local + dilated windows) to reduce compute on long sequences and tune window/dilation to balance accuracy vs cost.
  - Cache and reuse behavior encodings across candidate batches (parallel-inference / m2) to amortize sequence encoding cost for multi-candidate ranking.
  - Benchmark end-to-end latency: SUAN→ LightSUAN examples show you can gain ~+2.7–2.8% CTR while keeping inference in the ~40–50 ms range.
- Retrieval/sampling
  - If indexing/retrieval infrastructure exists, consider selective retrieval or smart sampling of long-term behaviors to reduce sequence length while keeping signal (works cited show matching full-sequence performance with much less cost).
- Operational prerequisites
  - Ensure sufficient logged data, low-latency retrieval/indexing, and capacity for teacher-model training (teacher models are expensive to train and not directly deployable).

### Limitations / open questions
- Data-dependence and generalization:
  - Scaling-law observations stem from large industrial datasets; applicability to smaller datasets or different domains is unclear — Unclear from text.
- Training cost and teacher feasibility:
  - High-grade SUANs are computationally expensive to train; total training cost and resource requirements are not quantified — Unclear from text.
- Sensitivity to architectural choices:
  - Normalization and SwiGLU were necessary for observed scaling; replacing them broke scaling behavior — sensitivity to these choices and alternatives needs more study.
- Sparse-attention hyperparameters and trade-offs:
  - Window size (k), dilation rate (r), and parallel batch size (m2) affect accuracy vs latency; optimal settings depend on system constraints and require empirical tuning.
- Metrics and evaluation:
  - Offline results focus on AUC and single-epoch training; broader evaluation (e.g., calibration, long-term business metrics, multi-epoch regimes) is limited — Unclear from text.
- Freshness vs long-term signal:
  - Balancing recency with long-term historical signals, and the impact of retrieval freshness, remain open engineering challenges.
- Reproducibility:
  - Many numeric specifics (exact AUC deltas vs baselines, R^2 values, distillation hyperparameters, training FLOPs) are not provided — Unclear from text.

## LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders

- **Source PDF**: `papers/LONGER Scaling Up Long Sequence Modeling in Industrial Recommenders.pdf`
- **DOI**: `https://doi.org/10.1145/3705328.3748065`

### TL;DR (3 bullets)
- LONGER enables end-to-end modeling of ultra-long user behavior sequences (≫1k) by augmenting inputs with a few global tokens and using a hybrid attention pipeline (cross-causal first layer + stacked self-causal layers).
- TokenMerge (adjacent-token grouping) ± InnerTrans gives large quadratic cost savings (example: for L=2048, d=32, K=4 FLOPs drop ≈42.8% from ~587M to ~336M) while keeping or slightly improving accuracy; sampling recent k≈100 tokens hits a practical sweet spot (~54% FLOPs vs full) with near-full performance.
- On a 5.2B-sample Douyin ads corpus LONGER: AUC 0.85290, LogLoss 0.47103 (relative +1.57% AUC / −3.39% LogLoss vs base); TokenMerge+InnerTrans improves AUC to 0.85332. Production A/B gains include up to +2.10% ADSS / +2.15% ADVV and large e‑commerce lifts (Order/U +7.92%, GMV/U +6.54%).

### Problem
Industrial recommenders need to ingest and reason over ultra-long user behavior sequences (much longer than 1k tokens) end-to-end. Prior two-stage pipelines (retrieval → downstream UE) create upstream–downstream inconsistency and inefficiency. The goal is to scale transformer-style sequence modeling to very long histories in a production recommender stack while keeping training and serving feasible.

### Approach
- Input design: prepend a small set of global tokens (target token, user ID token, CLS) to the sequence so candidate-level and user-level context are separated and fused.
- Hybrid attention pipeline:
  - First layer: cross-causal attention where queries = global tokens + sampled recent tokens; this fuses global context into the sequence early.
  - Followed by stacked self-causal attention layers to model temporal dependencies across the entire (possibly compressed) sequence.
- TokenMerge compression:
  - Spatially group adjacent tokens to reduce sequence length for attention (TokenMerge).
  - Optionally run a lightweight InnerTrans inside each group to preserve intra-group interactions.
  - This reduces quadratic attention cost substantially at small parameter cost increase.
- Sampling strategy: keep a recent-window of tokens (empirically k≈100) plus global tokens to trade compute vs performance.
- Engineering/serving:
  - Fully synchronous GPU training with unified placement of dense+sparse params across HBM/CPU/SSD tiers.
  - Mixed precision and activation recomputation for memory efficiency.
  - KV-cache serving: precompute and cache sequence K/V so per-candidate attention avoids recomputing long-sequence K/V.

### Results (include numbers)
- Dataset: 5.2B-sample Douyin ads dataset.
- Offline metrics:
  - LONGER AUC = 0.85290; LogLoss = 0.47103.
  - Relative improvement vs base: +1.57% AUC, −3.39% LogLoss.
  - TokenMerge + InnerTrans AUC = 0.85332 (further improvement).
- Complexity example (TokenMerge):
  - For L = 2048, d = 32, K = 4: FLOPs reduced ≈42.8% from ~587M to ~336M.
- Practical sampling tradeoff:
  - Sampling recent k = 100 gives about 54% of full-model FLOPs while retaining near-full performance.
- Online production / A/B:
  - Ads: up to +2.10% ADSS and +2.15% ADVV.
  - Live streaming / e‑commerce: Order per user +7.92%, GMV per user +6.54%.
- Scaling laws:
  - Model performance improves with longer sequences, more FLOPs, and larger parameter width following a power-law with diminishing returns for deeper models.

### Practical takeaways for DealSeek (deals/product recommendations)
- Architecture adoption:
  - Use a few global tokens (e.g., target, user-ID, CLS) to let candidate-level features interact with long user histories without leaking future info.
  - Implement the hybrid attention flow: cross-causal layer (global tokens + recent-window) → stacked self-causal layers.
- Sequence compression and sampling:
  - Apply TokenMerge (group adjacent interactions) to cut attention cost. Consider adding a lightweight InnerTrans inside groups to retain intra-group detail.
  - Use a recent-window sampler (start with k ≈ 100) — good tradeoff between compute and recommendation quality.
- Serving & latency:
  - Use KV-cache for user histories: precompute K/V for the long user sequence so per-candidate scoring only needs the small candidate interaction part.
  - Plan for mixed-precision and activation recomputation to reduce memory pressure.
- Infrastructure:
  - Expect nontrivial engineering: HBM/CPU/SSD parameter tiers and synchronous multi‑GPU training are part of the demonstrated system. Budget for the required cluster & storage architecture.
- Metrics to monitor:
  - Offline: AUC and LogLoss improvements; measure relative gains vs your baseline.
  - Online: track revenue/engagement analogues (e.g., ADSS/ADVV equivalents, conversion, GMV per user) to validate business impact.
- Experimentation suggestions:
  - A/B test TokenMerge vs no-merge and with/without InnerTrans to find the right group size and inner-transformer depth for deals data.
  - Tune recent-window size — start at 100 and probe larger windows until marginal gains diminish.

### Limitations / open questions
- Hardware dependency: model and reported gains assume large GPU clusters and an engineered HBM/CPU/SSD parameter hierarchy. General feasibility on smaller infra is unclear from text.
- Generalization: reported results are verified on ByteDance production (Douyin) — transfer to DealSeek’s domain and user behavior may require re‑tuning; generalization to other product verticals is uncertain.
- Cost vs latency tradeoffs: exact inference latency, throughput, and cost-per-query numbers are not reported. Unclear from text.
- TokenMerge/design hyperparameters: best choices for group size, K in TokenMerge, and InnerTrans configuration depend on data; guidance beyond the L=2048 example is limited.
- Privacy / data constraints: assumptions about tokenization and use of UID/global tokens may interact with privacy/compliance requirements; Unclear from text how these are handled.
- Cold-start and long-tail: impact on cold users or rare-item modeling is not described. Unclear from text.
- Causal masking assumptions: KV-cache and causal separation assume strict candidate/global token separation to avoid leakage; operationalizing this safely may require careful engineering.
- Training data scale: while 5.2B samples were used for reported results, the minimum data scale needed to realize benefits is Unclear from text.

## Modeling Long-term User Behaviors with Diffusion-driven Multi-interest Network for CTR Prediction

- **Source PDF**: `papers/Modeling Long-term User Behaviors with Diffusion-driven Multi-interest Network for CTR Prediction.pdf`
- **DOI**: `https://doi.org/10.1145/3705328.3748045`

### TL;DR (3 bullets)
- DiffuMIN: a two-stage CTR pipeline that disentangles long-term user histories into orthogonal multi-interest channels (OMIE), learns/generates augmented interests with a conditional diffusion model (DMIG), and aligns them with contrastive calibration (CMIC).  
- Achieves state-of-the-art offline results on two public + one industrial dataset and produced +1.52% CTR and +1.10% CPM in a 7-day online A/B test, while increasing inference latency only from 33ms → 35ms.  
- Key strengths: better multi-interest diversity, generative augmentation of latent interests, and modular gains from OMIE/DMIG/CMIC; key costs: extra training complexity, hyperparameter sensitivity, and modest inference overhead.

### Problem
Long-term user behavior sequences contain rich but noisy and coupled signals. Existing two-stage CTR pipelines typically filter/encode behaviors from a single perspective, producing redundant or limited interest representations and leaving much of the latent interest space unexplored. The challenge is to extract diverse, target-aware interests from very long behavior histories and to enrich/regularize those interests so the downstream CTR model better captures long-term intent.

### Approach
- Two-stage pipeline (DiffuMIN):
  - OMIE (Orthogonal Multi-Interest Extractor): project the target embedding into c orthogonal interest channels; compute behavior→channel relevance; route each behavior to its top channel (top-1) and keep the top-p% behaviors per channel; aggregate (mean pooling) per channel to form c target-aware interest vectors. This reduces inter-channel redundancy and promotes diverse, target-oriented perspectives.
  - DMIG (Diffusion Multi-Interest Generator): a conditional diffusion model (Transformer backbone) that, conditioned on contextual interests and a given channel, generates augmented interest vectors. Sampling begins from perturbed aggregated interests (not pure Gaussian) and uses a small number of reverse steps (T' << T) to preserve personalization and limit cost.
  - CMIC (Contrastive Multi-Interest Calibrator): a contrastive loss that treats a user's augmented interest as a positive for their aggregated interest and other users' augmented interests as negatives; trained jointly (without disturbing primary CTR loss) to improve representation quality.
- Efficiency & complexity: parameter overhead is modest (extra projection layers and the diffusion network). Time complexity approximated as O(b · l · c · d) assuming c << l and small T'. Training includes diffusion optimization and sampling; inference only runs the short sampling phase.

### Results (include numbers)
- Offline: DiffuMIN reported state-of-the-art performance on two public datasets and one industrial dataset. (Exact offline metric numbers and dataset names: Unclear from text.)
- Online (7-day A/B test, replacing a baseline CTR model):
  - CTR: +1.52%
  - CPM: +1.10%
  - Inference latency: increased from 33 ms → 35 ms (≈ +2 ms)
- Ablations: OMIE, DMIG, and CMIC each contribute positively to overall performance; replacing the diffusion component with a VAE performs worse (diffusion outperforms VAE). (Exact ablation numbers: Unclear from text.)
- Data scale: industrial experiments used behavior lengths up to 5,000.

### Practical takeaways for DealSeek (deals/product recommendations)
- Leverage long histories: If DealSeek has long user histories (hundreds to thousands of interactions), DiffuMIN-style multi-interest extraction can better capture diverse shopping intents and improve CTR/monetization metrics.
- Orthogonal multi-interest channels (OMIE):
  - Choose channel count c to match expected interest diversity; start small (e.g., 4–8) and validate.
  - Use top-1 routing + top-p% per-channel filtering to reduce noise and redundancy; tune top-p to balance coverage vs. noise.
- Generative augmentation (DMIG):
  - Use conditional diffusion to enrich sparse/underrepresented interest channels; keep reverse steps T' small to limit inference cost and preserve personalization.
  - Diffusion augmentation tends to outperform a VAE alternative for generating plausible latent interests.
- Contrastive calibration (CMIC):
  - Use contrastive loss with in-batch negatives to align augmented and aggregated interests; ensure sufficiently large batch sizes or other negative sampling strategies.
- Production considerations:
  - Expect modest inference overhead (the paper observed ≈ +2ms). Monitor per-request latency and throughput.
  - Training is more complex and computationally heavier due to diffusion optimization; budget GPU/compute accordingly.
  - Hyperparameter sensitivity matters (c, top-p, T', contrastive weight, temperature τ); plan systematic tuning and sanity checks.
- Practical recipe to try at DealSeek:
  1. Implement OMIE on existing long-sequence encoder, start with c = 4–8 and top-p ≈ 20–50% (tune).
  2. Train a lightweight conditional diffusion generator (Transformer backbone) and run short sampling (T' small) for augmentation.
  3. Add CMIC contrastive loss during training with in-batch negatives.
  4. Monitor online CTR/CPM and latency; rollback if latency budget exceeded.

### Limitations / open questions
- Hyperparameter sensitivity: performance depends on choices for c, top-p, T', contrastive temperature τ, and loss weights λs. Optimal settings are dataset-dependent.
- Potential information loss: top-1 routing and top-p% filtering can drop useful behaviors; trade-offs between noise reduction and information loss need empirical tuning.
- Diffusion costs and complexity: diffusion training increases training complexity and compute; sampling adds modest inference latency—exact T' and compute requirements are Unclear from text.
- Contrastive requirements: CMIC needs good negatives (large batch sizes or effective negative sampling); stability and effectiveness depend on batch composition.
- Generalization: reported gains come from two public + one industrial dataset; how well the method transfers to different domains, user behavior distributions, or sparser catalogs is Unclear from text.
- Missing specifics: exact offline metrics, dataset names, detailed ablation numbers, and chosen hyperparameter values (e.g., c, top-p, T') are Unclear from text.
- Long-term maintenance: diffusion/generative modules may require careful monitoring for distribution shift and retraining cadence in production.

## Non-parametric Graph Convolution for Re-ranking in Recommendation Systems

- **Source PDF**: `papers/Non-parametric Graph Convolution for Re-ranking in Recommendation Systems.pdf`
- **DOI**: `https://doi.org/10.1145/3705328.3748058`

### TL;DR (3 bullets)
- Introduces a plug-and-play, non-parametric graph-convolution re-ranking that injects user-user and item-item graph signals only at inference, avoiding expensive GNN training and distributed neighbor-fetch costs.  
- Across four public benchmarks it improves ranking (Recall/NDCG) by roughly +8.1% on average while adding very little runtime cost (~0.5% avg); naive graph-encoder alternatives blow up training/test costs (hundreds of percent).  
- Method requires precomputed top-nk neighbor lists and issues nk^2 model queries per candidate at inference (so engineer caching/sharding); works best where training-time CF signals are sparse.

### Problem
Graph-structured collaborative signals (user-user and item-item affinities from interaction data) help retrieval but are rarely used in the ranking/CTR stage because standard GNN-style message passing during training is prohibitively expensive for industrial systems: neighbour lookups across distributed ID embeddings and quadratic growth with batch/neighborhood size make training and serving impractical.

### Approach
- Build normalized similarity matrices from the interaction matrix: Ĥ_user = M M^T and Ĥ_item = M^T M, degree-normalized.
- Precompute top-nk similar users and items for each user/item (one-off or infrequently).
- At inference, for a given (user, candidate item) base set:
  - Retrieve top-nk neighbors for the user and for the item.
  - Form nk^2 cross user-item pairs (pairing user-neighbors × item-neighbors).
  - Query the existing pretrained ranking model for each constructed pair (the method is model-agnostic and requires the ranker to accept arbitrary user,item inputs).
  - Aggregate those predicted scores into a final re-ranked score using similarity-weighted fusion and an additional boost for the single most similar neighbor pair (heuristic normalization + max-weight adjustment).
- No extra trainable graph parameters and no message passing during training; all graph operations are precomputation and/or inference-time.

### Results (include numbers)
- Benchmarks: Yelp2018, Amazon-Books, ML-1M, Anime.
- Average ranking gains (Recall / NDCG improvements reported): ≈ +8.1% across the four datasets.
- Average additional runtime computational overhead: ≈ +0.5%.
- For one model family (DCN) the overall extra time was reported as <2% compared to the naive graph-trained model baseline.
- Naive graph-encoder substitution (training-time graph encoders) increases compute massively: average ≈ +480% training time, ≈ +605% test time; worst-cases >1000% increase.
- Inference cost scales roughly as nk^2 model queries per candidate; top-nk lists are small in practice and can be precomputed to control cost.
- Gains are larger on sparse datasets (where training embeddings saw fewer CF signals).

Where dataset- or metric-specific numeric breakdowns (per-dataset Recall/NDCG numbers, latency in ms, memory footprint) are required: Unclear from text.

### Practical takeaways for DealSeek (deals/product recommendations)
- Low-risk integration: you can keep your current pretrained ranking model and inject graph signals at inference without re-training the model or adding graph parameters.
- Implementation recipe:
  - Precompute and store top-nk item-item and user-user similarity lists from interaction logs (one-off or periodic).
  - At ranking time for a candidate set, generate neighbor cross-pairs and score them with the existing ranker, then aggregate with similarity-weighted fusion + optional max-neighbor boost.
  - Tune nk: small nk (e.g., 2–5) for dense catalogs or heavy-interaction users/coupled items; larger nk for sparse/new-item scenarios. (nk controls tradeoff between score quality and nk^2 query cost.)
- Engineering considerations:
  - Cache top-nk lists and candidate-level nk^2 score results where possible to avoid repeated scoring; shard neighbor lists across machines if memory is an issue.
  - Aim for the reported runtime overhead budget (~0.5–2%) as a target; measure end-to-end latency impact (Unclear from text what ms budget corresponds to those percentages).
  - A/B test on recall/NDCG and business KPIs; expect larger lifts for categories with sparse interaction histories (long-tail deals).
  - Monitor and validate the heuristic aggregation (weights and boost); include fallback to base ranker if aggregate candidates exceed latency/compute budgets.
- Privacy/operational notes: method requires access to the interaction matrix at inference and to query the ranker for arbitrary pairs. (Any policy or consent requirements for using user interaction data are context-dependent — Unclear from text.)

### Limitations / open questions
- Aggregation is heuristic: similarity-weighted fusion and max-boost are ad-hoc; no learned or theoretically-optimal weighting provided.
- nk selection is dataset-dependent and affects cost/benefit; no universal rule given (dense graphs prefer small nk, sparse prefer larger nk).
- Inference cost still grows quadratically with nk (nk^2 model queries per candidate); for very large candidate sets this can be non-negligible unless engineering mitigations (caching, batching, sharding) are applied.
- Experiments limited to four public datasets (Yelp2018, Amazon-Books, ML-1M, Anime); behavior at billion-scale industrial corpora is Unclear from text.
- Memory and latency footprints (in absolute terms, e.g., MBs, ms) are not reported: Unclear from text.
- Privacy/compliance implications of accessing and distributing the interaction matrix at inference are not discussed: Unclear from text.

## Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation

- **Source PDF**: `papers/Prompt-to-Slate Diffusion Models for Prompt-Conditioned Slate Generation.pdf`
- **DOI**: `https://doi.org/10.1145/3705328.3748072`

### TL;DR (3 bullets)
- Prompt-to-Slate (DMSG) generates entire ordered slates (playlists/bundles) from natural-language prompts by sampling a diffusion model over concatenated item embeddings conditioned on a text encoder.  
- Beats standard baselines offline (up to +17% NDCGSim, +12.9% MAPSim) and showed positive online A/B impacts (−13.4% duplicate recs, +6.8% stream curations) with P99 inference ≈150 ms on an NVIDIA L4.  
- Trades: more diverse, exploratory slates but small engagement drops (−5.6% minutes played, +3% skip rate); no explicit personalization and uses nearest-neighbor rounding that needs ad-hoc dedupe handling.

### Problem
Generate coherent, diverse, ordered slates (fixed-length groups of items) directly from natural-language prompts (editorial prompts, playlist seeds, bundle descriptions) without relying on per-user histories. The goal is to model joint item co-consumption structure (slate-level distribution) rather than scoring/ranking items independently.

### Approach
- Representation: each item mapped to a continuous embedding φ (fixed item encoder: Word2Vec for music, pre-trained sentence model for bundles). Prompts encoded with a text transformer τ.
- Generative model: a conditional diffusion model (transformer-based reverse process) trained to predict velocity v_t over concatenated item embeddings representing the whole slate. Cross-attention layers condition on prompt embeddings.
- Training: v-prediction loss with SNR+1 weighting for stability. Fixed item encoder used for stability (no end-to-end encoder fine-tune in reported work).
- Decoding: continuous slate samples are converted to discrete items via nearest-neighbor / rounding (non-learned). Duplicate prevention requires ad-hoc masking during mapping.
- Inference speedups: use DDIM sampling and reduce to ~50 inference steps (reported). Model uses fixed-length slates (pad/truncate).
- Key architectural choices: transformer blocks for diffusion reverse process, conditioning through cross-attention, and non-learned nearest-neighbor decoding to allow catalog updates by swapping embeddings.

### Results (include numbers)
Offline (music playlists and bundle tasks):
- Up to +17% NDCGSim and +12.9% MAPSim over the next-best baseline on curated playlists.
- Higher MAP / NDCG / BERTScore (~0.8 reported for BERTScore) across datasets vs. baselines (Popularity, Prompt2Vec, BM25, seq2seq).
- Tends to recommend less-popular items (improved exploration).

Online (two-week A/B, ~1M users):
- −13.4% reduction in duplicate recommendations.
- +6.8% uplift in stream curations.
- +10.5% uplift in likes added to "Liked Songs".
- Small engagement changes: −5.6% minutes played, +3% skip rate.
- Inference: P99 latency ≈150 ms on an NVIDIA L4 (control was 500 ms).

Datasets / scale used in experiments:
- 100K MPD playlists, 30K curated playlists, 5.4K bundles (training/eval used sampled subsets accordingly).

### Practical takeaways for DealSeek (deals/product recommendations)
- Use-case fit: Good for prompt-driven bundle generation (e.g., “weekend camping deals”, “gifts under $50 for cooks”) where joint compatibility matters (co-purchase/co-use), and where editorial or query prompts drive recommendations rather than per-user history.
- Catalog updates: Because decoding is nearest-neighbor on embeddings, updating catalog items can be handled at inference by updating item→embedding mapping without retraining the diffusion model.
- Diversity & exploration: Expect more diverse bundles and more exploration (less-popular items surfaced) — useful if DealSeek wants to surface long-tail deals or rotate inventory.
- Infrastructure / latency: Achievable P99 ≈150 ms on an NVIDIA L4 with ~50 DDIM steps; plan hardware and batching to meet SLAs. To reach faster production needs, consider further step reduction or distillation.
- Deduplication & constraints: Nearest-neighbor rounding often requires ad-hoc masking to avoid duplicate items or enforce constraints (e.g., stock, price ranges). Implement result-level masking and constraint filtering post-decoding.
- Personalization: The model as-is is non-personalized. For user-personalized deals, integrate user embeddings/filters into conditioning or combine slate generation with a downstream reranker personalized per user.
- Data/embedding choices: Fixed item encoder simplifies training; for product recommendations, choose encoders that capture product metadata (title, category, attributes). Evaluate whether to keep it fixed or fine-tune jointly (see Limitations).
- Metrics to track: slate-level quality (NDCG/MAP variants computed at slate granularity), diversity metrics, duplicates, and engagement signals (clicks, conversions, dwell/minutes, skips/returns).

### Limitations / open questions
- Personalization: No explicit per-user personalization in the reported model; integrating user signals is future work. (Unclear from text how best to fuse user embeddings.)
- Encoder training trade-offs: The paper uses fixed item encoders for stability; whether joint end-to-end training yields better slates for product data is not answered.
- Nearest-neighbor decoding: Non-learned rounding is simple but brittle — requires masking to prevent duplicates and may not handle inventory/constraint enforcement without additional logic.
- Efficiency & production scaling: Reported ≈50 DDIM steps and P99 ≈150 ms on NVIDIA L4; further step reduction, distillation, or optimized samplers may be required for stricter latency/cost targets.
- Evaluation caveats: Offline relevance metrics treat reference slates as oracle; scores are lower bounds because many valid alternatives are counted as mismatches. Also experiments used sampled subsets (100K/30K/5.4K), so behavior at much larger catalog scale is not fully characterized.
- Missing details / Unclear from text: exact model sizes (parameter counts), training compute & costs, hyperparameters for diffusion and transformer depths, how to incorporate rich structured product metadata, and how the method scales as slate length or catalog size increases.

## Test-Time Alignment with State Space Model for Tracking User Interest Shis in Sequential Recommendation

- **Source PDF**: `papers/Test-Time Alignment with State Space Model for Tracking User Interest Shifts in Sequential Recommendation.pdf`
- **DOI**: `https://doi.org/10.1145/3705328.3748060`

### TL;DR (3 bullets)
- T2ARec adapts sequential recommenders at test time to user-interest shifts by running a few self-supervised gradient steps per test batch on two alignment losses (time-interval and interest-state) before predicting.
- It uses a Structured State Space (SSM) backbone for long-range sequences and estimates per-sequence SSM step-sizes Δ to align model time dynamics with real timestamps.
- Empirically improves next-item Recall/MRR/NDCG@10 vs several baselines on ML-1M, Amazon Prime Pantry, and Zhihu-1M, at the cost of extra test-time compute (authors set M=1 test-step to restrain cost).

### Problem
Static sequential recommenders degrade when users’ interests shift between training and serving time. The paper addresses online (test-time) adaptation to temporal distribution shifts for next-item prediction given interaction timestamps, i.e., improving robustness to interest drift without labeled feedback at serving time.

### Approach
- Backbone: a Structured State Space Model (Mamba-2 style SSM) to efficiently model long interaction sequences.
- Two self-supervised alignment modules applied at test time:
  - Time-interval alignment: the model learns adaptive SSM step-sizes Δi from inputs; estimates prediction step Δ_{n+1} by re-feeding the model’s last output embedding (stop-gradient used). A hinge pairwise loss aligns predicted pairwise differences (Δi − Δj) to true timestamp differences Tij. To scale, the loss is computed block-wise to avoid full O(n^2) pairwise computation.
  - Interest-state alignment: constructs a forward predicted next-state ˆh_{n+1}, applies a learned backward update to produce reconstructed backward state ˆh^b_n, and minimizes ||h_n − ˆh^b_n||^2 normalized by Δ_{n+1}^2. The paper provides a theoretical upper bound (Theorem 4.1) linking this loss to prediction interval Δ_{n+1} and instance-dependent reconstruction errors.
- Test-time training procedure: for each test batch, run a small number M of gradient steps on the combined self-supervised loss (L_time + L_state), use the adapted model to predict next items, then restore original parameters. In experiments authors use M = 1 to limit runtime overhead.

### Results (include numbers)
- Datasets evaluated: ML-1M, Amazon Prime Pantry, Zhihu-1M.
- Metrics reported: Recall / MRR / NDCG @10 (exact metric values and percentage improvements vs baselines are Unclear from text).
- Baselines: SASRec, Mamba4Rec, TiSASRec, TiM4Rec, TTT4Rec; T2ARec is claimed to outperform these baselines on the listed datasets (exact numbers Unclear from text).
- Ablation findings: both alignment losses (time-interval and interest-state) are necessary for the full gains; improvements are larger on later test partitions (where drift is stronger).
- Throughput / compute: test-time training reduces iterations/sec relative to non-adaptive SSM baselines—authors report roughly ~50% of Mamba-like throughput in their tests; to limit cost they used M = 1 test-step.
- Loss scaling: the interest-state loss is normalized by Δ_{n+1}^2 (i.e., scales like Δ_{n+1}^{-2}), implying sensitivity to small prediction intervals.

### Practical takeaways for DealSeek (deals/product recommendations)
- Use-case fit: T2ARec is suitable if you have timestamped user interactions and want per-batch, online robustness to rapid interest shifts (flash deals / seasonal trends).
- Infrastructure and latency:
  - Expect ~2x test-time cost (authors report ~50% throughput of non-adaptive SSM). If low-latency serving is critical, restrict test-time steps (authors used M=1) or run adaptation in parallel/offline micro-batches.
  - Implement parameter snapshot/restore to keep production model unchanged between batches.
- Data requirements and preconditions:
  - Requires absolute event timestamps and a prediction timestamp for each sequence.
  - Relies on the model’s last output embedding as a proxy to estimate Δ_{n+1} (uses stop-gradient)—validate this proxy on your catalogs where items may change rapidly.
- Scalability:
  - Use block-wise pairwise losses to keep O(n)–ish scaling instead of full O(n^2) when aligning time intervals.
  - SSM backbone chosen for long sequences—beneficial if users have long interaction histories.
- Engineering knobs to tune:
  - Number of test-time steps M (cost vs. adaptivity). Start with M=1 as in the paper.
  - Block size for pairwise loss (trade global consistency vs throughput).
  - Loss weights for L_time and L_state; both matter per ablation.
- Reproducibility / tooling:
  - Check Recbole and TTT4Rec implementations referenced in the paper for baseline code patterns and test-time training recipes.
- Monitoring:
  - Track per-partition lift (later partitions may see larger gains), latency, and failure modes tied to very small Δ_{n+1}.
- Extensions:
  - Combine with exploration/feedback policies or graph-based signals if long-term metrics (retention / conversion) are primary.

### Limitations / open questions
- Exact numeric gains and variance: the chunked summaries do not provide concrete Recall/MRR/NDCG numbers or effect sizes—exact improvements are Unclear from text.
- Computational cost vs. benefit trade-off: authors report ~50% throughput relative to non-adaptive SSM; how that translates to end-to-end latency and business KPI improvement on DealSeek is dataset- and platform-dependent.
- Sensitivity to small Δ_{n+1}: interest-state loss scales with Δ_{n+1}^{-2} and depends on instance reconstruction errors—this can cause instability for very short prediction intervals or noisy embeddings.
- Heuristic proxy for Δ_{n+1}: estimating the next-step time via re-feeding the last output embedding is a heuristic (stop-gradient used); its robustness when item embeddings are poor or rapidly changing is uncertain.
- Block-wise pairwise loss: improves scalability but may sacrifice global temporal consistency—optimal block size is task-dependent.
- Hyperparameter tuning at test time: requires tuning loss weights, block size, learning rate and M for test-time updates; practical deployment needs validation protocols and safety guards.
- Theoretical bounds: Theorem 4.1 links losses to Δ_{n+1} and instance errors, but practical guidance for controlling instance-dependent terms (ε_n) is limited in the summary.
- Reproducibility: implementation details (exact SSM config, learning rates, batch sizes) and full metric tables are not present in the chunked text—consult the full paper and referenced codebases (Recbole, TTT4Rec) before productionizing.

## Unified Embedding Based Personalized Retrieval in Siddharth Subramaniyam†*

- **Source PDF**: `papers/Unified Embedding Based Personalized Retrieval in Etsy Search.pdf`
- **DOI**: `https://doi.org/10.1145/3292500.3330759`

### TL;DR (3 bullets)
- Presented UEPPR: a two-tower cosine retrieval system with a heavy, unified product encoder and a lightweight query+user encoder to meet strict latency while closing semantic gaps and personalizing short e‑commerce queries.
- Training + indexing innovations: product-side docT5query-style pretraining, multi-part hinge loss, aggressive negative mining, and ANN-based Product Boosting (numeric quality features + learned static query-side weights) to combine relevance and product quality without retraining embeddings.
- Production: Faiss HNSW + 4-bit PQ fastscan + re-ranking yields P99 ≈ 18 ms with <4% recall loss post‑rerank; offline Recall@100 up to ≈0.708 (baselines 0.33–0.60); online A/B: +5.58% organic search purchases, +2.63% site-wide conversion.

### Problem
Etsy needs a candidate retrieval system that:
- Closes vocabulary/semantic gaps for tail, ambiguous queries.
- Personalizes results for very short, low-context e‑commerce queries.
- Meets strict latency constraints for production low‑latency search.

### Approach
- Architecture
  - Two‑tower cosine similarity retrieval:
    - Product tower P(p): unified encoder that concatenates transformer text outputs, bipartite query–product graph embeddings, token/ID embeddings, location encoders and numeric features.
    - Query+user tower Q(q,u): lightweight, transformer‑free encoder to keep per‑query latency low.
- Training
  - DocT5query-style pretraining: T5-small trained to generate historical purchased queries from product text (pretraining applied to product encoder).
  - Multi-part hinge loss: learns different thresholds for interaction types instead of a single margin.
  - Negative mining: combination of hard in-batch, uniform, and dynamic hard negatives with time-varying loss weights.
- Product quality integration (ANN-based Product Boosting)
  - Augment indexed product vectors with numeric quality features.
  - Learn static, query-side feature weights (optimized by Bayesian black‑box optimization) so ANN dot-products represent relevance + quality without re-training embeddings.
- Serving/engineering
  - Faiss-based ANN (HNSW + 4-bit PQ fastscan) with re-ranking to balance recall/latency.
  - Caching strategies to preserve in-session personalization while keeping TTLs practical.

### Results (include numbers)
- Latency/recall tradeoff (production):
  - P99 latency ≈ 18 ms.
  - Recall loss after ANN + re-ranking < 4%.
- Offline retrieval quality:
  - UEPPR Recall@100 up to ≈ 0.708.
  - Baseline lexical / prior EBR methods ranged ≈ 0.33–0.60 for Recall@100.
  - Ablation: adding graph embeddings yielded the single largest uplift — ≈ 15% relative gain in Recall@100.
- Online A/B test:
  - +5.58% organic search purchase rate.
  - +2.63% site-wide conversion rate.
- Behavioral observations:
  - Personalization (query+user tower, location) benefits head/broad queries most.
  - Graph and text encoders improve tail/ambiguous query recall.
  - ANN-based boosting helped head queries more than tail.

### Practical takeaways for DealSeek (deals/product recommendations)
- Favor a heavy, unified product representation: combine text transformers, graph embeddings, ID/token embeddings, location and numeric features into a single product vector to boost recall across query types.
- Keep the online query/user encoder lightweight to meet latency targets; push transformer capacity to the product side (pretrain product encoder with query-generation objectives).
- Prioritize graph embeddings: they produced the largest single relative lift (~15% in Recall@100) and help tail/ambiguity handling — useful for matching deals to rare or imprecise queries.
- Use aggressive, varied negative mining (in-batch hard, uniform, dynamic) and a multi-part hinge loss to better separate interaction types.
- Combine embedding relevance with product-quality signals via ANN-level boosting (augment index vectors + learned static weights) to avoid full embedding re-training when optimizing for business metrics (conversion, margin).
- Engineering choices:
  - Use Faiss HNSW + PQ fastscan + re-rank to hit latency/recall targets (example: P99 ≈ 18 ms, <4% recall drop).
  - Cache user/session query vectors judiciously to preserve personalization with reasonable TTLs.
- Expected impact: based on Etsy results, improvements can materially lift purchase and conversion rates — treat ANN boosting and personalization as high-impact, incremental layers over a unified embedding retrieval backbone.

### Limitations / open questions
- Asymmetric design tradeoff: lightweight query tower limits modeling capacity on the query side; transformer gains are realized mainly via product-side pretraining (no heavy query transformer). Tradeoffs between latency and expressivity remain.
- Graph encoder stability: required freezing in training to avoid overfitting; how to continually update graph embeddings without overfitting or stale signals is an open operational challenge.
- Product Boosting limitations: learned weights are static (not per-query) and can overfit proxy metrics if learned end‑to‑end; per-query or adaptive boosting not addressed here.
- Exposure/presentation bias: training on logged interactions introduces bias; debiasing strategies are not detailed.
- Compute and convergence costs: training requires substantial multi‑GPU compute and long training runs — exact resource/cost numbers: Unclear from text.
- Generalization and long tails: ANN boosting improved head queries more than tail; effectiveness for extreme long-tail or cold-start items is not fully quantified.
- Operational tuning specifics (e.g., exact HNSW/PQ hyperparams, PQ codebook sizes, cache TTLs per segmentation) — Unclear from text.

## You Don't Bring Me Flowers: Mitigating Unwanted Recommendations Through Conformal Risk Control

- **Source PDF**: `papers/You Don't Bring Me Flowers Mitigating Unwanted Recommendations Through Conformal Risk Control.pdf`
- **DOI**: `https://doi.org/10.1145/3705328.3748054`

### TL;DR (3 bullets)
- Post-hoc, model-agnostic pipeline uses conformal risk control to choose a score threshold λ that provably bounds expected fraction of unwanted/flagged recommendations: E[risk] ≤ α (finite-sample, distribution-free under exchangeability and monotonicity).
- Instead of removing flagged items (which shrinks recommendation size), the pipeline replaces them with previously consumed, non‑flagged items that meet a safety criterion C > β to preserve utility and recommendation length while keeping the conformal guarantees.
- Works without retraining (threshold computation O(Q) for calibration; per-request cost dominated by re-ranking O(|I| log |I|)), but guarantees are in expectation, rely on calibration representativeness and a “safe repeats” assumption, and may increase repetitions/cold‑start issues.

### Problem
Recommender systems can repeatedly expose users to unwanted or harmful items (user flags, moderation labels) despite sparse negative feedback. Platforms need a practical, auditable method to limit such unwanted recommendations with provable guarantees, without retraining existing rankers or relying on ad‑hoc filters that may over‑prune or shrink recommendation sets.

### Approach
- Use conformal risk control (a generalization of conformal prediction) as a post-processing layer on any base ranker.
- From a calibration set, find a score threshold λ (per Theorem 4.1) so that selecting items with score ≥ λ yields a controlled expected risk. Theoretical guarantee requires the risk function to be non‑increasing in λ and exchangeability/representativeness of calibration.
- To avoid shrinking recommendation size when filtering, define:
  - T_λ = {items with base score ≥ λ}
  - T_safe = previously consumed, non‑flagged items with safety criterion C > β (e.g., first‑view watch‑time > β)
  - Final output = top‑k items from union(T_λ, T_safe) by base ranker score.
- This replacement strategy preserves monotonicity required for conformal guarantees, needs no retraining, can be cached/precomputed, and integrates into two‑stage recommenders.

### Results (include numbers)
- Theoretical guarantee: E[risk] ≤ α (distribution‑free, finite‑sample guarantee under stated assumptions).
- Empirical evaluation on KuaiRand dataset:
  - Method reliably enforces target reductions in unwanted content (i.e., meets the α risk targets in expectation).
  - Replace (replacement strategy) retains substantially more nDCG@20 and Recall@20 than Remove (simple filtering) for the same risk target; Remove often over‑prunes.
  - Sign‑aware models required more replacements (attributed to sparse negative graphs).
- Complexity/operational numbers:
  - Calibration/threshold computation: O(Q) where Q is calibration size.
  - Per-request cost: dominated by re‑ranking, O(|I| log |I|).
- Precise numeric gains (e.g., % nDCG retained, absolute risk reductions) — Unclear from text.

### Practical takeaways for DealSeek (deals/product recommendations)
- Plug-in safety layer: You can enforce a user‑set maximum expected rate α of unwanted deals without retraining your ranker by computing λ on a held-out calibration set and applying the replacement pipeline.
- Design replacement pool C for deals: use previously purchased/viewed non‑flagged deals with a safety signal C (examples: past purchase count, conversion rate, dwell/time‑on‑deal, repeat purchases) and set replacement threshold β to balance safety vs. pool size.
- Choose α and β with business constraints:
  - Lower α (stricter) → fewer flagged deals but more replacements and potential utility loss.
  - Higher β (stricter safety for replacements) → safer replacements but smaller replacement pool and more repeats or possibly fewer than k items.
- Calibration representativeness: ensure the held‑out calibration set reflects deployment users (watch for high‑reporting users biasing λ to be overly conservative for the majority).
- Cold‑start users: have fallback strategies (e.g., conservative defaults, content-based safe items, editorial lists) because replacement pool may be empty.
- Operational integration:
  - Compute λ offline and cache T_safe per user (or user segment) to keep per-request latency low.
  - Instrument and monitor: risk actualization, user reporting distribution, repeated-item rates, utility metrics (CTR, conversion, revenue), and distribution shift.
- Auditability/regulatory fit: this approach gives an auditable, provable expectation bound useful for safety/moderation requirements, but document assumptions (exchangeability, Property 1) when presenting guarantees.

### Limitations / open questions
- Guarantees are in expectation (E[risk] ≤ α), not high‑probability bounds; tail behavior/user‑level worst cases can still occur.
- Dependence on calibration:
  - Requires a representative, exchangeable calibration set; sensitive to distribution shift and skew (e.g., few high‑reporting users can bias λ).
  - Needs labeled indicators of “unwantedness” for calibration; availability and definition of such labels may vary across domains.
- Assumption of “safe repeats” (Property 1): requires existence of previously consumed items with near‑zero re‑report probability; only approximately true and dataset‑dependent. Cold‑start users may have no safe pool.
- Replacement side‑effects:
  - Increases repetition of previously seen items, potentially harming UX (echo chambers, stale recommendations).
  - May reduce diversity or novelty; possible negative impact on long‑term engagement not covered by finite‑sample guarantees.
- Evaluation scope: empirical validation reported on a single public dataset (KuaiRand); transferability to DealSeek’s distribution/metrics is Unclear from text and requires in‑house testing.
- Model/operational gaps:
  - Does not directly address online sequential/adversarial feedback loops, or strategic manipulation by users.
  - Conformal control may be conservative, reducing utility; tradeoffs need tuning per business metrics.
- Open engineering questions:
  - Best practices to construct C and set β for deal/product domains.
  - Handling multi‑objective constraints (e.g., safety + fairness + revenue) jointly with conformal risk control.
  - Extensions to provide per-user or high‑probability guarantees rather than only expectation.
