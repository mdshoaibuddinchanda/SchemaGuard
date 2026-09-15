# SchemaGuard Master Plan

## 1. Final research identity

| Item                         | Frozen name                                                                                |
| ---------------------------- | ------------------------------------------------------------------------------------------ |
| Repository                   | `schemaguard`                                                                              |
| Paper                        | **SchemaGuard: Certified Prediction Consistency under Lossless Tabular Schema Migrations** |
| Benchmark                    | **SchemaOrbit-14**                                                                         |
| Canonical representation     | **Schema Canonical Normal Form — SCNF**                                                    |
| Robust aggregation algorithm | **Cost-Aware Orbit Sparse Aggregation — COSA**                                             |
| Main instability metric      | **Schema Invariance Gap — SIG**                                                            |
| Main risk metric             | **Worst-View Excess Risk — WVER**                                                          |
| Classification disagreement  | **Schema Flip Rate — SFR**                                                                 |

## 2. Revised scientific core

The strongest question is:

> Do complete tabular prediction pipelines produce consistent predictions after certified, lossless schema migrations, and can a deterministic canonicalization-and-aggregation wrapper restore consistency under a bounded inference budget?

This is stronger than asking whether models are sensitive to column order or preprocessing.

The paper studies transformations such as:

* Unit conversion
* Reversible nonlinear numerical representation
* Categorical code replacement
* Categorical-to-one-hot migration
* Feature duplication
* Deterministic redundant features
* Reversible integer decomposition
* Composite schema migrations

Row permutation, column permutation, and class permutation must be treated as controls rather than claimed as new contributions. A May 2026 mechanistic study already analyzes row, column, and class-permutation invariance in tabular foundation models. [A Mechanistic Study of Tabular Foundation Models](https://arxiv.org/abs/2605.21288)

Metamorphic testing is also established literature. Do not claim that SchemaGuard invented testing through output-preserving transformations. The new claim must be the combination of:

1. Certified lossless schema migrations.
2. End-to-end tabular pipelines.
3. Current tree and foundation models.
4. A frozen benchmark.
5. A canonical normal form.
6. A compute-bounded repair method.
7. External validation using real schema pairs.

## 3. Model decision

### 3.1 Frozen model matrix

| ID          | Exact model                               |        Package |  Version | Role                                  |
| ----------- | ----------------------------------------- | -------------: | -------: | ------------------------------------- |
| `LR-1.9`    | `sklearn.linear_model.LogisticRegression` | `scikit-learn` |  `1.9.1` | Linear sanity baseline                |
| `CAT-1.2`   | `catboost.CatBoostClassifier`             |     `catboost` | `1.2.10` | Native categorical boosted-tree model |
| `XGB-3.4`   | `xgboost.XGBClassifier`                   |      `xgboost` |  `3.4.1` | Numeric/encoded boosted-tree model    |
| `TPFN3-8.5` | `tabpfn.TabPFNClassifier`                 |       `tabpfn` |  `8.5.0` | TabPFN-3 foundation model             |
| `TICL2-2.2` | `tabicl.TabICLClassifier`                 |       `tabicl` |  `2.2.0` | TabICLv2 foundation model             |

### 3.2 LightGBM, boosting, and bagging decision

* LightGBM: exclude.
* Random forest: exclude.
* ExtraTrees: exclude.
* AdaBoost: exclude.
* Generic bagging: exclude.
* Generic deep tabular network: exclude from the primary benchmark.
* CatBoost: retain.
* XGBoost: retain.

Eliminating all boosting models would weaken the study. Tabular reviewers will expect at least one strong tree-based family. CatBoost and XGBoost are retained because their treatment of categorical and encoded schemas differs materially.

LightGBM would largely duplicate the XGBoost role. Random forest and other bagging models would increase conditions without resolving a central scientific question.

The resulting model families are:

1. Linear.
2. Native categorical boosting.
3. Encoded numerical boosting.
4. TabPFN foundation model.
5. TabICL foundation model.

### 3.3 Python 3.12 compatibility

Compatibility is confirmed at the package level:

* TabICL 2.2.0 requires Python 3.10 or newer and explicitly lists Python 3.12. Its default classification checkpoint is `tabicl-classifier-v2-20260212.ckpt`. [TabICL PyPI](https://pypi.org/project/tabicl/), [official TabICL repository](https://github.com/soda-inria/tabicl)
* TabPFN 8.5.0 supports Python 3.12. The fixed checkpoint is `tabpfn-v3-classifier-v3_default.ckpt`. [TabPFN PyPI](https://pypi.org/project/tabpfn/)
* XGBoost 3.4.1 requires Python 3.12 or newer. [XGBoost PyPI](https://pypi.org/project/xgboost/)
* CatBoost 1.2.10 publishes Python 3.12 wheels. [CatBoost PyPI](https://pypi.org/project/catboost/)
* Scikit-learn 1.9.1 supports Python 3.12. [Scikit-learn PyPI](https://pypi.org/project/scikit-learn/)

TabICL’s Python compatibility is confirmed. Its compatibility with 4 GB VRAM is not guaranteed by its documentation. The implementation must use:

```python
TabICLClassifier(
    checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
    n_estimators=8,
    batch_size=1,
    kv_cache=False,
    offload_mode="auto",
    use_amp="auto",
    device="cuda",
    random_state=seed,
)
```

If CUDA fails:

```python
device = "cpu"
offload_mode = "auto"
n_jobs = 2
```

TabICL supports KV caching, but its maintainers state that it consumes additional memory. Therefore `kv_cache=False` is mandatory on the 4 GB laptop during the primary run. KV caching is tested only as a resource ablation. [TabICL repository](https://github.com/soda-inria/tabicl)

### 3.4 Exact model configurations

```yaml
models:
  LR-1.9:
    class: sklearn.linear_model.LogisticRegression
    package_version: "1.9.1"
    preprocessing: common_onehot_standard
    parameters:
      penalty: l2
      C: 1.0
      solver: lbfgs
      max_iter: 2000
      tol: 1.0e-6
      class_weight: null

  CAT-1.2:
    class: catboost.CatBoostClassifier
    package_version: "1.2.10"
    preprocessing: native_catboost
    parameters:
      iterations: 500
      depth: 6
      learning_rate: 0.05
      l2_leaf_reg: 3.0
      random_strength: 0.0
      bootstrap_type: No
      loss_function: auto
      task_type: CPU
      thread_count: 2
      verbose: false
      allow_writing_files: false

  XGB-3.4:
    class: xgboost.XGBClassifier
    package_version: "3.4.1"
    preprocessing: common_onehot_no_scaling
    parameters:
      n_estimators: 500
      max_depth: 6
      learning_rate: 0.05
      min_child_weight: 1
      subsample: 1.0
      colsample_bytree: 1.0
      reg_lambda: 1.0
      reg_alpha: 0.0
      tree_method: hist
      n_jobs: 2
      verbosity: 0

  TPFN3-8.5:
    class: tabpfn.TabPFNClassifier
    package_version: "8.5.0"
    checkpoint: tabpfn-v3-classifier-v3_default.ckpt
    preprocessing: native_tabpfn
    parameters:
      n_estimators: 8
      auto_scale_n_estimators: false
      fit_mode: low_memory
      memory_saving_mode: true
      n_preprocessing_jobs: 1
      inference_precision: auto
      ignore_pretraining_limits: false
      show_progress_bar: false

  TICL2-2.2:
    class: tabicl.TabICLClassifier
    package_version: "2.2.0"
    checkpoint: tabicl-classifier-v2-20260212.ckpt
    preprocessing: native_tabicl
    parameters:
      n_estimators: 8
      batch_size: 1
      kv_cache: false
      offload_mode: auto
      use_amp: auto
      feat_shuffle_method: latin
      class_shuffle_method: shift
      average_logits: true
      n_jobs: 2
```

TabPFN documents `low_memory` as the most memory-efficient fit mode. `fit_with_cache` accelerates repeated inference but consumes substantially more memory; it should not be used on the primary 4 GB path. [TabPFN classifier implementation](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/classifier.py)

## 4. Frozen dataset suite

### 4.1 SchemaOrbit-14

All primary datasets are fixed by OpenML ID and version 1.

|   ID | Dataset                            |  Rows | Features | Task       | Primary schema property               |
| ---: | ---------------------------------- | ----: | -------: | ---------- | ------------------------------------- |
|    3 | `kr-vs-kp`                         | 3,196 |       36 | Binary     | Predominantly categorical             |
|   23 | `cmc`                              | 1,473 |        9 | Multiclass | Mixed categorical/numerical           |
|   29 | `credit-approval`                  |   690 |       15 | Binary     | Mixed and missing values              |
|   31 | `credit-g`                         | 1,000 |       20 | Binary     | Mixed financial schema                |
|   36 | `segment`                          | 2,310 |       19 | 7-class    | Numerical multiclass                  |
|   37 | `diabetes`                         |   768 |        8 | Binary     | Small numerical                       |
|   38 | `sick`                             | 3,772 |       29 | Binary     | Mixed and imbalanced                  |
|   44 | `spambase`                         | 4,601 |       57 | Binary     | Continuous numerical                  |
|   46 | `splice`                           | 3,190 |       60 | 3-class    | High-cardinality categorical sequence |
|   50 | `tic-tac-toe`                      |   958 |        9 | Binary     | Entirely categorical                  |
|   54 | `vehicle`                          |   846 |       18 | 4-class    | Numerical multiclass                  |
| 1067 | `kc1`                              | 2,109 |       21 | Binary     | Numerical and imbalanced              |
| 1464 | `blood-transfusion-service-center` |   748 |        4 | Binary     | Small numerical                       |
| 1489 | `phoneme`                          | 5,404 |        5 | Binary     | Medium numerical                      |

Dataset pages use:

```text
https://www.openml.org/d/{dataset_id}
```

Examples:

* [credit-g, ID 31](https://www.openml.org/d/31)
* [spambase, ID 44](https://www.openml.org/d/44)
* [tic-tac-toe, ID 50](https://www.openml.org/d/50)
* [blood-transfusion-service-center, ID 1464](https://www.openml.org/d/1464)
* [phoneme, ID 1489](https://www.openml.org/d/1489)

The suite is drawn from small and medium OpenML datasets. OpenML-CC18 supplies established dataset-selection principles and programmatic benchmark retrieval. [OpenML benchmarking suites](https://docs.openml.org/benchmark/)

### 4.2 Pilot subset

The pilot uses exactly six datasets:

```yaml
pilot_datasets:
  - 31    # credit-g
  - 36    # segment
  - 38    # sick
  - 44    # spambase
  - 50    # tic-tac-toe
  - 1489  # phoneme
```

This covers:

* Numerical and categorical data
* Binary and multiclass targets
* Small and medium sample sizes
* Imbalanced and balanced targets
* Four to sixty input features

### 4.3 Smoke dataset

```yaml
smoke_dataset: 1464
```

`blood-transfusion-service-center` is small enough to test the complete pipeline quickly.

### 4.4 Dataset constraints

Every fetched dataset must pass:

```yaml
constraints:
  min_rows: 500
  max_rows: 6000
  min_features: 4
  max_features: 100
  min_classes: 2
  max_classes: 10
  require_default_target: true
  require_public_access: true
  require_stable_checksum: true
```

The `splice` identifier column must be excluded using OpenML metadata.

## 5. Frozen splits and seeds

### 5.1 Seeds

```yaml
pilot_seeds:
  - 1729
  - 2718
  - 31415

main_seeds:
  - 1729
  - 2718
  - 31415
  - 57721
  - 161803
```

### 5.2 Split policy

```yaml
split:
  train: 0.60
  calibration: 0.20
  test: 0.20
  method: stratified_group
  strategy: stratified_group_5fold_v1
  group_by: predictors
  group_folds: 5
  minimum_class_count_per_partition: 5
```

Rules:

1. Create row IDs and exact type-aware predictor-group IDs before splitting.
2. Exclude row IDs and target columns from predictor grouping.
3. Generate five shuffled `StratifiedGroupKFold` folds with a deterministically derived seed.
4. Evaluate every three-fold-train, one-fold-calibration, one-fold-test assignment.
5. Select by row-fraction deviation, class-proportion deviation, and deterministic lexicographic tie-breaking.
6. Require complete, disjoint assignments with both classes in every partition and zero group crossing.
7. Persist the split assignments and manifest under the versioned grouped-strategy location.
8. Apply schema transformations after splitting.
9. Preserve identical row IDs across every view.
10. Use training data for model fitting.
11. Use calibration data for COSA selection and weights.
12. Use test data exactly once after configuration freeze.

The former row-stratified split is retained locally as a deprecated audit baseline. It must not be
silently overwritten or used as the active split.

All training partitions remain below the documented 5,000-sample CPU limit for the current default TabPFN-3 path.

## 6. Frozen schema views

Each dataset-seed pair has eleven scheduled view records. A record can have `not_applicable` status.

| View  | Name                         | Operation                           |
| ----- | ---------------------------- | ----------------------------------- |
| `V00` | `identity`                   | Original schema                     |
| `V01` | `numeric_affine_units`       | \(x'=ax+b\)                         |
| `V02` | `numeric_asinh`              | \(x'=\operatorname{asinh}(x/s)\)    |
| `V03` | `category_permutation`       | Reversible categorical code mapping |
| `V04` | `categorical_onehot`         | Reversible one-hot expansion        |
| `V05` | `duplicate_feature`          | Add an exact duplicate              |
| `V06` | `redundant_affine_feature`   | Add \(z=3x+7\)                      |
| `V07` | `integer_quotient_remainder` | Replace \(x\) by \(q,r\)            |
| `V08` | `column_permutation_control` | Reorder columns                     |
| `V09` | `row_permutation_control`    | Reorder training rows               |
| `V10` | `composite_migration`        | Compose V01, V03, and V08           |

### 6.1 Affine migration

$$
g(x)=ax+b,\qquad a\neq0
$$

$$
g^{-1}(x')=\frac{x'-b}{a}
$$

Parameters:

```yaml
scales: [0.1, 10.0, 1000.0]
offsets: [-7.0, 13.0]
maximum_columns: 3
```

Columns are selected deterministically using:

$$
j=\operatorname{argsort}
\left(
SHA256(dataset\_id\Vert seed\Vert column\_name)
\right)
$$

### 6.2 Asinh migration

$$
g(x)=\operatorname{asinh}\left(\frac{x}{s}\right)
$$

$$
g^{-1}(z)=s\sinh(z)
$$

where:

$$
s=\max\left(
\operatorname{median}\left|x-\operatorname{median}(x)\right|,
10^{-12}
\right)
$$

The scale is fitted from training features only and stored in the certificate.

### 6.3 Category permutation

For sorted category set:

$$
C=(c_1,\ldots,c_K)
$$

create a seeded cyclic permutation:

$$
g(c_j)=c_{((j+r)\bmod K)+1}
$$

The complete inverse dictionary must be stored.

### 6.4 Reversible one-hot migration

A categorical value \(c_j\) becomes:

$$
g(c_j)=e_j
$$

The inverse is valid only when:

* Exactly one category indicator is active, or
* A declared missing-category indicator is active.

Invalid multi-hot or zero-hot rows fail certification.

### 6.5 Duplicate feature

$$
g(X)=[X,x_j]
$$

Projection:

$$
\pi(g(X))=X
$$

### 6.6 Redundant affine feature

$$
z=3x_j+7
$$

The parent table is recovered by dropping \(z\).

### 6.7 Integer quotient-remainder split

For modulus \(m=10\):

$$
q=\left\lfloor\frac{x}{m}\right\rfloor
$$

$$
r=x-mq
$$

$$
x=mq+r
$$

Required certificate condition:

$$
0\leq r<m
$$

### 6.8 Negative controls

V08 and V09 are not novelty claims. They establish whether the implementation detects already-studied permutation effects.

## 7. Experiment count

### 7.1 Pilot

$$
6\text{ datasets}
\times
3\text{ seeds}
\times
5\text{ models}
\times
11\text{ views}
=
990
$$

The pilot manifest contains exactly **990 scheduled condition records**.

Not-applicable transformations remain in the manifest with a reason. They are not silently removed.

### 7.2 Main experiment

$$
14
\times
5
\times
5
\times
11
=
3850
$$

The complete main manifest contains exactly **3,850 scheduled condition records**.

### 7.3 Distribution by model family

| Family              | Maximum conditions |
| ------------------- | -----------------: |
| Logistic regression |                770 |
| CatBoost            |                770 |
| XGBoost             |                770 |
| TabPFN-3            |                770 |
| TabICLv2            |                770 |
| Total               |              3,850 |

### 7.4 Foundation-model workload

$$
14\times5\times2\times11=1540
$$

Maximum foundation-model jobs: **1,540**.

### 7.5 Reuse

The 990 pilot conditions are part of the 3,850 main conditions. Passing pilot predictions are reused when every hash matches.

Additional maximum jobs after the pilot:

$$
3850-990=2860
$$

## 8. Formal definitions

### 8.1 Certified lossless migration

Let:

$$
D=(X,y,\mathcal S)
$$

where \(\mathcal S\) is the schema.

A migration \(g\) is certified lossless on \(D\) when at least one of the following exists:

#### Bijection certificate

$$
r_g(g(X))=X
$$

#### Projection certificate

$$
\pi_g(g(X))=X
$$

#### Permutation certificate

$$
P_g^{-1}P_gX=X
$$

It must also preserve:

$$
y'=y
$$

and row identities:

$$
ID(g(x_i))=ID(x_i)
$$

### 8.2 Pipeline prediction

For dataset \(d\), model \(m\), seed \(s\), view \(v\), and test row \(i\):

$$
p_{dmsvi}
=
A_m\left(
g_v(D_d^{train});s
\right)
\left(
g_v(x_i)
\right)
$$

The complete preprocessing-and-model pipeline \(A_m\) is evaluated, not only the final estimator.

### 8.3 Jensen–Shannon divergence

$$
JSD(p,q)
=
\frac12 KL(p\Vert m)
+
\frac12 KL(q\Vert m)
$$

where:

$$
m=\frac12(p+q)
$$

### 8.4 Schema Invariance Gap

Define the mean probability across valid views:

$$
\bar p_i
=
\frac{1}{|\mathcal V|}
\sum_{v\in\mathcal V}p_{vi}
$$

Row-level gap:

$$
SIG_i
=
\max_{v\in\mathcal V}
JSD(p_{vi},\bar p_i)
$$

Dataset-level gap:

$$
SIG
=
\frac1n\sum_{i=1}^{n}SIG_i
$$

### 8.5 Schema Flip Rate

$$
SFR
=
\frac1n
\sum_{i=1}^{n}
\mathbf 1
\left[
\left|
\left\{
\arg\max_c p_{vic}:v\in\mathcal V
\right\}
\right|>1
\right]
$$

### 8.6 Multiclass Brier risk

$$
R_v
=
\frac1n
\sum_{i=1}^{n}
\sum_{c=1}^{C}
\left(
p_{vic}-\mathbf1[y_i=c]
\right)^2
$$

### 8.7 Worst-View Excess Risk

$$
WVER
=
\max_{v\in\mathcal V}R_v-R_{V00}
$$

### 8.8 Identity noise floor

Repeat V00 once under the same frozen seed and environment:

$$
\eta
=
\left|
R_{V00}^{repeat}-R_{V00}
\right|
$$

Noise-adjusted risk:

$$
WVER^+
=
\max(0,WVER-\eta)
$$

### 8.9 Recovery

$$
Recovery
=
\frac{
WVER^+_{raw}-WVER^+_{SchemaGuard}
}{
\max(WVER^+_{raw},10^{-12})
}
$$

### 8.10 SCNF guarantee

The canonicalizer must satisfy:

$$
C(g_v(D))=C(D)
$$

for every supported certified migration.

Consequently:

$$
A_m(C(g_v(D)))=A_m(C(D))
$$

up to the declared deterministic numerical tolerance.

### 8.11 COSA prediction

For selected view set \(S\):

$$
q_i(\alpha)
=
\sum_{v\in S}\alpha_vp_{vi}
$$

subject to:

$$
\alpha_v\ge0
$$

$$
\sum_{v\in S}\alpha_v=1
$$

$$
|S|\le B
$$

### 8.12 COSA optimization

$$
\min_{\alpha,S}
\max_{e\in\mathcal E}
\hat R_e(\alpha)
+
\lambda
\sum_{v\in S}\alpha_vc_v
$$

subject to:

$$
\alpha\in\Delta^{|\mathcal V|}
$$

$$
\|\alpha\|_0\le B
$$

where:

* \(\mathcal E\) contains calibration folds.
* \(c_v\) is normalized inference cost.
* \(B\in\{2,4\}\).
* \(\lambda\in\{0,0.01,0.05\}\).

COSA uses deterministic forward selection:

1. Start with the SCNF view.
2. Test every unselected candidate.
3. Solve simplex weights using SciPy SLSQP.
4. Measure worst-fold Brier loss plus cost.
5. Add the best candidate.
6. Stop at budget \(B\) or improvement below \(10^{-4}\).
7. Resolve ties using canonical view ID.

## 9. Compared methods

| ID            | Method                                        |
| ------------- | --------------------------------------------- |
| `RAW`         | Frozen native pipeline on V00                 |
| `RANK`        | Rank-normalized numerical baseline            |
| `SCNF`        | Schema Canonical Normal Form                  |
| `UNIFORM`     | Equal-weight average across valid orbit views |
| `COSA-B2`     | COSA with maximum two views                   |
| `COSA-B4`     | COSA with maximum four views                  |
| `SCHEMAGUARD` | SCNF followed by COSA-B4 fallback             |

Primary repair comparison:

$$
SCHEMAGUARD
\quad\text{versus}\quad
\max(RAW,RANK,SCNF,UNIFORM)
$$

## 10. Primary hypotheses

| ID  | Hypothesis                                | Pass condition                                                                     |
| --- | ----------------------------------------- | ---------------------------------------------------------------------------------- |
| H1  | Practical pipelines are schema-sensitive  | Median \(WVER^+\ge0.005\) or median \(SFR\ge0.05\) in at least one nonlinear model |
| H2  | Effect is not confined to permutations    | At least two non-permutation migration families show positive adjusted drift       |
| H3  | Effect is not purely preprocessing noise  | Remains in the common-representation attribution experiment                        |
| H4  | SCNF provides exact supported invariance  | Maximum deterministic probability difference \(\le10^{-8}\) on CPU models          |
| H5  | SchemaGuard repairs the effect            | Median recovery \(\ge40\%\)                                                        |
| H6  | Repair causes no material predictive harm | Upper confidence bound for Brier increase \(<0.005\)                               |
| H7  | COSA contributes beyond simple averaging  | COSA-B4 beats UNIFORM across datasets                                              |
| H8  | Result is broad                           | Positive repair in at least three nonlinear model families                         |
| H9  | Method is computationally viable          | Median inference-cost multiplier \(\le4\)                                          |
| H10 | Real migrations reproduce the effect      | At least three of five real schema pairs show measurable drift                     |

## 11. Main weaknesses and corrections

| Weakness                                                   | Severity | Correction                                                                 |
| ---------------------------------------------------------- | -------- | -------------------------------------------------------------------------- |
| Existing 2026 work studies TFM permutation invariance      | Critical | Treat row/column/class permutations as controls                            |
| Metamorphic testing is established                         | Critical | Do not claim the testing concept as new                                    |
| Pipeline preprocessing may cause the effect                | High     | Add common-representation attribution experiment                           |
| Losslessness may fail because of floating-point conversion | High     | Require inverse/projection certificate and tolerance                       |
| One-hot encoding may not preserve unknown categories       | High     | Include missing/unknown token and strict inverse rules                     |
| Schema migrations may appear artificial                    | High     | Add at least five real schema pairs                                        |
| Canonicalization can make aggregation unnecessary          | High     | Report SCNF alone; COSA must beat it                                       |
| COSA may reduce to ordinary ensembling                     | High     | Compare against uniform and single-best-view baselines                     |
| GPU nondeterminism may imitate drift                       | High     | Identity rerun noise floor and CPU confirmation                            |
| Different model preprocessing harms comparability          | High     | Separate practical-pipeline and common-representation protocols            |
| Failed TabPFN or TabICL jobs can bias results              | High     | Report all scheduled jobs, failures, and not-applicable conditions         |
| Only fourteen datasets may appear narrow                   | Medium   | Emphasize transformation depth and real schema validation                  |
| Dataset versions can drift                                 | Medium   | Store OpenML ID, version, URL, and downloaded checksum                     |
| Model releases can change                                  | Medium   | Pin packages, checkpoint names, checkpoint hashes, and `uv.lock`           |
| TabPFN weights have non-commercial terms                   | Medium   | Record license text and acceptance date                                    |
| CPU/GPU latency is not directly comparable                 | Medium   | Report hardware-specific cost; do not create a single misleading ranking   |
| Multiple views and seeds are statistically dependent       | High     | Aggregate at dataset level and cluster bootstrap by dataset                |
| Calibration reuse can overfit COSA                         | High     | Use nested folds inside calibration and untouched test data                |
| Cache contamination can create false reproducibility       | Critical | Content-addressed cache with code, lock, data, view, and checkpoint hashes |

TabPFN-2.5, 2.6, and 3 weights are distributed under non-commercial licenses. This does not automatically block academic research, but the accepted license and checkpoint hash must be archived. [TabPFN licensing](https://github.com/PriorLabs/TabPFN)

## 12. Repository structure

```text
schemaguard/
├── SCHEMAGUARD_MASTER_PLAN.md
├── AGENTS.md
├── README.md
├── CHANGELOG.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env.example
├── .gitignore
├── Makefile
├── configs/
│   ├── project.yaml
│   ├── datasets.yaml
│   ├── models.yaml
│   ├── views.yaml
│   ├── smoke.yaml
│   ├── pilot.yaml
│   ├── main.yaml
│   ├── statistics.yaml
│   └── resources.yaml
├── schemas/
│   ├── dataset_card.schema.json
│   ├── split_manifest.schema.json
│   ├── view_certificate.schema.json
│   ├── run_manifest.schema.json
│   ├── prediction_record.schema.json
│   ├── metric_record.schema.json
│   └── validation_record.schema.json
├── data/
│   ├── registry/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/schemaguard/
│   ├── __init__.py
│   ├── cli.py
│   ├── constants.py
│   ├── types.py
│   ├── data/
│   │   ├── registry.py
│   │   ├── fetch_openml.py
│   │   ├── fingerprints.py
│   │   ├── validate.py
│   │   └── splits.py
│   ├── views/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── affine.py
│   │   ├── asinh.py
│   │   ├── categorical.py
│   │   ├── onehot.py
│   │   ├── duplicate.py
│   │   ├── redundant.py
│   │   ├── integer_split.py
│   │   ├── permutations.py
│   │   ├── composition.py
│   │   ├── certificate.py
│   │   └── materialize.py
│   ├── models/
│   │   ├── base.py
│   │   ├── preprocessing.py
│   │   ├── logistic.py
│   │   ├── catboost.py
│   │   ├── xgboost.py
│   │   ├── tabpfn.py
│   │   ├── tabicl.py
│   │   └── factory.py
│   ├── cache/
│   │   ├── keys.py
│   │   ├── locks.py
│   │   ├── store.py
│   │   └── index.py
│   ├── runner/
│   │   ├── plan.py
│   │   ├── worker.py
│   │   ├── scheduler.py
│   │   ├── state.py
│   │   └── resources.py
│   ├── metrics/
│   │   ├── predictive.py
│   │   ├── invariance.py
│   │   ├── calibration.py
│   │   └── compute.py
│   ├── repair/
│   │   ├── scnf.py
│   │   ├── orbit.py
│   │   ├── simplex.py
│   │   ├── cosa.py
│   │   └── gate.py
│   ├── statistics/
│   │   ├── aggregate.py
│   │   ├── paired.py
│   │   ├── bootstrap.py
│   │   ├── multiplicity.py
│   │   └── mixed_effects.py
│   ├── validation/
│   │   ├── schema.py
│   │   ├── dataset.py
│   │   ├── split.py
│   │   ├── transform.py
│   │   ├── prediction.py
│   │   ├── leakage.py
│   │   ├── statistics.py
│   │   ├── reproduction.py
│   │   └── suite.py
│   └── reporting/
│       ├── tables.py
│       ├── plot_overview.py
│       ├── plot_heatmaps.py
│       ├── plot_paired.py
│       ├── plot_pareto.py
│       ├── plot_calibration.py
│       ├── plot_ablation.py
│       ├── build_report.py
│       └── lineage.py
├── scripts/
│   ├── check_environment.py
│   ├── fetch_datasets.py
│   ├── build_splits.py
│   ├── build_views.py
│   ├── run_smoke.py
│   ├── run_pilot.py
│   ├── validate_pilot.py
│   ├── freeze_main.py
│   ├── run_main.py
│   ├── run_repair.py
│   ├── analyze_results.py
│   ├── make_paper_artifacts.py
│   └── reproduce_results.py
├── tests/
│   ├── unit/
│   ├── property/
│   ├── integration/
│   ├── regression/
│   ├── statistical/
│   └── smoke/
├── results/
│   ├── manifests/
│   ├── predictions/
│   ├── metrics/
│   ├── statistics/
│   ├── validation/
│   └── resources/
├── artifacts/
│   ├── figures/
│   ├── tables/
│   ├── reports/
│   └── handoff/
└── paper/
    ├── manuscript.tex
    ├── references.bib
    ├── sections/
    ├── generated/
    └── supplement/
```

## 13. File contracts

### 13.1 Root and configuration files

| File                         | Input                     | Output                           | Validation                     | Optimization                              |
| ---------------------------- | ------------------------- | -------------------------------- | ------------------------------ | ----------------------------------------- |
| `SCHEMAGUARD_MASTER_PLAN.md` | Frozen research decisions | Complete implementation contract | Manual review and hash         | No runtime role                           |
| `AGENTS.md`                  | Phase rules               | Luna execution instructions      | Prohibit scope changes         | Short context-efficient directives        |
| `README.md`                  | Validated commands        | User documentation               | Commands executed in CI        | Avoid duplicated documentation            |
| `pyproject.toml`             | Package policy            | Installable project              | `uv lock`, build test          | Optional dependency groups                |
| `uv.lock`                    | Resolved packages         | Exact dependency graph           | Lock hash recorded             | Enables reproducible environment cache    |
| `.python-version`            | `3.12`                    | Interpreter constraint           | Runtime check                  | Prevents accidental interpreter switching |
| `.env.example`               | Path and device variables | Safe environment template        | No secret values               | Centralized cache paths                   |
| `configs/datasets.yaml`      | Frozen OpenML IDs         | Dataset configuration            | Pydantic and source validation | Read once and memoize                     |
| `configs/models.yaml`        | Frozen model parameters   | Model configurations             | Unknown keys rejected          | Hash configurations once                  |
| `configs/views.yaml`         | Eleven view definitions   | Transformation plan              | Applicability schema           | Deterministic parameter generation        |
| `configs/resources.yaml`     | Laptop limits             | Scheduler limits                 | Hardware preflight             | Prevent oversubscription                  |
| `configs/statistics.yaml`    | Endpoints and margins     | Frozen analysis plan             | Hash before test               | Reuse bootstrap indices                   |

### 13.2 Dataset files

| File                   | Input                      | Output                        | Validation                      | Optimization                              |
| ---------------------- | -------------------------- | ----------------------------- | ------------------------------- | ----------------------------------------- |
| `data/registry.py`     | `datasets.yaml`            | Typed `DatasetSpec`           | ID and version uniqueness       | Module-level immutable cache              |
| `data/fetch_openml.py` | `DatasetSpec`              | Raw ARFF/Parquet and metadata | URL, ID, target, checksum       | Download once; conditional reuse          |
| `data/fingerprints.py` | Files and DataFrames       | SHA-256 fingerprints          | Round-trip consistency          | Stream hashing; avoid full duplicate copy |
| `data/validate.py`     | Raw table                  | Dataset-quality JSON          | Target, rows, types, duplicates | Vectorized checks                         |
| `data/splits.py`       | Validated dataset and seed | Split Parquet                 | No overlap and class coverage   | Persist assignments; never recompute      |

### 13.3 Transformation files

| File                     | Input                 | Output                             | Validation                     | Optimization                        |
| ------------------------ | --------------------- | ---------------------------------- | ------------------------------ | ----------------------------------- |
| `views/base.py`          | Table bundle          | Transform protocol                 | Static typing                  | Zero-copy references where safe     |
| `views/registry.py`      | View config           | Transform instance                 | Unique view IDs                | Cached constructor                  |
| `views/affine.py`        | Numerical columns     | Affine view and inverse parameters | Reconstruction tolerance       | NumPy vectorization                 |
| `views/asinh.py`         | Numerical columns     | Asinh view and scale parameters    | Finite inverse check           | Vectorized ufuncs                   |
| `views/categorical.py`   | Category dictionaries | Permuted categories                | Exact inverse mapping          | Pandas categorical codes            |
| `views/onehot.py`        | Categorical features  | One-hot view                       | Exactly-one-active invariant   | Sparse intermediate construction    |
| `views/duplicate.py`     | Selected feature      | Expanded table                     | Projection equality            | Arrow zero-copy when possible       |
| `views/redundant.py`     | Numerical feature     | Derived redundant column           | Formula equality               | Vectorized arithmetic               |
| `views/integer_split.py` | Integer feature       | Quotient and remainder             | Exact reconstruction           | NumPy `divmod`                      |
| `views/permutations.py`  | Rows or columns       | Permuted table and inverse index   | Index-map equality             | Store indices, not copied metadata  |
| `views/composition.py`   | Certified transforms  | Composite view                     | Certificate-chain verification | Materialize once after composition  |
| `views/certificate.py`   | Parent and child      | Certificate JSON                   | JSON Schema and proof checks   | Hash normalized metadata            |
| `views/materialize.py`   | Split and transform   | Cached Parquet view                | Parent/view hash equality      | Zstandard Parquet and atomic writes |

### 13.4 Model files

| File                      | Input                       | Output                  | Validation                         | Optimization                       |
| ------------------------- | --------------------------- | ----------------------- | ---------------------------------- | ---------------------------------- |
| `models/base.py`          | Model config                | Adapter protocol        | Output probability contract        | Shared batching API                |
| `models/preprocessing.py` | Raw view and manifest       | Model-specific matrices | Train-only fitting                 | Cache fitted transformers          |
| `models/logistic.py`      | Encoded table               | Probability predictions | Convergence and class order        | Sparse one-hot matrix              |
| `models/catboost.py`      | Native mixed table          | Probability predictions | Category indices and finite output | Two threads; no bootstrap          |
| `models/xgboost.py`       | Encoded table               | Probability predictions | Feature order and class order      | Histogram trees; two threads       |
| `models/tabpfn.py`        | Native table and checkpoint | Probability predictions | Checkpoint hash and OOM handling   | Low-memory mode; batched test rows |
| `models/tabicl.py`        | Native table and checkpoint | Probability predictions | Checkpoint hash and offload status | Batch size one; CPU/disk offload   |
| `models/factory.py`       | Model ID                    | Adapter instance        | Registered IDs only                | Lazy import of heavy libraries     |

### 13.5 Cache and scheduler files

| File                  | Input              | Output                          | Validation                       | Optimization                                |
| --------------------- | ------------------ | ------------------------------- | -------------------------------- | ------------------------------------------- |
| `cache/keys.py`       | All causal hashes  | Cache key                       | Deterministic serialization test | Hash small manifests, not arrays repeatedly |
| `cache/locks.py`      | Cache key          | File lock                       | Stale-lock timeout               | One lock per artifact                       |
| `cache/store.py`      | Typed artifact     | Atomic cached file              | Checksum after write             | Temporary write followed by rename          |
| `cache/index.py`      | Artifact manifests | DuckDB index                    | Unique artifact key              | Parquet predicate pushdown                  |
| `runner/plan.py`      | Configurations     | 3,850-row run manifest          | Expected Cartesian product       | Generate lazily                             |
| `runner/worker.py`    | One condition      | Predictions and resource record | Status transition checks         | Reuse loaded dataset in worker              |
| `runner/scheduler.py` | Pending conditions | Completed conditions            | No duplicate execution           | CPU pool plus exclusive GPU queue           |
| `runner/state.py`     | Run events         | State JSON                      | Legal state-machine transitions  | Append-only event log                       |
| `runner/resources.py` | Process and device | RAM/VRAM/time records           | Nonnegative and plausible values | Low-frequency sampling thread               |

### 13.6 Metrics and repair files

| File                     | Input                             | Output                     | Validation                       | Optimization                          |
| ------------------------ | --------------------------------- | -------------------------- | -------------------------------- | ------------------------------------- |
| `metrics/predictive.py`  | Labels and probabilities          | Brier, log loss, accuracy  | Reference fixtures               | NumPy vectorization                   |
| `metrics/invariance.py`  | Aligned view predictions          | SIG, SFR, WVER             | Hand-calculated examples         | Batched pairwise divergence           |
| `metrics/calibration.py` | Labels and probabilities          | ECE and reliability data   | Bin-count checks                 | Precomputed bin indices               |
| `metrics/compute.py`     | Resource records                  | Cost-normalized metrics    | Hardware stratification          | DuckDB aggregation                    |
| `repair/scnf.py`         | Certified view                    | Canonical table            | Idempotence and orbit equality   | Canonical output cache                |
| `repair/orbit.py`        | Certificates                      | Ordered orbit              | Closure and canonical ordering   | Reuse existing materialized views     |
| `repair/simplex.py`      | Prediction matrix                 | Optimal weights            | Simplex and solver-status checks | Warm-start SLSQP                      |
| `repair/cosa.py`         | Calibration predictions and costs | Selected views and weights | Nested calibration validation    | Greedy search and memoized candidates |
| `repair/gate.py`         | Frozen repair and uncertainty     | Deployment certificate     | Non-inferiority rule             | Vectorized confidence calculation     |

### 13.7 Statistics and reporting files

| File                            | Input                    | Output                         | Validation                     | Optimization                   |
| ------------------------------- | ------------------------ | ------------------------------ | ------------------------------ | ------------------------------ |
| `statistics/aggregate.py`       | Row metrics              | Dataset-level analysis Parquet | Grain and duplicate tests      | DuckDB group-by                |
| `statistics/paired.py`          | Dataset-level pairs      | Wilcoxon and effect sizes      | Pair-key equality              | Operate on small aggregates    |
| `statistics/bootstrap.py`       | Dataset statistics       | Confidence intervals           | Coverage simulation            | Cache 10,000 bootstrap indices |
| `statistics/multiplicity.py`    | Raw p-values             | Holm-adjusted results          | Monotonic adjusted values      | Vectorized sorting             |
| `statistics/mixed_effects.py`   | Long-form analysis table | Interaction estimates          | Convergence diagnostics        | Fit only after aggregation     |
| `reporting/tables.py`           | Validated analysis       | LaTeX/CSV tables               | Source-hash footers            | Single query per table         |
| `reporting/plot_overview.py`    | Summary data             | Main overview figure           | Axis and sample-count checks   | Preaggregate data              |
| `reporting/plot_heatmaps.py`    | Model-view matrix        | SIG/WVER heatmaps              | Complete-cell check            | Rasterized heatmap body        |
| `reporting/plot_paired.py`      | Paired method results    | Slope and forest plots         | Identical dataset set          | Shared figure style            |
| `reporting/plot_pareto.py`      | Risk and cost data       | Risk-cost frontier             | Dominance calculation test     | Convex-hull reduction          |
| `reporting/plot_calibration.py` | Calibration records      | Reliability diagrams           | Bin support labels             | Reuse calibration bins         |
| `reporting/plot_ablation.py`    | Ablation records         | Ablation forest plot           | Baseline alignment             | Precomputed intervals          |
| `reporting/build_report.py`     | All validated artifacts  | Pilot/main Markdown report     | Lineage audit                  | Incremental rebuild            |
| `reporting/lineage.py`          | Figures and tables       | Artifact lineage manifest      | Every output has source hashes | Hash only changed artifacts    |

## 14. Content-addressed cache

### 14.1 Cache key

$$
K=
SHA256(
H_D
\Vert H_S
\Vert H_V
\Vert H_M
\Vert H_C
\Vert H_L
\Vert seed
\Vert H_{code}
\Vert device\_policy
)
$$

where:

* \(H_D\): dataset checksum
* \(H_S\): split checksum
* \(H_V\): view-certificate checksum
* \(H_M\): model-configuration checksum
* \(H_C\): checkpoint checksum
* \(H_L\): dependency-lock checksum

### 14.2 Cache hierarchy

```text
cache/
├── datasets/{dataset_hash}.parquet
├── splits/{dataset_hash}/{split_hash}.parquet
├── views/{dataset_hash}/{split_hash}/{view_hash}.parquet
├── preprocessors/{preprocessor_hash}.joblib
├── models/{model_fit_hash}/
├── predictions/{prediction_hash}.parquet
├── metrics/{metric_hash}.parquet
└── bootstrap/{statistics_hash}.npy
```

### 14.3 Storage format

* Parquet compression: Zstandard level 3.
* Probability storage: `float32`.
* Statistical calculations: cast to `float64`.
* Row IDs: fixed-width string or `uint64`.
* Partition keys: dataset, model, seed, view, split.
* Metadata and certificates: canonical JSON.
* Query engine: DuckDB.
* Completed artifacts: immutable.
* Failed artifacts: never reused.
* Writes: temporary file followed by atomic rename.

## 15. Parallelization policy

```yaml
resources:
  ram_soft_limit_gb: 24
  ram_hard_limit_gb: 28
  vram_soft_limit_gb: 3.6
  cpu_condition_workers: 2
  cpu_threads_per_worker: 2
  transform_workers: 4
  gpu_workers: 1
  tabpfn_workers: 1
  tabicl_workers: 1
```

Rules:

1. Parallelize CPU models across dataset-seed conditions.
2. Do not run CatBoost and XGBoost with unrestricted internal threads.
3. Set BLAS threads to two per worker.
4. Run only one foundation-model process.
5. Use an exclusive GPU lock.
6. Do not run TabPFN and TabICL simultaneously.
7. Cache predictions immediately after validation.
8. Release model and CUDA memory after every condition.
9. Call `torch.cuda.empty_cache()` only after object deletion.
10. Perform DuckDB result consolidation with one writer.
11. Allow many read-only report processes.
12. Resume only from validated immutable artifacts.

## 16. Validation layers

| Layer          | Checks                                                  | Blocking |
| -------------- | ------------------------------------------------------- | -------- |
| Environment    | Python, wheels, CUDA, package imports                   | Yes      |
| Registry       | Dataset ID, version, target, license                    | Yes      |
| Data           | Checksum, row IDs, duplicates, feature types            | Yes      |
| Split          | No overlap, class coverage, identical mappings          | Yes      |
| Transformation | Inverse/project, target preservation, row alignment     | Yes      |
| SCNF           | Idempotence and orbit equality                          | Yes      |
| Model          | Class order, finite output, deterministic configuration | Yes      |
| Prediction     | Completeness, uniqueness, probability simplex           | Yes      |
| Leakage        | No test use in fitting or COSA selection                | Yes      |
| Statistics     | Correct grain, paired keys, multiplicity                | Yes      |
| Reporting      | Every value traceable to source artifact                | Yes      |
| Reproduction   | Fresh environment reproduces reported precision         | Yes      |

### 16.1 Numerical tolerances

| Test                            |                  Tolerance |
| ------------------------------- | -------------------------: |
| Integer/category reconstruction |                      Exact |
| Missing masks                   |                      Exact |
| Floating transformation inverse | `rtol=1e-10`, `atol=1e-12` |
| Probability sum                 |                `atol=1e-6` |
| CPU pipeline repeat             |    `max_abs_diff <= 1e-10` |
| GPU TFM repeat                  |     `max_abs_diff <= 1e-5` |
| SCNF idempotence                |          Exact schema hash |
| Paper table reproduction        |  Exact displayed precision |

## 17. Statistical analysis

### 17.1 Primary inferential unit

The dataset is the primary inferential unit.

Seeds and rows are repeated measurements, not independent datasets.

### 17.2 Seed aggregation

For each dataset and model:

$$
\widetilde{WVER}_{dm}
=
\operatorname{median}_{s}
WVER_{dms}
$$

For repair:

$$
\Delta_{dm}
=
\operatorname{median}_{s}
\left(
WVER^{raw}_{dms}
-
WVER^{SchemaGuard}_{dms}
\right)
$$

Dataset-level aggregate across nonlinear models:

$$
\Delta_d
=
\frac14
\sum_{m\in
\{CAT,XGB,TPFN,TICL\}}
\Delta_{dm}
$$

### 17.3 Primary tests

| Test                               | Method                                                 |
| ---------------------------------- | ------------------------------------------------------ |
| Existence of schema sensitivity    | One-sided dataset-blocked sign-flip test               |
| Repair improvement                 | One-sided Wilcoxon signed-rank on \(\Delta_d\)         |
| Effect interval                    | 10,000-replicate dataset bootstrap                     |
| Multiple methods                   | Friedman test followed by corrected paired comparisons |
| Model-specific repair              | Wilcoxon with Holm correction                          |
| Model × transformation interaction | Mixed-effects model plus bootstrap sensitivity         |
| Original-performance protection    | Non-inferiority confidence bound                       |

### 17.4 Non-inferiority

$$
Z_d
=
Brier_d^{SchemaGuard}
-
Brier_d^{RAW}
$$

Hypotheses:

$$
H_0:\mu_Z\ge0.005
$$

$$
H_1:\mu_Z<0.005
$$

SchemaGuard passes only when the upper 95% cluster-bootstrap confidence bound is below `0.005`.

### 17.5 Required reporting

For every comparison report:

* Number of datasets
* Number of valid conditions
* Number of failed conditions
* Median difference
* Mean difference
* 95% confidence interval
* Hodges–Lehmann estimate
* Raw p-value
* Adjusted p-value
* Win–tie–loss count
* Effect direction
* Predeclared decision threshold

## 18. Required figures

| Figure | File                        | Content                                                    |
| ------ | --------------------------- | ---------------------------------------------------------- |
| F1     | `fig01_problem.pdf`         | Same information, different schemas, different predictions |
| F2     | `fig02_architecture.pdf`    | Certificates, SCNF, models, audit, COSA                    |
| F3     | `fig03_applicability.pdf`   | Dataset-by-transformation applicability heatmap            |
| F4     | `fig04_wver.pdf`            | WVER distributions by model and migration                  |
| F5     | `fig05_sig_heatmap.pdf`     | SIG heatmap                                                |
| F6     | `fig06_repair_pairs.pdf`    | Raw versus SchemaGuard paired dataset results              |
| F7     | `fig07_recovery_forest.pdf` | Recovery estimates with confidence intervals               |
| F8     | `fig08_risk_cost.pdf`       | COSA risk-cost Pareto frontier                             |
| F9     | `fig09_flip_ecdf.pdf`       | SFR empirical cumulative distributions                     |
| F10    | `fig10_calibration.pdf`     | Reliability diagrams before and after repair               |
| F11    | `fig11_resources.pdf`       | Runtime, RAM, VRAM, and failure rates                      |
| F12    | `fig12_ablations.pdf`       | SCNF, uniform, B2, and B4 ablations                        |
| F13    | `fig13_real_pairs.pdf`      | Real schema-pair case studies                              |

Plot requirements:

* Use vector PDF and SVG for the paper.
* Use PNG only for README previews.
* Display dataset counts.
* Display failed-condition counts.
* Use identical method colors across every figure.
* Avoid bar charts for paired outcomes.
* Show raw observations where readable.
* Do not hide datasets that contradict the main result.

## 19. Required tables

| Table | Content                                                                     |
| ----- | --------------------------------------------------------------------------- |
| T1    | Difference from permutation invariance, robustness, and metamorphic testing |
| T2    | SchemaOrbit-14 dataset registry                                             |
| T3    | Exact models, versions, checkpoints, and parameters                         |
| T4    | Transformation definitions and certificate types                            |
| T5    | Scheduled, completed, failed, and not-applicable conditions                 |
| T6    | Main WVER, SIG, and SFR results                                             |
| T7    | SchemaGuard recovery and non-inferiority                                    |
| T8    | Model-specific adjusted statistical comparisons                             |
| T9    | COSA ablations                                                              |
| T10   | Runtime, RAM, VRAM, and cost multipliers                                    |
| T11   | Real schema-pair results                                                    |
| T12   | Reproducibility and artifact inventory                                      |

## 20. Stepwise implementation plan

### Phase 0 — Freeze specification

Implement:

* Root files
* Configurations
* Schemas
* Dataset/model/view identifiers

Tests:

* Configuration parsing
* Unknown key rejection
* Duplicate ID rejection
* Expected pilot count equals 990
* Expected main count equals 3,850

Exit artifact:

```text
artifacts/handoff/experiment_registry_review.md
```

Do not continue if the Cartesian-product counts differ.

### Phase 1 — Environment compatibility

Run imports and one-row constructors for all five models.

Test:

* Python 3.12
* CatBoost import
* XGBoost import
* TabPFN checkpoint access
* TabICL checkpoint access
* CUDA visibility
* CPU fallback
* License records
* Peak model-load VRAM

Exit artifact:

```text
results/validation/environment_report.json
```

Do not continue to the pilot if TabPFN or TabICL lacks a working CPU fallback.

### Phase 2 — Dataset registry

Fetch all fourteen datasets and store:

* Metadata
* Original file
* Parquet conversion
* Checksum
* Dataset card
* Feature types
* Target
* Ignored columns

Tests:

* Exact OpenML ID and version
* Target exists
* Row count matches metadata
* Unique internal row IDs
* No target in predictors
* Parquet round trip

Exit artifact:

```text
results/validation/dataset_validation.parquet
```

### Phase 3 — Split system

Generate five split manifests for all fourteen datasets.

Tests:

* Train/calibration/test are disjoint
* Union equals complete dataset
* Every class appears where possible
* Same seed reproduces same split hash
* Different seeds produce different assignment hashes

Exit:

```text
data/processed/splits/
```

### Phase 4 — Transformation engine

Implement one transformation at a time:

1. V00 identity
2. V01 affine
3. V02 asinh
4. V03 category permutation
5. V04 one-hot
6. V05 duplicate
7. V06 redundant affine
8. V07 quotient-remainder
9. V08 column permutation
10. V09 row permutation
11. V10 composition

Each implementation requires:

* Unit test
* Property-based test
* Certificate test
* Parquet round-trip test
* Null-value test
* Invalid-input test

Do not implement the next transformation until the current inverse or projection property passes 1,000 generated examples.

### Phase 5 — Model adapters

Implement in this order:

1. Logistic regression
2. CatBoost
3. XGBoost
4. TabPFN
5. TabICL

Each adapter requires:

* Binary classification fixture
* Multiclass fixture
* Missing-value fixture
* Categorical fixture
* Class-order test
* Probability-simplex test
* Save/reload or prediction-cache test
* Resource record
* Failure classification

### Phase 6 — Cache and scheduler

Implement:

* Cache keys
* Atomic writes
* File locks
* Resume logic
* CPU worker pool
* Exclusive GPU queue
* Resource monitor

Fault-injection tests:

* Process terminates during write
* Stale lock exists
* Cache file is truncated
* Model configuration changes
* Code commit changes
* Dataset checksum changes
* Two workers request the same artifact

No duplicate validated artifact may be written.

### Phase 7 — Smoke test

Run:

```text
dataset: 1464
seed: 1729
models: all five
views: V00 and V01
```

Maximum conditions:

$$
1\times1\times5\times2=10
$$

Required:

* Ten condition records
* Valid predictions or explicit capability failures
* Resource report
* Complete validation summary

### Phase 8 — Pilot

Run the exact 990-condition manifest.

Order:

1. Logistic regression
2. CatBoost
3. XGBoost
4. TabPFN
5. TabICL
6. Metrics
7. SCNF
8. Uniform orbit baseline
9. COSA-B2
10. COSA-B4
11. Statistical pilot report

Pilot gate:

* H1, H2, H4, H5, and H6 must pass.
* TabPFN and TabICL must each complete at least four of six datasets.
* COSA must beat uniform averaging.
* SCNF must not make COSA unnecessary on every dataset.

Failure means redesign or stop. Do not launch the main experiment automatically.

### Phase 9 — Freeze main analysis

Create:

```text
configs/main.frozen.yaml
configs/statistics.frozen.yaml
results/manifests/main_plan.parquet
results/manifests/freeze_manifest.json
```

Record hashes and prohibit further edits.

### Phase 10 — Main runs

Execute the remaining 2,860 maximum conditions.

Run CPU conditions in parallel. Run TFM conditions sequentially.

After every dataset-model-seed block:

1. Validate predictions.
2. Write resource record.
3. Update condition status.
4. Release memory.
5. Recompute completeness counts.

### Phase 11 — Repair and ablations

Use calibration predictions only.

Run:

* RANK
* SCNF
* UNIFORM
* COSA-B2
* COSA-B4
* Full SchemaGuard

Ablations use cached predictions. They should not refit the five base models unless the model’s own estimator-count ablation is being tested.

### Phase 12 — Statistics

Generate one immutable analysis table:

```text
results/statistics/analysis_dataset.parquet
```

No plotting function may read raw prediction files directly. All paper analysis must use the validated analysis table.

### Phase 13 — Reporting and reproduction

Generate:

* Thirteen figures
* Twelve tables
* Validation appendix
* Failure appendix
* Hardware appendix
* Reproduction manifest
* Paper source
* Supplementary source

Final command:

```bash
uv run python scripts/reproduce_results.py \
  --manifest results/manifests/release_manifest.json
```

## 21. Luna execution rules

Place this in `AGENTS.md`:

```markdown
# SchemaGuard implementation rules

1. Read SCHEMAGUARD_MASTER_PLAN.md before changing code.
2. Implement exactly one numbered phase at a time.
3. Do not rename datasets, models, views, metrics, or outputs.
4. Do not change frozen configurations without a decision record.
5. Add tests before completing a phase.
6. Run the entire relevant test group after every implementation.
7. Stop when a blocking test fails.
8. Do not replace failures with missing values.
9. Do not manually edit generated predictions, metrics, tables, or figures.
10. All expensive operations must use content-addressed caching.
11. CPU jobs may use two workers.
12. GPU jobs must be sequential and protected by a file lock.
13. Every output must record its source hashes.
14. Test labels must never enter preprocessing fitting, COSA selection, or parameter selection.
15. At the end of each phase, create a semantic review file in artifacts/handoff/.
16. Do not begin the next phase until the semantic handoff review reports PASS.
```

Required Luna phase response:

````markdown
# Phase NN result

## Status
PASS | FAIL | BLOCKED

## Files created
- path
- path

## Files modified
- path
- path

## Commands executed
```text
exact commands
````

## Tests

| Test group | Passed | Failed | Skipped |
| ---------- | -----: | -----: | ------: |

## Artifacts

| Artifact | SHA-256 |
| -------- | ------- |

## Resource use

* Runtime:
* Peak RAM:
* Peak VRAM:

## Deviations

* None, or exact deviation and reason

## Blocking failures

* None, or exact failure

````

## 22. User validation procedure

For every completed phase, collect these files:

```text
artifacts/handoff/split_generation_review.md
results/validation/validation_summary.json
results/manifests/run_manifest.json
uv.lock
git_diff.patch
pytest_output.txt
````

For pilot and main phases, also collect:

```text
results/statistics/analysis_dataset.parquet
results/metrics/condition_metrics.parquet
results/resources/resource_records.parquet
artifacts/figures/
artifacts/tables/
```

Run:

```bash
git status --short
git diff --stat
uv run ruff check .
uv run mypy src/schemaguard
uv run pytest -q
uv run python scripts/validate_pilot.py
```

For the main experiment:

```bash
uv run python scripts/reproduce_results.py \
  --manifest results/manifests/release_manifest.json
```

The review package supplied for independent validation must contain:

1. Frozen configurations.
2. Dependency lock.
3. Dataset registry.
4. Condition manifest.
5. Validation summary.
6. Prediction and metric Parquet files.
7. Statistical analysis table.
8. Resource records.
9. Figures.
10. Tables.
11. Exact executed commands.
12. Git commit hash.

Screenshots and verbal result summaries are insufficient. Raw Parquet, JSON, YAML, logs, and hashes are required.

## 23. Final viability decision

The revised idea remains viable only under the narrowed claim.

Defensible claim:

> Modern end-to-end tabular pipelines can violate prediction consistency under certified lossless schema migrations; SchemaGuard measures these violations and restores consistency through a canonical normal form and bounded orbit aggregation.

Undefendable claims:

* First study of permutation invariance.
* First use of metamorphic testing for classifiers.
* Universal invariance across arbitrary schemas.
* Automatic recovery of semantics without metadata.
* Guaranteed IJCAI acceptance.
* Exact GPU determinism.
* Robustness to lossy corruption, missingness, adversarial examples, or distribution shift.

The project is strong only if the pilot establishes a substantial non-permutation effect and SchemaGuard beats SCNF and uniform orbit averaging.
