# SchemaGuard Representation-Sensitivity Pilot Protocol Review

Status: `RESOURCE_REVIEW_REQUIRED`

Protocol base commit: `9a59e5a5dda320ec00f1915c80fb82d2531cb442`
Delivery commit: the Git commit containing this review; its exact hash and verified Actions run are recorded in the delivery response.

This is a plan-only protocol. No pilot model fit, prediction, pilot metric, or pilot test-label access occurred.
The private root `.docx` was not opened or modified and remains untracked.

## Frozen design

- Selected datasets: 3, 23, 29, 36, 37, 38, 46, 1464; anchor 1464 included.
- Excluded registry datasets: 31, 44, 50, 54, 1067, 1489; each has a metadata-only rationale.
- Seeds: 1729, 2718, 31415; 24 selected dataset-seed pairs bind passing grouped splits.
- Views: 11; planned tuples 264, applicable 192, controlled N/A 72.
- Model conditions: 960 of a maximum 1320; CPU 576, CUDA 384.
- Protocol SHA-256: `5dad78c35a332077c3790e0eac8fc355adf48d93bd8e9f815d23636c9d7f3fe9`.
- Condition inventory SHA-256: `29ed6759ada3b9f23302288161f54e57a5e49ef538a1023565cb5780509d4e76`.

## Resource estimate

- Conservative wall time: 75040.1 s; preferred limit 28800 s; hard limit 43200 s.
- Prediction storage: 261605640 bytes; cache allowance 523211280 bytes; preferred storage 21474836480 bytes.
- Staged schedule required: `True`; full condition matrix remains frozen.
- Staged schedule SHA-256: `6291534aec5fd156e3e8c2f9d494b69700ccb7b9efb7e0e1fba6c6029e75c4b8`; 4 deterministic dataset-pair stages; cumulative staged estimate 75220.1 s (20.89 h); each stage is within the 43200-second hard limit: `True`. The cumulative full-matrix estimate still requires resource review; staging does not reduce total work.
| Stage | Dataset IDs | Conditions (CPU/CUDA) | Conservative wall | Hard limit |
| --- | --- | ---: | ---: | ---: |
| dataset_group_3_23 | 3, 23 | 255 (153 / 102) | 15463.4 s | 43200 s |
| dataset_group_29_36 | 29, 36 | 255 (153 / 102) | 22655.5 s | 43200 s |
| dataset_group_37_38 | 37, 38 | 255 (153 / 102) | 17954.7 s | 43200 s |
| dataset_group_46_1464 | 46, 1464 | 195 (117 / 78) | 19146.5 s | 43200 s |

## Acceptance gates

Passed: 40; failed: 0; not verified: 0.

| Gate | Result | Evidence SHA-256 | Detail |
| --- | --- | --- | --- |
| PF01 | PASS | `91256050cec26262b25ed0631d4195eeab229ffe87621c93575064693955afd9` | repository must remain on the authorized base commit before delivery |
| PF02 | PASS | `dfc087c7bc3d8cebec7266521594cc944021f9e8b704307a3ed98ecd53388619` | accepted independent smoke identity and no-pilot flag are bound by source hash |
| PF03 | PASS | `0fd6d13270e4bc9bab907bec8f5e1dbcd71d7d46b32ed3f3f39e1826015e951a` | content-addressed protected-tree snapshot is unchanged through validation; pre-existing tracked schemas match the authorized base |
| PF04 | PASS | `745b70d70edf457aa035956bb5f96fb036bd084bc9e429da28892aff3601c5ed` | the closed SchemaOrbit-14 registry and complete accepted 70-split inventory validate |
| PF05 | PASS | `ad4e02d93bf3dae53c94e219694a7b5dcd81315f088a9cb73371530c2cbdfefd` | rebuilding from sanitized registry metadata produces identical selected IDs and hashes |
| PF06 | PASS | `f10ceac14967a26befbb55a9e9b0a033c7ffe92f4faa3a589bce67d57b41cda6` | eight-dataset cohort covers every frozen schema/task/size/balance feature and explains six exclusions |
| PF07 | PASS | `771f240a579c3b7f3aa690c2e2b3939906251ad331cdd0553a9d20b13e58b71b` | the first three canonical configured seeds include 1729 and contain no generated seed |
| PF08 | PASS | `5a447d3ef06b3c1591c690cd91b47b7073311d66ef1e5ab00a55295eee42be04` | all 24 selected dataset-seed pairs bind passing grouped assignments with zero crossing groups |
| PF09 | PASS | `403ba3b10778b656fd8729600fc6c29ccc1fdcfbfcbd46769c6cd0f430fcc355` | all eleven view IDs and names are bound to the frozen transformation configuration |
| PF10 | PASS | `91ea912d12b9a7a620e259a739f9bcfe640767cf339cea228d86653c094f2368` | applicable and controlled NOT_APPLICABLE states cover every tuple; V00 is always applicable |
| PF11 | PASS | `2b9212b1eab3a0137a6864c46f8770f45cdf7001391f94523006334f6d3a8018` | the five accepted frozen model IDs retain canonical order and execution devices |
| PF12 | PASS | `66afea8eb8493334b7736a24570cc71a203915709c42db1f5c056c8def408f05` | exact package versions, checkpoint identifiers, and frozen checkpoint hashes are bound |
| PF13 | PASS | `20929b27807734f91892e239e12b0e311b480f89980413603e69c6c322d13b67` | condition inventory contains exactly five conditions per applicable view tuple |
| PF14 | PASS | `872ed8b57e07ebe939241121f0ae8657fe9a3855f5a38ae06a2fedc9740b8643` | all condition IDs are unique canonical hashes of the complete identity payload |
| PF15 | PASS | `2028bfa221953e2e8a54c36d506e87a8ef88899f221c2245bb91805b2d9af436` | frozen leakage policy confines fitting to training rows and orders evaluation events |
| PF16 | PASS | `6f1e024e4b7a7de9aa7607fa11bc7d2989b09560d0a63282d9735e591971cb03` | planning remains outcome-blind; label opening occurs only after structural prediction validation |
| PF17 | PASS | `169be88ecfb3bac9ed4b033e2aa5513de9842f5391644d39fe48c023cfec4e1c` | primary Brier degradation, relative epsilon, SII, flips, and undefined-metric policy are frozen |
| PF18 | PASS | `a4c32de77624291034ce4866988459457354f90b1047c5463741f5a13bcbc6a6` | raw and tolerance-aware AUROC use chunked comparisons and explicit numerical-tie handling |
| PF19 | PASS | `df1dba9328d14809ae4d1b2c99cee1976209efe200ab29c1d9ccdddeca597769` | seed repeats are summarized within datasets; datasets remain the cross-dataset unit |
| PF20 | PASS | `b1f0c61812972db671f0debe2e1f581fe768206f9f92073fe9180fbea4083958` | nontrivial-effect, breadth, residual-preprocessing, comparison, and six outcomes are frozen |
| PF21 | PASS | `8bc79fb07923a5f4b6eaf6d21e72d47ab3cb3895e7799ae678fb8a02f9492800` | retryable failures, attempt limits, failed-cache prohibition, and same-identity resume are frozen |
| PF22 | PASS | `53fd71486faa1b595307c5e3e2116297fd325c2d156114300087a302419c222a` | two CPU workers, one sequential GPU worker, 3600 MiB VRAM cap, and offline execution are frozen |
| PF23 | PASS | `5a6e64040b9425296b189be9987a0ad6423a335e4e0e2d644dbe1851b13f11b7` | runtime estimate and deterministic staged schedule cover every planned condition; each stage fits the hard wall limit |
| PF24 | PASS | `366d75501802a95b478b4480ec57f5404bb5c6aee4feeeb0061d900ca3efdd87` | storage estimate includes row IDs, probabilities, metadata, headers, and cache allowance |
| PF25 | PASS | `ffbfd125488f20cb09e7848f065e2a35d250c0796dfedfb10df187e1e8afd38d` | protocol, dataset, condition, inventory, metric, decision, and validation contracts are strict and immutable |
| PF26 | PASS | `ea7b3a35309cc371efcc0379e45eae9b42ae2b9c22a14098886955289cc6f427` | generated contract schemas exactly match current Pydantic contracts |
| PF27 | PASS | `95eab200114bb217d934e64f5d570537b224676e0aab92f3618b2b061cb57ceb` | non-network, non-GPU, non-foundation-model unit tests passed (exit 0) |
| PF28 | PASS | `8448834db780643401163451ed2d2a00b4d460c0cff933c634db1d447d0aec4e` | non-network integration tests; pilot plan test is metadata-only passed (exit 0) |
| PF29 | PASS | `81128cb8e8fd854f1d623b25b4329356e78976a71b93a34571d1d29eea92236b` | Ruff passed (exit 0) |
| PF30 | PASS | `255e432db5c7e741bea44ffb7fc5337dd36a63e77d9c629a58ebf9ab40ccbf04` | Mypy passed (exit 0) |
| PF31 | PASS | `6cf5aca2145c6cad9b23bf66d89e0c6c167e964f3e98a1be717b23154e406fa9` | semantic repository naming validation passed (exit 0) |
| PF32 | PASS | `b0780499748bc84401d9145b3b8963c227c5f52323cbffea09e89f6fc25e9d00` | repository structural and local-evidence validation passed (exit 0) |
| PF33 | PASS | `d59b57af5064601d69b10dbe506f420e81523e8898273e7b90b69b2d2edea350` | all accepted split inventory records remain PASS with zero predictor-group crossings |
| PF34 | PASS | `975d813804ceff31f68a78b6a4aba6f3e24f81f0c81b4fac422189d881f86a71` | accepted transformation applicability inventory and implementation identity remain unchanged |
| PF35 | PASS | `91ba71612ec1d53d8bbdcc1fa86378b7f5bf699c56d5bdec2627ace713d35ad5` | all accepted adapter evidence remains passing and is identity-bound into the frozen model matrix |
| PF36 | PASS | `49151adc24245892d898f41d1960e1e888cf1aee0efa33c2901ea1127d50f28f` | accepted cache/scheduler inventory, 30 fault cases, and resume probe remain strictly valid |
| PF37 | PASS | `ef35178888d044e8363d1212cd3f771e26060d954a5b7a58629beb5db25b119c` | accepted smoke protocol and run evidence remain byte-identical and explicitly report no pilot |
| PF38 | PASS | `105252ebe8cd977e9ac9c9eff0141ab803ccb45148d321de8bf9102a1381c6d2` | source, schemas, and focused protocol tests pass in a temporary local clean clone without local datasets or private files |
| PF39 | PASS | `e9fa8cb7dbc22f4006be7e07495d51f7f19222b5930a096e3a527694f6286686` | only a plan and condition manifest exist; no pilot fit, prediction, or pilot metric output was created |
| PF40 | PASS | `dbb0eb38be36eeb5aa73e7d81b506ceafd34c3e4ba214d080ef26a7fec9e1906` | private root reference document was not opened or modified and remains untracked |

## Quality command results

Commands ran with the P12 interpreter; full raw output was not embedded in this tracked handoff.
- `unit_tests`: exit 0; output SHA-256 `a1bad6ab23b9c839447aa6094a6f825f3b745a505e465457320cac0e3caa618b`; command `D:\Conda\P12\python.exe -m pytest -o addopts= -q -m not integration and not network and not gpu and not foundation_model -rA`.
- `integration_tests`: exit 0; output SHA-256 `60235d7b594eb81164e0953fb5c25cca5dc48dff4eb37b16a9d96952cdfa67f8`; command `D:\Conda\P12\python.exe -m pytest -o addopts= -q -m integration and not network and not gpu and not foundation_model -rA`.
- `ruff`: exit 0; output SHA-256 `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18`; command `D:\Conda\P12\python.exe -m ruff check .`.
- `mypy`: exit 0; output SHA-256 `a7c4aae49ff9d21cbf4edf1b955366f4cb545b2dbf41b63329a9f1c16b3b0c5d`; command `D:\Conda\P12\python.exe -m mypy src/schemaguard`.
- `naming`: exit 0; output SHA-256 `e23d637605363cef935aea29e38cb53219836c64fd7926223c496cf0bc5cd12a`; command `D:\Conda\P12\python.exe scripts/validate_repository_naming.py`.
- `repository`: exit 0; output SHA-256 `3aa38f78a4a87d3eb34b43a8951a0441c368b20973a6381f0db6e56bed900ec7`; command `D:\Conda\P12\python.exe scripts/validate_repository_repair.py --local-evidence`.

## Preservation and portability

- Protected local evidence snapshot unchanged during validation: `True`; before/after identity `dc02ae3f16feb1b24bd9c77d285f3860c4c4b792ed7fd07db3def42c64edc3b4` / `dc02ae3f16feb1b24bd9c77d285f3860c4c4b792ed7fd07db3def42c64edc3b4`.
- Clean-clone focused tests: `True`; overlay file count 24.
- The planner validates source/processed/target hashes and accepted split assignment hashes using manifests and byte hashing only; it does not decode target rows or open pilot test labels.
- Next permitted workstream: independent pilot-protocol review. Pilot execution remains unauthorized until that review accepts this handoff.

## Frozen output files

- `configs/runtime/pilot_protocol.yaml`
- `src/schemaguard/experiments/pilot_contracts.py`
- `src/schemaguard/experiments/pilot_planning.py`
- `src/schemaguard/experiments/pilot_validation.py`
- `scripts/freeze_pilot_protocol.py`
- `scripts/validate_pilot_protocol.py`
- Eight generated schemas under `schemas/`.
- `artifacts/handoff/pilot_protocol.json`
- `artifacts/handoff/pilot_condition_inventory.json`
- `artifacts/handoff/pilot_staged_schedule.json`
- `artifacts/handoff/pilot_protocol_validation.json`
- `artifacts/handoff/pilot_protocol_review.md`

No data, split, transformation, adapter, scheduler, or smoke evidence was rewritten.


## Complete selection and model record

### Selected datasets

| ID | Name | Task | Rows | Features N/C | Coverage | Rationale |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 3 | kr-vs-kp | binary | 3196 | 0/36 | balanced, binary, categorical_only, medium, substantial_categorical | Selected by the metadata-only maximum-coverage objective; contributes balanced, binary, categorical_only, medium, substantial_categorical while minimizing duplicate task/schema/size/balance profiles. |
| 23 | cmc | multiclass | 1473 | 2/7 | balanced, medium, mixed_numeric_categorical, multiclass, substantial_categorical | Selected by the metadata-only maximum-coverage objective; contributes balanced, medium, mixed_numeric_categorical, multiclass, substantial_categorical while minimizing duplicate task/schema/size/balance profiles. |
| 29 | credit-approval | binary | 690 | 6/9 | balanced, binary, missing_values, mixed_numeric_categorical, small, substantial_categorical | Selected by the metadata-only maximum-coverage objective; contributes balanced, binary, missing_values, mixed_numeric_categorical, small, substantial_categorical while minimizing duplicate task/schema/size/balance profiles. |
| 36 | segment | multiclass | 2310 | 19/0 | balanced, medium, multiclass, numeric_only | Selected by the metadata-only maximum-coverage objective; contributes balanced, medium, multiclass, numeric_only while minimizing duplicate task/schema/size/balance profiles. |
| 37 | diabetes | binary | 768 | 8/0 | balanced, binary, numeric_only, small | Selected by the metadata-only maximum-coverage objective; contributes balanced, binary, numeric_only, small while minimizing duplicate task/schema/size/balance profiles. |
| 38 | sick | binary | 3772 | 7/22 | binary, imbalanced, medium, missing_values, mixed_numeric_categorical, substantial_categorical | Selected by the metadata-only maximum-coverage objective; contributes binary, imbalanced, medium, missing_values, mixed_numeric_categorical, substantial_categorical while minimizing duplicate task/schema/size/balance profiles. |
| 46 | splice | multiclass | 3190 | 0/60 | balanced, categorical_only, high_cardinality_categorical, high_dimensional, medium, multiclass, substantial_categorical | Selected by the metadata-only maximum-coverage objective; contributes balanced, categorical_only, high_cardinality_categorical, high_dimensional, medium, multiclass, substantial_categorical while minimizing duplicate task/schema/size/balance profiles. |
| 1464 | blood-transfusion-service-center | binary | 748 | 4/0 | binary, imbalanced, numeric_only, small | Protected SchemaGuard anchor; preserves the existing small numerical binary and grouped-split lineage without using smoke outcomes for selection. |

| Dataset ID | Raw SHA-256 | Features SHA-256 | Targets SHA-256 | Quality SHA-256 | Metadata SHA-256 |
| ---: | --- | --- | --- | --- | --- |
| 3 | `b22a8a12bd40648400b000bad0545683b8ecd33ac84f2c265dd5c309c832fd69` | `5f00b895df526f1fb9e5e08e501bfaf98962d3d0144a8ecbcc4658fc30c23fa8` | `39d6f903f648012a651dde9f540df2deb4908be9895dbf050a6fa077a700fc87` | `df2a460a21b29a2283813a53443ede51dd3643fb92d3afeb0875678363ee95be` | `7ec46a1746cdac796e77c994a807dfdafd26ba9777bce2868ad5e972ea8ab2ed` |
| 23 | `9d8694d39e05d9d2963fccbe4f71e2af04c72c13de45f5dcbe775a586f214567` | `6b3dddb626c42d0cb85cd22ef4ad83ab161a5fe27621e4418c2b794324282ed2` | `3d4a15c1ba0b651b6c6ee25bf78d920b42ee2c95b119fb584c9a6193dd435283` | `580238408cd13aade23e39a8d4cb6bf968896dd6c050eeaf86af258c3be9c0f6` | `166b1a90ff7d6e1d7b3e026efb81135dfa63bcadf545ae2536097e4dcad93baa` |
| 29 | `ae8e5eadb70e187885ddc6defb7f3e4afa726aa472ac63ebf5e387f0a8467891` | `0d2989443ecfe78a90c67548fd392dc0be2da5159fa1c16f49dc9cf347ca37f4` | `8288e72fe060dcec77aa7e7b2c85ccc2acbb5a6f782e49c4e02a8b7ae6ceaeac` | `a3f5d3347da6909d4ac17d8013d3ab4c6a939b1ae7d24b13e5eb1be257f56a27` | `34a81aac0d6278238bb0dc4e6aa93f6a8f62a33ac3f5d80adef5f80ea53010f5` |
| 36 | `f54038600f6b8ab6cb6edf58887ba98c891781250b4140f5edeb74296e466854` | `cc062c157e8bf98531e74d57c1fec2dd2bf45ce7215f78b04a3948a771fa7e96` | `38cfb543da007ce09183b0d304bcc359ef6ac4b208531387db41761dd8af3d93` | `075396bdeb071f1970364c06e7329acaae825cfa484711124d1f5ea494b78430` | `976926846cc7c32f6dacb3bf95b51128c267bdb44ec5e809d7c610b8affba3bb` |
| 37 | `4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40` | `e789c4527c8e493298b8ece6ad7663cfb89c9384dfa43dd8d357f5fecf559746` | `264d703ed9cd6429588872ccaef067d67ff1288b78b7f627579d1e3ff4b52967` | `800e5140cf3ca34eacb91c2c0692dd0badfac8a631ce5833675d090082cfbfc6` | `7caa393593f1ce9b80a2b8afd487a85c8c900eee13723149e4a8f75ec257eaa3` |
| 38 | `85c39615380ca4c05c17a53c3faad9523924958d1e9bc5af0dcbefc6339e7800` | `b78facfdc50687d1b243b4132c63652991d0f4cc2100fbc8857b90cffc2d63de` | `d07a486c9140898254347d5e9500221a6692fa94083517f56fe4f041791e1f76` | `45098275cacdcbefe74bd2254cf9ed894e93e170fb43668da5e5068f05cd3efd` | `03a7ae949296c0194d8264ce079e3901ab3b55b403dcadd84da5f0cad2dff25b` |
| 46 | `4b22fd2fc0564f0b5af59a5a40ee0dd77476835ba48b717960d85d095be53036` | `f564525a798b5b90e7662de96f783aa8a253a4a459e1fd98efc1eac91446b4f9` | `2a2fc561d27a55ebeb9cf3f44e11f05dfd79380ea68fb66afe3653602bf13688` | `0db1e722947cbf7b809e3744aa8dff97df05b76b3a7e964038c06ccfbc1f7410` | `3d0134a49bee23fed19d1548db93cd0ccfc4ec7d034591fa0e4e03bdf842a44d` |
| 1464 | `ee1304cac4a650ac31afe7395a100536a165f935bbe718a4643757c1a842316a` | `9d3168b0bafc5af610caea42c96f95033f32bf5ae4dc011797f45d1809af458f` | `9d4e0f665127dd2c41e990d4b129e08d8a0c55f83d42d2e57b758959a9fc23d0` | `47e2c0bf256453278a0bec96099fc980f972fb9661c914a71818f920546f9e04` | `1dbcb2e287b890b951931bd800f03bc7e8af2bda0c90cdf94ae0db2130bee702` |

### Excluded datasets

| ID | Name | Reason | Closest selected dataset | Evidence SHA-256 |
| ---: | --- | --- | ---: | --- |
| 31 | blood-transfusion-service-center | Valid registered candidate, but deterministic maximum-coverage selection favors complementary strata; nearest selected metadata profile is dataset 29. No outcome or transformation result was used. | 29 | `1dbcb2e287b890b951931bd800f03bc7e8af2bda0c90cdf94ae0db2130bee702` |
| 44 | blood-transfusion-service-center | Valid registered candidate, but deterministic maximum-coverage selection favors complementary strata; nearest selected metadata profile is dataset 3. No outcome or transformation result was used. | 3 | `1dbcb2e287b890b951931bd800f03bc7e8af2bda0c90cdf94ae0db2130bee702` |
| 50 | blood-transfusion-service-center | Valid registered candidate, but deterministic maximum-coverage selection favors complementary strata; nearest selected metadata profile is dataset 3. No outcome or transformation result was used. | 3 | `1dbcb2e287b890b951931bd800f03bc7e8af2bda0c90cdf94ae0db2130bee702` |
| 54 | blood-transfusion-service-center | Valid registered candidate, but deterministic maximum-coverage selection favors complementary strata; nearest selected metadata profile is dataset 36. No outcome or transformation result was used. | 36 | `1dbcb2e287b890b951931bd800f03bc7e8af2bda0c90cdf94ae0db2130bee702` |
| 1067 | blood-transfusion-service-center | Valid registered candidate, but deterministic maximum-coverage selection favors complementary strata; nearest selected metadata profile is dataset 38. No outcome or transformation result was used. | 38 | `1dbcb2e287b890b951931bd800f03bc7e8af2bda0c90cdf94ae0db2130bee702` |
| 1489 | blood-transfusion-service-center | Valid registered candidate, but deterministic maximum-coverage selection favors complementary strata; nearest selected metadata profile is dataset 38. No outcome or transformation result was used. | 38 | `1dbcb2e287b890b951931bd800f03bc7e8af2bda0c90cdf94ae0db2130bee702` |

### Exact view registry

| View | Name | Family | Certificate | Role |
| --- | --- | --- | --- | --- |
| V00 | identity | Identity control | BIJECTION | CONTROL |
| V01 | numeric_affine_units | Numeric affine | BIJECTION | PRIMARY_MIGRATION |
| V02 | numeric_asinh | Strictly monotone piecewise numeric mapping | BIJECTION | PRIMARY_MIGRATION |
| V03 | category_permutation | Category permutation | BIJECTION | PRIMARY_MIGRATION |
| V04 | categorical_onehot | Reversible categorical one-hot representation | BIJECTION | PRIMARY_MIGRATION |
| V05 | duplicate_feature | Duplicate feature | PROJECTION | PRIMARY_MIGRATION |
| V06 | redundant_affine_feature | Redundant affine feature | PROJECTION | PRIMARY_MIGRATION |
| V07 | integer_quotient_remainder | Reversible integer split | BIJECTION | PRIMARY_MIGRATION |
| V08 | column_permutation_control | Column permutation | PERMUTATION | CONTROL |
| V09 | row_permutation_control | Row permutation | PERMUTATION | CONTROL |
| V10 | composite_migration | Two-transform composition | COMPOSITION | PRIMARY_MIGRATION |

### Exact model matrix

| ID | Model class | Package/version | Device | Preprocessing | Precision/batch | Checkpoint identifier and SHA-256 | Model-config SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LR-1.9 | sklearn.linear_model.LogisticRegression | scikit-learn/1.9.1 | cpu | common_onehot_standard | estimator_native/adapter_full_partition | None | `04d6098c32387189d2fe5b99ee54c44ee5a7f8d0fa55d99366c4c682a7b41f06` |
  Frozen parameters: `{"C":1.0,"class_weight":null,"max_iter":2000,"penalty":"l2","solver":"lbfgs","tol":1e-06}`
| CAT-1.2 | catboost.CatBoostClassifier | catboost/1.2.10 | cpu | native_catboost | estimator_native/adapter_full_partition | None | `b3d309ec73d88e2c34c1b0168593344d0ed5c3cae1407ed798573259032ffcc2` |
  Frozen parameters: `{"allow_writing_files":false,"bootstrap_type":"No","depth":6,"iterations":500,"l2_leaf_reg":3.0,"learning_rate":0.05,"loss_function":"auto","random_strength":0.0,"task_type":"CPU","thread_count":2,"verbose":false}`
| XGB-3.4 | xgboost.XGBClassifier | xgboost/3.4.1 | cpu | common_onehot_no_scaling | estimator_native/adapter_full_partition | None | `b2616410ba0906e1089e5b18138686bb022ab3e537fb74c8f002139c98de2203` |
  Frozen parameters: `{"colsample_bytree":1.0,"learning_rate":0.05,"max_depth":6,"min_child_weight":1,"n_estimators":500,"n_jobs":2,"reg_alpha":0.0,"reg_lambda":1.0,"subsample":1.0,"tree_method":"hist","verbosity":0}`
| TPFN3-8.5 | tabpfn.TabPFNClassifier | tabpfn/8.5.0 | cuda | native_tabpfn | inference_precision_auto/adapter_frozen_default | tabpfn-v3-classifier-v3_default.ckpt / d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988 | `7b6c249ee0fc37d904c9b3c965d79d0f69388704b44188e8d23fd467f6926024` |
  Frozen parameters: `{"auto_scale_n_estimators":false,"fit_mode":"low_memory","ignore_pretraining_limits":false,"inference_precision":"auto","memory_saving_mode":true,"n_estimators":8,"n_preprocessing_jobs":1,"show_progress_bar":false}`
| TICL2-2.2 | tabicl.TabICLClassifier | tabicl/2.2.0 | cuda | native_tabicl | amp_auto/batch_size_one | tabicl-classifier-v2-20260212.ckpt / bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0 | `8d7335e579e4d945fcb173ca0f9ca160c9bbcf72f8cd7e43cf1f4482ecb0aa50` |
  Frozen parameters: `{"average_logits":true,"batch_size":1,"checkpoint_version":"tabicl-classifier-v2-20260212.ckpt","class_shuffle_method":"shift","feat_shuffle_method":"latin","kv_cache":false,"n_estimators":8,"n_jobs":2,"offload_mode":"auto","use_amp":"auto"}`

## Exact condition counts

Seeds: 1729, 2718, 31415.
Applicable view tuples: 192; controlled NOT_APPLICABLE tuples: 72; total: 264.
Model conditions: 960; CPU: 576; CUDA: 384; maximum: 1320.
Protocol SHA-256: `5dad78c35a332077c3790e0eac8fc355adf48d93bd8e9f815d23636c9d7f3fe9`.
Condition-inventory SHA-256: `29ed6759ada3b9f23302288161f54e57a5e49ef538a1023565cb5780509d4e76`.

| Dimension | Key | Condition count |
| --- | --- | ---: |
| Dataset | 1464 | 105 |
| Dataset | 23 | 165 |
| Dataset | 29 | 150 |
| Dataset | 3 | 90 |
| Dataset | 36 | 105 |
| Dataset | 37 | 105 |
| Dataset | 38 | 150 |
| Dataset | 46 | 90 |
| Seed | 1729 | 320 |
| Seed | 2718 | 320 |
| Seed | 31415 | 320 |
| View | V00 | 120 |
| View | V01 | 90 |
| View | V02 | 90 |
| View | V03 | 75 |
| View | V04 | 75 |
| View | V05 | 120 |
| View | V06 | 90 |
| View | V07 | 15 |
| View | V08 | 120 |
| View | V09 | 120 |
| View | V10 | 45 |
| Model | CAT-1.2 | 192 |
| Model | LR-1.9 | 192 |
| Model | TICL2-2.2 | 192 |
| Model | TPFN3-8.5 | 192 |
| Model | XGB-3.4 | 192 |

## Bound inputs and source hashes

| Repository-relative input | SHA-256 |
| --- | --- |
| `artifacts/handoff/cache_scheduler_fault_evidence.json` | `b7f25cc9b6a12f7fadeab4742f3f58411a70b242490e336df28d16ab33049dd2` |
| `artifacts/handoff/cache_scheduler_inventory.json` | `0b697c8f484b73c09413eaa7f266c5900a531ed3a896c3135abc5f6d5517b42f` |
| `artifacts/handoff/cache_scheduler_probe_evidence.json` | `7908fde03dc6618ad37ee7ddece7acc256b82c7f9988465bdb34b09471aef743` |
| `artifacts/handoff/cache_scheduler_protected_hash_comparison.json` | `e3a1c059eeeebf0f5c0bcc298cd07e6c35a6c2ff29a95306d39fb08810aa6192` |
| `artifacts/handoff/model_adapter_inventory.json` | `119ea901a3b81de726b154af1ac4b8601605139bc82b5593eaf7787427a42b56` |
| `artifacts/handoff/model_adapter_leakage_evidence.json` | `8992d9286e68da8c19dbe1bf6e2eec295495b725cf3b5406f785e8ea8212bf78` |
| `artifacts/handoff/smoke_experiment_inventory.json` | `2d171556316e41699acec04f7e162cd0b23a91a48dbb9fa3813f5e8c5d9915fd` |
| `artifacts/handoff/smoke_independent_review.json` | `28c5e0d1cadfc8f7835786526728942e3eba94ec95eb88fad0fdc1ee01d5478a` |
| `artifacts/handoff/smoke_independent_review.schema.json` | `9a4b889808d2b74b607f43f6b5b2654035f0b709aefb9323ca9cf541e612b799` |
| `artifacts/handoff/split_generation_inventory.json` | `368acbd6ecaffafbecf6b5bdbce301211da4902ceaea05fbfb0db933a3611482` |
| `artifacts/handoff/transformation_inventory.json` | `6984ecef7342a0b7c6190f4636ccc69684e90a7a0343e88fc686362dd9ada104` |
| `configs/datasets/schemaorbit14.yaml` | `2ca5f63308eef9e94907a07cedfd60ae5ad96dac90c544517f73c8bde5ab75fa` |
| `configs/experiment_registry.yaml` | `3afded17158dee3b913c03189956f4dbc47fa5c1b7763354e4a3ef20b8474f65` |
| `configs/runtime/model_adapters.yaml` | `11dacd74b88ed29b93fab1e18aa9eef8c5375eaac386321d83a2313eb5b28739` |
| `configs/runtime/model_compatibility.yaml` | `e482566d7398c893d37e98a578a6f1b8c89324ad91988994211b7a42a8756c34` |
| `configs/runtime/pilot_protocol.yaml` | `ad8b680efd0d135d9d31a800234a134253b55d47901bd5910e3fd3ad8416ca7d` |
| `configs/runtime/split_generation.yaml` | `b963790d00086db0e96a46a24f37731493a61fcad953d44fc8cc8280a40b7811` |
| `configs/runtime/transformation_engine.yaml` | `b6c5aa5c08397904548f5d177f5bd7e2195a02f4fee9ecf5a2b2b8146767f6f2` |
| `data/processed/openml/1464/data_manifest.json` | `c1bfecd0545ce173e35d2123c6c082df38a21aff80bdfac3e15ba73f64d0dedf` |
| `data/processed/openml/1464/features.parquet` | `9d3168b0bafc5af610caea42c96f95033f32bf5ae4dc011797f45d1809af458f` |
| `data/processed/openml/1464/quality_report.json` | `47e2c0bf256453278a0bec96099fc980f972fb9661c914a71818f920546f9e04` |
| `data/processed/openml/1464/schema.json` | `cba51d40f1ecd2f34b4b83e4d55a6a134708d4d1a4a11d6ed263b161792d4e9c` |
| `data/processed/openml/1464/targets.parquet` | `9d4e0f665127dd2c41e990d4b129e08d8a0c55f83d42d2e57b758959a9fc23d0` |
| `data/processed/openml/23/data_manifest.json` | `c36b783ed032c10ae9129936152c9c1c96ed6eb712fd8d28c2692c7e7d91e8ec` |
| `data/processed/openml/23/features.parquet` | `6b3dddb626c42d0cb85cd22ef4ad83ab161a5fe27621e4418c2b794324282ed2` |
| `data/processed/openml/23/quality_report.json` | `580238408cd13aade23e39a8d4cb6bf968896dd6c050eeaf86af258c3be9c0f6` |
| `data/processed/openml/23/schema.json` | `9eb94452afdded2055e839fab4399f1f644091cd03999231ee1a1d6e0035194e` |
| `data/processed/openml/23/targets.parquet` | `3d4a15c1ba0b651b6c6ee25bf78d920b42ee2c95b119fb584c9a6193dd435283` |
| `data/processed/openml/29/data_manifest.json` | `86adaf2fe45a03a73f4b2cc7ce2d7585deab66c3af7bc2e60bf5687e46e775d9` |
| `data/processed/openml/29/features.parquet` | `0d2989443ecfe78a90c67548fd392dc0be2da5159fa1c16f49dc9cf347ca37f4` |
| `data/processed/openml/29/quality_report.json` | `a3f5d3347da6909d4ac17d8013d3ab4c6a939b1ae7d24b13e5eb1be257f56a27` |
| `data/processed/openml/29/schema.json` | `3a8d37db9ae65a35e3af6afe2fc8a08d6bd21fbd258294693bef9a602e683865` |
| `data/processed/openml/29/targets.parquet` | `8288e72fe060dcec77aa7e7b2c85ccc2acbb5a6f782e49c4e02a8b7ae6ceaeac` |
| `data/processed/openml/3/data_manifest.json` | `1dd87ca84547e024aa52072a13b0d5ecd974c5e4fd664c71121a60ea28036b82` |
| `data/processed/openml/3/features.parquet` | `5f00b895df526f1fb9e5e08e501bfaf98962d3d0144a8ecbcc4658fc30c23fa8` |
| `data/processed/openml/3/quality_report.json` | `df2a460a21b29a2283813a53443ede51dd3643fb92d3afeb0875678363ee95be` |
| `data/processed/openml/3/schema.json` | `54c4d6dbd326de322e5de185c7c14ac5b45d1e7fc9fa5bf191a8c4c212eea298` |
| `data/processed/openml/3/targets.parquet` | `39d6f903f648012a651dde9f540df2deb4908be9895dbf050a6fa077a700fc87` |
| `data/processed/openml/36/data_manifest.json` | `e006e030c3584fd8cc356a864da40cc569e26ff5a1a854ca0cbdb98be636a03e` |
| `data/processed/openml/36/features.parquet` | `cc062c157e8bf98531e74d57c1fec2dd2bf45ce7215f78b04a3948a771fa7e96` |
| `data/processed/openml/36/quality_report.json` | `075396bdeb071f1970364c06e7329acaae825cfa484711124d1f5ea494b78430` |
| `data/processed/openml/36/schema.json` | `03f0610a767ac349ce5526b8e5abffec595558baca5ededb6b55cbc6b4575b01` |
| `data/processed/openml/36/targets.parquet` | `38cfb543da007ce09183b0d304bcc359ef6ac4b208531387db41761dd8af3d93` |
| `data/processed/openml/37/data_manifest.json` | `f923209d3ff32be62351d04d11c85238d20fc0a5d8ec39d9230ba139485856c7` |
| `data/processed/openml/37/features.parquet` | `e789c4527c8e493298b8ece6ad7663cfb89c9384dfa43dd8d357f5fecf559746` |
| `data/processed/openml/37/quality_report.json` | `800e5140cf3ca34eacb91c2c0692dd0badfac8a631ce5833675d090082cfbfc6` |
| `data/processed/openml/37/schema.json` | `eb7f30ba0064d0996e961ab77a1c51df04cf8acccce8d1ffc4c49c4ddd294d2e` |
| `data/processed/openml/37/targets.parquet` | `264d703ed9cd6429588872ccaef067d67ff1288b78b7f627579d1e3ff4b52967` |
| `data/processed/openml/38/data_manifest.json` | `4b63331c4864b5a0dfb8e23fd17212e936039f0f1febfd28acde56bd4d3eca26` |
| `data/processed/openml/38/features.parquet` | `b78facfdc50687d1b243b4132c63652991d0f4cc2100fbc8857b90cffc2d63de` |
| `data/processed/openml/38/quality_report.json` | `45098275cacdcbefe74bd2254cf9ed894e93e170fb43668da5e5068f05cd3efd` |
| `data/processed/openml/38/schema.json` | `76d4a08768559d88a1a4bc32af8f2e031d39963e8cdc0126bae77075959eb121` |
| `data/processed/openml/38/targets.parquet` | `d07a486c9140898254347d5e9500221a6692fa94083517f56fe4f041791e1f76` |
| `data/processed/openml/46/data_manifest.json` | `892ec6dcd36fe79233aa703e7cb96e368e2494a6c98985de9152e01c18d92dc7` |
| `data/processed/openml/46/features.parquet` | `f564525a798b5b90e7662de96f783aa8a253a4a459e1fd98efc1eac91446b4f9` |
| `data/processed/openml/46/quality_report.json` | `0db1e722947cbf7b809e3744aa8dff97df05b76b3a7e964038c06ccfbc1f7410` |
| `data/processed/openml/46/schema.json` | `203f39daaf9ea1f03a1445bd1550be3bc12e1fa726d2806e7c05c40bf00c2c12` |
| `data/processed/openml/46/targets.parquet` | `2a2fc561d27a55ebeb9cf3f44e11f05dfd79380ea68fb66afe3653602bf13688` |
| `data/raw/openml/1464/blood-transfusion-service-center.arff` | `ee1304cac4a650ac31afe7395a100536a165f935bbe718a4643757c1a842316a` |
| `data/raw/openml/1464/source_manifest.json` | `4fbaedd74510c93364ba31ec92cfc83999de0f8ee7bcb40315d3e01b40efb755` |
| `data/raw/openml/23/cmc.arff` | `9d8694d39e05d9d2963fccbe4f71e2af04c72c13de45f5dcbe775a586f214567` |
| `data/raw/openml/23/source_manifest.json` | `e73c5a24c0a269b6c049b533b48e3746697353fe4b327371bd21159d22f442cb` |
| `data/raw/openml/29/credit-approval.arff` | `ae8e5eadb70e187885ddc6defb7f3e4afa726aa472ac63ebf5e387f0a8467891` |
| `data/raw/openml/29/source_manifest.json` | `78216ea8501c02e9586743dc1cb3f4b0f179737bb6e951ab9c18cbe38435822e` |
| `data/raw/openml/3/kr-vs-kp.arff` | `b22a8a12bd40648400b000bad0545683b8ecd33ac84f2c265dd5c309c832fd69` |
| `data/raw/openml/3/source_manifest.json` | `76ff9bdade77e926b4ff1143a25b51bfe7c54ef9aab16d4e98b8c3218456fb93` |
| `data/raw/openml/36/segment.arff` | `f54038600f6b8ab6cb6edf58887ba98c891781250b4140f5edeb74296e466854` |
| `data/raw/openml/36/source_manifest.json` | `3cba5c0e2bda360946c049a35347c0362504cc9f86e15a5dfaf798ad5ad6dfc0` |
| `data/raw/openml/37/diabetes.arff` | `4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40` |
| `data/raw/openml/37/source_manifest.json` | `06359bd8d9ebda54d1c4759f789d402a41a992fe063338c93465621f1ce6e0e7` |
| `data/raw/openml/38/sick.arff` | `85c39615380ca4c05c17a53c3faad9523924958d1e9bc5af0dcbefc6339e7800` |
| `data/raw/openml/38/source_manifest.json` | `878d23e2ba4fce0e73e29e6bb60863dcae3a2ff7abd18596b2cdc477fc62bb14` |
| `data/raw/openml/46/source_manifest.json` | `52e2e8aaf836018e95df851c2f01ed1318d6807788e7f48c2dd8920698c00c95` |
| `data/raw/openml/46/splice.arff` | `4b22fd2fc0564f0b5af59a5a40ee0dd77476835ba48b717960d85d095be53036` |
| `results/smoke/runs/de24211de43552175461d23fb08342c368fd9c3ff66c128b16c9011c3d99f7c6_cold.json` | `bc7fc0c72e3f597484b81f5462f95385bc5ccad23214b893b93639cbedb25d86` |
| `results/validation/checkpoint_inventory.json` | `8d21dba76c7937100bbd1b6ca04dd96046378e25c0f48b12c0492e8f61934204` |
| `results/validation/dataset_registry_report.json` | `97a71873bf80cfad008bd58a2b3fe635d63847739f54f49501569e333c87c849` |
| `results/validation/gpu_capacity_report.json` | `f7ac2cc00507c901a90761eecad469f4d65340114d32549e953c1e01c83677db` |
| `results/validation/license_inventory.json` | `002620c4a447ace5deada41e68f9a5b3d1c4b53174bd7270b39a4f6881a1a3e2` |
| `results/validation/model_compatibility_report.json` | `8eb6bdbd2203072dd9c493265ed0f12797a3398a7002d4c73badf865ee58feb3` |
| `scripts/freeze_pilot_protocol.py` | `4bee26705243290782c1e766cfa661e79b3e12fab5684edf7ab7c6818ec7adf5` |
| `scripts/validate_pilot_protocol.py` | `79185109691b2fe7eacebe5c18399adc2727ec64eb2852f0c527e8a55460a4af` |
| `src/schemaguard/cache/contracts.py` | `c8f61c8196ff1ff93ede5df44bd656e81ca033d743d7c3a0796fd2d064daa7c4` |
| `src/schemaguard/experiments/contracts.py` | `2e717a0929b4133e0ae0ceecfae84f0f47aa4a982a9b08bb6b78855e10f25da3` |
| `src/schemaguard/experiments/evaluation.py` | `2282888e853cb16836748ab61411b239d84ea8f901ceeac2dcf547b3f44eeb02` |
| `src/schemaguard/experiments/evidence.py` | `c282a16cda1703ae865b94a557a54cee0050062ce39859b1c9e586d8ab4bce30` |
| `src/schemaguard/experiments/execution.py` | `bae088afbacb5a15a37176c49216251f9ef661515b114f78aea8afdfc001239b` |
| `src/schemaguard/experiments/pilot_contracts.py` | `19df5eb02151ab5d23671ab6c02ccf6a70f10d767e5571e1a9efb377e2495ebf` |
| `src/schemaguard/experiments/pilot_planning.py` | `a9e56ecb857ef2dd5bb274c6fe02243ab930b5a843249d2bde06c9d13f44c4d2` |
| `src/schemaguard/experiments/pilot_validation.py` | `b724e616c846333f40cf9f474215b5f9cdb64e3a2877556c2ef69f8bf5b26e7e` |
| `src/schemaguard/experiments/planning.py` | `0946d36167abdddf3ca902d952e4a1aeb202a1ae0d44c55f9a4231f9b2dd97a4` |
| `src/schemaguard/models/adapters/catboost_adapter.py` | `b02cf47a48c97f16cb27118254476d62119321af0f2c023053c2529cc726a719` |
| `src/schemaguard/models/adapters/contracts.py` | `6003a309591d381519cd364bacde2e88ba8d2a5c07129007214324077647dc31` |
| `src/schemaguard/models/adapters/logistic_regression_adapter.py` | `a5c5c883157185c6738731dd1b95b5b840bed0665894555a0c82047a2a1c3625` |
| `src/schemaguard/models/adapters/preprocessing.py` | `740920dded34e94ab12f9f92913f473476cb495bf99878992152e8a86d63fc04` |
| `src/schemaguard/models/adapters/tabicl_adapter.py` | `f33b78c066cae6e29ec3973b8981ef2dcd774b856a2cb36291a749a3753612b6` |
| `src/schemaguard/models/adapters/tabpfn_adapter.py` | `cea5701305d6c9708d561bdf6e69abecd3e028fd6285fada66f603a3d3350d2e` |
| `src/schemaguard/models/adapters/xgboost_adapter.py` | `9affa8c09b3920fa7477969e7824309bf9bc9a288b2a27cef0ad2ee848984c56` |
| `src/schemaguard/models/registry.py` | `5b2d4b624c2feb7f5111e4c8eb04427e57bfd61b879fc2990a99caf98251e760` |
| `src/schemaguard/runner/contracts.py` | `13e5603a2235315ebc2161303bc89b1aaf71f5aba0c6c87c6f1f3a280683a442` |
| `src/schemaguard/splits/contracts.py` | `145212f5e16c7d00d18b233482c0d26a156416b16357217219e95797a1616ba1` |
| `src/schemaguard/transformations/contracts.py` | `7e96dfb837b7da1981bf7cb214ca05dd143e5d7fca2cda0d7a2c3fb8ddf0faf8` |
| `src/schemaguard/transformations/implementation.py` | `25a59f1eedcccf4c7827ac7c9897639d132e2bfe27f5f21d5501f9120211a879` |
| `src/schemaguard/transformations/implementation_tree` | `f7473c6caec0b2600e50fb929b3bb18edce4583863caaa26ea8d3810c8b6a1ee` |
| `src/schemaguard/transformations/registry.py` | `6414bb4b318d651fc830b55a66a854c658613c010ee4f4abc2e449fc1d8cdec8` |
| `src/schemaguard/utils/hashing.py` | `437363e7203cf439a5dfc40899dd039d48dac0c0776fc786dc29510d2488981c` |
| `src/schemaguard/utils/io.py` | `268d688b6627645d7e3ef70e45b2e1df51d63601817b579ae2724279bb24dc22` |
| `src/schemaguard/utils/process_lock.py` | `322cc152d669985ff3769d21db1210227c316c7184acd25fb0dac56f7ddab07b` |
| `src/schemaguard/utils/resource_monitor.py` | `8881a4197996a21d0ee6815b4a3c130085982eb17293e808322eea95da059354` |
| `uv.lock` | `0b6b2f3903d4cc16860c0da1c35c77096c04c2306256371b5d8181218000762f` |

## Frozen formulas, leakage, and decision criteria

```text
Brier = mean_i sum_c (P[i,c] - 1[y[i] == c])^2
delta(view) = Brier(view) - Brier(V00)
WBD = max(delta(view)) across applicable non-V00 views
RWBD = WBD / max(Brier(V00), epsilon); epsilon = 1e-12
SII_i = max pairwise JSD across applicable views; logarithm base 2
SII clips at 1e-15 then renormalizes each row
Flip_i = any argmax-class difference across applicable views
```
Label flips also record pairwise rates against V00, the worst view, and its transformation family. Raw AUROC is retained; tolerance-aware AUROC uses tau=1e-15, half-credit ties and chunk size 256, and records the number of tolerance ties. Tie-only classification requires equivalent probabilities/labels/Brier/log loss under the separate non-rank tolerance, raw AUROC change, and no tolerance-aware change.

Secondary metrics: log_loss, accuracy, balanced_accuracy, raw_auroc, tolerance_aware_auroc, expected_calibration_error, maximum_absolute_probability_difference, mean_absolute_probability_difference, runtime, fit_time, prediction_time, peak_process_tree_ram, peak_cuda_allocated_memory, peak_cuda_reserved_memory, cache_status, failure_category. Undefined metrics carry an explicit reason.
Rows are calculation units only; seeds are repeatability units; dataset-model values are medians across seeds; datasets are the cross-dataset unit/effective sample size. Report median, IQR, minimum, maximum, win/tie/loss, transformation-family, and model-family summaries. No confirmatory significance claims; intervals are exploratory.

Test-label sequence: plan_frozen -> test_labels_sealed -> training_completed -> calibration_predictions_completed -> test_predictions_completed -> prediction_structure_validated -> test_labels_opened -> metrics_generated -> results_sealed. Labels cannot influence planning, applicability, fitting, preprocessing, batch size, timeout, retry, resource fallback, view/model selection, or condition exclusion.

Integrity failures mean `REPAIR_REQUIRED`. Feasibility requires all V00 conditions, at least 95% transformed completion, classified failures, unchanged frozen model configuration, valid cache/resume, and respected resource ceilings. Nontrivial effect requires a foundation model with median dataset WBD >= 0.03 or RWBD >= 0.10. Breadth requires at least 3/8 datasets, 2 families, effect direction in 2/3 seeds on each counted dataset, and not tie-only; effect must remain after frozen preprocessing.
Classical LR/CatBoost/XGBoost comparisons characterize whether sensitivity is broad or TFM-focused. No positive result outside the frozen six-outcome vocabulary is allowed.

Retry policy: maximum 2 attempts; retryable WORKER_CRASH, TRANSIENT_IO, TIMEOUT; non-retryable GPU_OOM, INVALID_PROBABILITY, MISSING_ROW, CHECKPOINT_FAILURE, PACKAGE_OR_API_MISMATCH, OFFLINE_NETWORK_ATTEMPT, IDENTITY_MISMATCH. Failed outputs are not cache hits; partial artifacts cannot be promoted; condition identity must match.

## Resource and quality evidence

Runtime estimate method: For each scheduled applicable tuple, use the maximum of measured smoke wall time, median accepted adapter-probe time, and maximum accepted GPU-profile time for its model; scale by train-row ratio, square-root feature ratio, and class ratio; apply the configured 1.5 safety factor; CPU work is scheduled over two workers and CUDA work is strictly serial. Storage includes calibration/test rows, class probabilities, row IDs, per-row metadata allowance, two partition-artifact headers, and a 2x cache allowance. The full condition matrix is retained; a staged schedule is required if the preferred local target is exceeded.; measurement SHA-256 `5a12410f62ab16036b3691c760efb976ba385466da4c1443a6aedf81858c0ed1`.
CPU work 54478.8s; CUDA work 74980.1s; conservative wall 75040.1s.
Preferred/hard wall limits 28800/43200s; peak RAM/VRAM estimate 2109.3/1306.0 MiB.
Prediction/cache storage estimate 261605640/523211280 bytes; preferred storage 21474836480 bytes.
Resource ceilings: 2 CPU workers × 2 threads, 1 sequential GPU worker, 3600 MiB VRAM, 28672 MiB RAM. Timeouts: {'cpu_model_condition': 3600, 'cuda_model_condition': 1800}. Offline; no silent fallback/precision/batch/ensemble changes; atomic publication and validated resume.

## Commands, preservation, files, and deviations

Protected tree unchanged: `True`; files 1265; bytes 368851119; snapshot SHA-256 `dc02ae3f16feb1b24bd9c77d285f3860c4c4b792ed7fd07db3def42c64edc3b4`. Phase 01 data and split outputs were not regenerated.
Clean-clone validation: `True`; overlay files 24.
Executed command outputs are represented by exit codes, hashes, and test-summary highlights below. Absolute interpreter paths are omitted.
- `unit_tests`: exit 0; output SHA-256 `a1bad6ab23b9c839447aa6094a6f825f3b745a505e465457320cac0e3caa618b`; `P12 Python -m pytest -o addopts= -q -m not integration and not network and not gpu and not foundation_model -rA`.
  - PASSED tests/unit/test_transformation_validator_policy.py::test_alternate_ignored_inventory_output_preserves_tracked_inventory
  - PASSED tests/unit/test_xgboost_adapter.py::test_xgboost_binary_and_multiclass_probability_contract
  - PASSED tests/unit/test_xgboost_adapter.py::test_xgboost_unseen_category_uses_training_onehot_vocabulary
  - 502 passed, 35 deselected, 5 warnings in 104.56s (0:01:44)
- `integration_tests`: exit 0; output SHA-256 `60235d7b594eb81164e0953fb5c25cca5dc48dff4eb37b16a9d96952cdfa67f8`; `P12 Python -m pytest -o addopts= -q -m integration and not network and not gpu and not foundation_model -rA`.
  - PASSED tests/integration/test_smoke_experiment.py::test_tiny_smoke_plan_cold_execution_metrics_and_resume
  - PASSED tests/integration/test_split_generation.py::test_grouped_split_generation_write_validate_promote_reload_and_cache
  - PASSED tests/integration/test_transformation_roundtrip.py::test_end_to_end_materialization_cache_and_promotion_for_all_views
  - 22 passed, 515 deselected, 5 warnings in 39.61s
- `ruff`: exit 0; output SHA-256 `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18`; `P12 Python -m ruff check .`.
  - All checks passed!
- `mypy`: exit 0; output SHA-256 `a7c4aae49ff9d21cbf4edf1b955366f4cb545b2dbf41b63329a9f1c16b3b0c5d`; `P12 Python -m mypy src/schemaguard`.
- `naming`: exit 0; output SHA-256 `e23d637605363cef935aea29e38cb53219836c64fd7926223c496cf0bc5cd12a`; `P12 Python scripts/validate_repository_naming.py`.
- `repository`: exit 0; output SHA-256 `3aa38f78a4a87d3eb34b43a8951a0441c368b20973a6381f0db6e56bed900ec7`; `P12 Python scripts/validate_repository_repair.py --local-evidence`.

Deviation: the pasted specification cites Actions run `35296257149`, which GitHub shows as an in-progress push workflow, not the accepted smoke workflow. The accepted independent-smoke report binds successful run `35291513248`, verdict VERIFIED_PASS, 32/32 gates, and pilot_started=false.
No pilot model was trained; no pilot predictions or metrics were produced; no pilot test labels were decoded or evaluated. The private root reference `.docx` remained untouched and untracked.
Created/modified deliverables: protocol configuration, contracts, planning and validation source, freeze/validation scripts, eight generated schemas, focused tests, sanitized protocol JSON, condition inventory, staged schedule, validation evidence, and this review.
Next permitted workstream: independent pilot-protocol review. Pilot execution remains prohibited until that review accepts the handoff.
