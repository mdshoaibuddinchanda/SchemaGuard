# SchemaGuard Split Generation Review

## Status

SUPERSEDED — REPAIR REQUIRED IN THE ORIGINAL REVIEW

This historical review recorded the pre-audit result. The independent audit
identified incorrect persisted fold provenance and verification false-pass
paths. Those defects were repaired and revalidated in
`artifacts/handoff/split_verification_repair.md`; use that handoff for the
current split-generation status.

This workstream implements and validates deterministic grouped split generation only. Workstreams 4 and 5, including transformation code, model adapters, model training, experiments, SCNF, COSA, statistical analysis, figures, and paper results, were not started or modified.

## Starting state

* Starting commit: bbc09c271d4024cd12ab0f84a4912507f46e1203
* Branch: main
* Python executable: D:\Conda\P12\python.exe
* Python version: 3.12.14
* Conda environment: P12
* Initial tracked worktree state: clean.
* Initial untracked state: SchemaGuard_Complete_Research_and_Engineering_Plan.docx only.
* The private root .docx was not opened, edited, moved, deleted, staged, committed, or pushed.
* Implementation commit: c9e06528f37c6c63def3dc64bc1cbb9c13bf7dfa

## Scope and frozen configuration

* Dataset count: 14.
* Dataset IDs: 3, 23, 29, 31, 36, 37, 38, 44, 46, 50, 54, 1067, 1464, 1489.
* Seed count: 5.
* Seeds: 1729, 2718, 31415, 57721, 161803.
* Expected active split count: 70.
* Observed active split count: 70.
* Strategy: stratified_group_5fold_v1.
* Grouping method: typed_predictor_sha256_v1.
* Grouping implementation hash: f59c2e4a38e7df03e16f6941188b430fcd033288702f8af8dcbd8223f2416556.
* Configuration hash: f7243331ef1688739b69843fd40928c0a0f4f15c79cdbdd23b3b5a0d81fca619.
* CPU worker limit used: 2.
* Network: disabled for generation and validation; no dataset was downloaded or reconstructed.

## Repository hygiene

The README now reports the split-generation workstream as VERIFIED_PASS. The master plan uses semantic handoff wording and points to artifacts/handoff/split_generation_review.md.

The generated-path policy was checked with git check-ignore -v:

* schemas/example.schema.json: not ignored.
* tests/fixtures/example.csv: not ignored.
* tests/fixtures/example.tsv: not ignored.
* data/raw/example.csv: ignored by .gitignore:10:data/raw/.
* data/processed/example.parquet: ignored by .gitignore:11:data/processed/.
* results/example.csv: ignored by .gitignore:35:results/.
* artifacts/handoff/example.md: not ignored by .gitignore:24:!artifacts/handoff/*.md.

No generated dataset, split Parquet, cache, or temporary file is tracked.

## Existing OpenML 1464 artifacts

The two pre-existing artifacts were audited without overwrite or deletion:

* Active grouped split: data/splits/openml/1464/stratified_group_5fold_v1/seed_1729/.
* Deprecated audit baseline: data/splits/openml/1464/seed_1729/, the older row-stratified artifact. It does not count toward the 70 active manifests.

The protected active assignment remains:

* Assignment SHA-256: e08b9d5c94578b317dcc844c2a6aa7a4f96ad8df699352d1e11d86a7dd452236.
* Logical assignment SHA-256: f66b2ab092218813b82e49b6a3c19df98b06840d90b463bd9d09ba1bce385365.
* Rows: train 449, calibration 150, test 149.
* Class counts: train 0:342, 1:107; calibration 0:114, 1:36; test 0:114, 1:35.
* Predictor groups: 502; duplicate groups: 69; conflicting-target groups: 31; crossing groups: 0.

## Complete 70-split evidence

The following table is generated from the final validated manifests. Partition sizes use train/calibration/test. Class counts use train; calibration; test. Group counts use predictor groups/duplicate groups/conflicting-target groups; every row has cross=0.

| Dataset / seed | Partition sizes | Class counts | Groups / duplicate / conflict / cross | Logical assignment SHA-256 |
| --- | --- | --- | --- | --- |
| 3 kr-vs-kp / 1729 | 1918/639/639 | tr[0:917,1:1001]; cal[0:305,1:334]; te[0:305,1:334] | 3196/0/0/0 | 405fbe6cdd976257abaaecbf1d8175098752b247c67932766c43ee0b9c230f47 |
| 3 kr-vs-kp / 2718 | 1918/639/639 | tr[0:917,1:1001]; cal[0:305,1:334]; te[0:305,1:334] | 3196/0/0/0 | 623dd5d64a3492c66f6d9d1e42ce0d3d85d8b2f13e7a65c9d87ac0a9545ab76d |
| 3 kr-vs-kp / 31415 | 1918/639/639 | tr[0:917,1:1001]; cal[0:305,1:334]; te[0:305,1:334] | 3196/0/0/0 | 475422ea247f910ebd79dbb1e2ec49537f2b619e4c9afe6326721c2a07bd030b |
| 3 kr-vs-kp / 57721 | 1918/639/639 | tr[0:917,1:1001]; cal[0:305,1:334]; te[0:305,1:334] | 3196/0/0/0 | 40d692f0ad03153b2888c837747b84b2a816772fb0699559bc03d2294722988a |
| 3 kr-vs-kp / 161803 | 1918/639/639 | tr[0:917,1:1001]; cal[0:305,1:334]; te[0:305,1:334] | 3196/0/0/0 | a346bc8d6e331db3811b8062f4232f135db1acbdd634347afdf577e3e5a4f988 |
| 23 cmc / 1729 | 883/295/295 | tr[0:377,1:199,2:307]; cal[0:126,1:67,2:102]; te[0:126,1:67,2:102] | 1358/96/62/0 | 66b2974f3903974bf86c9690c743d49f9e44b9b928142a7efbab46d1b3ad2d8c |
| 23 cmc / 2718 | 883/295/295 | tr[0:377,1:199,2:307]; cal[0:126,1:67,2:102]; te[0:126,1:67,2:102] | 1358/96/62/0 | 1fb17155edd739cdfba1e42faccb7c738b78a7016a5ba496c9d8ed7ce71ceeea |
| 23 cmc / 31415 | 883/295/295 | tr[0:377,1:199,2:307]; cal[0:126,1:67,2:102]; te[0:126,1:67,2:102] | 1358/96/62/0 | 737303a706048ce767b4a616e9a03dac61ded634fb11b7eaae21a400ca55cb74 |
| 23 cmc / 57721 | 883/295/295 | tr[0:377,1:199,2:307]; cal[0:126,1:67,2:102]; te[0:126,1:67,2:102] | 1358/96/62/0 | a0bdd4ca0f28bc4aca70409b12e50e7a4e2fdcd4ef960acbfef1ed9754098286 |
| 23 cmc / 161803 | 883/295/295 | tr[0:377,1:199,2:307]; cal[0:126,1:67,2:102]; te[0:126,1:67,2:102] | 1358/96/62/0 | 289dd6e4b7a50d91a20bb2707faccbfff33e72831724801de9cee466dfcffce3 |
| 29 credit-approval / 1729 | 414/138/138 | tr[0:185,1:229]; cal[0:61,1:77]; te[0:61,1:77] | 690/0/0/0 | 97129696d11211654ab8bff4e4940cfc3778130353f7a30951083a997da18a6e |
| 29 credit-approval / 2718 | 414/138/138 | tr[0:185,1:229]; cal[0:61,1:77]; te[0:61,1:77] | 690/0/0/0 | 7caf746bfcf5f3f07ded55462f9675e600aea19ec6e2191abb0d4a01cd7230f8 |
| 29 credit-approval / 31415 | 414/138/138 | tr[0:185,1:229]; cal[0:61,1:77]; te[0:61,1:77] | 690/0/0/0 | 04b0cf42318d4e5c786f69ac5125eed6b6182a7e07c12d74cbb4e43f875ca166 |
| 29 credit-approval / 57721 | 414/138/138 | tr[0:185,1:229]; cal[0:61,1:77]; te[0:61,1:77] | 690/0/0/0 | 502265b44df5bd95f168d019ee82e100bc120bb9f8d49c294290f1a846f6b8c5 |
| 29 credit-approval / 161803 | 414/138/138 | tr[0:185,1:229]; cal[0:61,1:77]; te[0:61,1:77] | 690/0/0/0 | ecf409af5ce37d3d053f25e6ca9e711b168a2fdbbb9a9fd655ead5299b5465a8 |
| 31 credit-g / 1729 | 600/200/200 | tr[0:180,1:420]; cal[0:60,1:140]; te[0:60,1:140] | 1000/0/0/0 | bd9b4c1af4ac1f800604bea9ba51754625ccad5f3d84df5099a53fe50bae5186 |
| 31 credit-g / 2718 | 600/200/200 | tr[0:180,1:420]; cal[0:60,1:140]; te[0:60,1:140] | 1000/0/0/0 | f179704281144f25bd19431653b791aa472082f15de0742e8cb6c14c5b8967dd |
| 31 credit-g / 31415 | 600/200/200 | tr[0:180,1:420]; cal[0:60,1:140]; te[0:60,1:140] | 1000/0/0/0 | 265178294702b4a01e6ab82005ab904e31391b83043322ef37faac274f2d4d4c |
| 31 credit-g / 57721 | 600/200/200 | tr[0:180,1:420]; cal[0:60,1:140]; te[0:60,1:140] | 1000/0/0/0 | 88477908fd688477e1135bf3b4e79951f7c7f487102b9bab6ebe75f33ce18c14 |
| 31 credit-g / 161803 | 600/200/200 | tr[0:180,1:420]; cal[0:60,1:140]; te[0:60,1:140] | 1000/0/0/0 | a94abe92e49d09facb35080783dffb936bb126e7313799831d113e7f1aa750f5 |
| 36 segment / 1729 | 1386/462/462 | tr[0:198,1:198,2:198,3:198,4:198,5:198,6:198]; cal[0:66,1:66,2:66,3:66,4:66,5:66,6:66]; te[0:66,1:66,2:66,3:66,4:66,5:66,6:66] | 2086/222/0/0 | dc241df0e87b226bf7732a2c4b3ddb2bdf9f4eeecd76863f3465c88c0fb43d27 |
| 36 segment / 2718 | 1386/462/462 | tr[0:198,1:198,2:198,3:198,4:198,5:198,6:198]; cal[0:66,1:66,2:66,3:66,4:66,5:66,6:66]; te[0:66,1:66,2:66,3:66,4:66,5:66,6:66] | 2086/222/0/0 | d65af1a393e3b3d8e77a18952dee8ecf34b0bded71511d9f8fedf01ca6fe4add |
| 36 segment / 31415 | 1386/462/462 | tr[0:198,1:198,2:198,3:198,4:198,5:198,6:198]; cal[0:66,1:66,2:66,3:66,4:66,5:66,6:66]; te[0:66,1:66,2:66,3:66,4:66,5:66,6:66] | 2086/222/0/0 | 2f23706f33aaf6e8be41918dea0077444543164ce982adaf3d934cce192d2033 |
| 36 segment / 57721 | 1386/462/462 | tr[0:198,1:198,2:198,3:198,4:198,5:198,6:198]; cal[0:66,1:66,2:66,3:66,4:66,5:66,6:66]; te[0:66,1:66,2:66,3:66,4:66,5:66,6:66] | 2086/222/0/0 | 5e7c81c73cd1e79d635454f5ce1bced12ffc872b3b7a4fd796ba384742495d89 |
| 36 segment / 161803 | 1386/462/462 | tr[0:198,1:198,2:198,3:198,4:198,5:198,6:198]; cal[0:66,1:66,2:66,3:66,4:66,5:66,6:66]; te[0:66,1:66,2:66,3:66,4:66,5:66,6:66] | 2086/222/0/0 | 647624ac4df99bd1859ebe292597612959d31c420706399d96747e87df69a105 |
| 37 diabetes / 1729 | 461/154/153 | tr[0:300,1:161]; cal[0:100,1:54]; te[0:100,1:53] | 768/0/0/0 | 253c02dde0d8ba5340a0e0b138f8b13db784e6156d936a036a398787a23a3796 |
| 37 diabetes / 2718 | 461/153/154 | tr[0:300,1:161]; cal[0:100,1:53]; te[0:100,1:54] | 768/0/0/0 | 128dd49dd56db42000264f8d363c3f1f338c4fcc0bc2a91e4d3c12ac180b39c9 |
| 37 diabetes / 31415 | 461/154/153 | tr[0:300,1:161]; cal[0:100,1:54]; te[0:100,1:53] | 768/0/0/0 | 4c318f750a3715e6feedc8289d2b20e3dee96d8284ea5c129888fbe98af8eef6 |
| 37 diabetes / 57721 | 461/153/154 | tr[0:300,1:161]; cal[0:100,1:53]; te[0:100,1:54] | 768/0/0/0 | 5b608fd7dc5ff041576777b93486b6d58f93ebd93f2d02d2276f069028f0bf45 |
| 37 diabetes / 161803 | 461/153/154 | tr[0:300,1:161]; cal[0:100,1:53]; te[0:100,1:54] | 768/0/0/0 | aec09d467ed20dc640c30a3965b1843e4375b946f082672890aa836d8dc3d918 |
| 38 sick / 1729 | 2263/755/754 | tr[0:2124,1:139]; cal[0:709,1:46]; te[0:708,1:46] | 3711/33/0/0 | 1327136ffd7cd4de6823ec3ebf016c8960de81a8b70657b62d8f704b1c0e74ea |
| 38 sick / 2718 | 2263/755/754 | tr[0:2124,1:139]; cal[0:709,1:46]; te[0:708,1:46] | 3711/33/0/0 | 7d0878af1d2bd344115bbbf4ae65099249ffc4c2d30eaa3ccec47fb42e22c7d9 |
| 38 sick / 31415 | 2263/755/754 | tr[0:2124,1:139]; cal[0:709,1:46]; te[0:708,1:46] | 3711/33/0/0 | c5181f664785b79ff0e8ee6c0b58d6dbb65ceb77d6f5acb1bf422e9080c261ea |
| 38 sick / 57721 | 2263/755/754 | tr[0:2124,1:139]; cal[0:709,1:46]; te[0:708,1:46] | 3711/33/0/0 | 03bb6a43f26e07c26c543f196f201636bb9ec632aa6900e15aee36905184ab3d |
| 38 sick / 161803 | 2263/755/754 | tr[0:2124,1:139]; cal[0:709,1:46]; te[0:708,1:46] | 3711/33/0/0 | 52f9bbef8646c5e712ae350af2b313a0a59ab70ea493a7ecc278b22a116d4e7f |
| 44 spambase / 1729 | 2761/919/921 | tr[0:1673,1:1088]; cal[0:557,1:362]; te[0:558,1:363] | 4207/183/3/0 | 5fbdd846273d70f9e8dea96f125be76838dde2cc525ac0a93d47075cf048cab5 |
| 44 spambase / 2718 | 2761/919/921 | tr[0:1673,1:1088]; cal[0:557,1:362]; te[0:558,1:363] | 4207/183/3/0 | 306f08ea7e75dd6db890a6c71288b7e80e911aef7ef2a779d96949afa8778906 |
| 44 spambase / 31415 | 2761/919/921 | tr[0:1673,1:1088]; cal[0:557,1:362]; te[0:558,1:363] | 4207/183/3/0 | 7789c5fcdce0b6a04ceaaac2e5a997484403b819ba7d5a31ba098da5f1b04be1 |
| 44 spambase / 57721 | 2761/919/921 | tr[0:1673,1:1088]; cal[0:557,1:362]; te[0:558,1:363] | 4207/183/3/0 | 61b04db5262939bdeac06722e837aceab65e63cb2afca8d278d716ef43ad7d29 |
| 44 spambase / 161803 | 2761/919/921 | tr[0:1673,1:1088]; cal[0:557,1:362]; te[0:558,1:363] | 4207/183/3/0 | 272b47e5977ce4cbcb15f3f2cf63c4a9a72092c378720ecd64df0ed9900d0210 |
| 46 splice / 1729 | 1914/638/638 | tr[0:461,1:460,2:993]; cal[0:153,1:154,2:331]; te[0:153,1:154,2:331] | 3005/125/1/0 | 7c92f36efa49d078fdd59854812e4339b03118c04e7444b420643f133e224d51 |
| 46 splice / 2718 | 1914/638/638 | tr[0:460,1:461,2:993]; cal[0:154,1:153,2:331]; te[0:153,1:154,2:331] | 3005/125/1/0 | e149e0c58e09f501967f842864a30cd8e4c7840043e1d055a31a485244506cea |
| 46 splice / 31415 | 1914/638/638 | tr[0:461,1:460,2:993]; cal[0:153,1:154,2:331]; te[0:153,1:154,2:331] | 3005/125/1/0 | 15400e04f2c101191c91419523375288dcc0f346eec5de7d8268c83fe178914b |
| 46 splice / 57721 | 1914/638/638 | tr[0:461,1:460,2:993]; cal[0:153,1:154,2:331]; te[0:153,1:154,2:331] | 3005/125/1/0 | fa0cfe562bf628e2b389ad063dd60ce4e8f151fd323f4bf857e65c1badc1b820 |
| 46 splice / 161803 | 1914/638/638 | tr[0:460,1:461,2:993]; cal[0:153,1:154,2:331]; te[0:154,1:153,2:331] | 3005/125/1/0 | dd08840d7c2a9d46a3d23900817d46ec6465e835bb4dbbb6507744d70518271f |
| 50 tic-tac-toe / 1729 | 575/191/192 | tr[0:199,1:376]; cal[0:66,1:125]; te[0:67,1:125] | 958/0/0/0 | 1a5a8c297e44c40f1130353a3eb9a7638aae67c3e24566847ab7c219a9374d6a |
| 50 tic-tac-toe / 2718 | 575/192/191 | tr[0:199,1:376]; cal[0:67,1:125]; te[0:66,1:125] | 958/0/0/0 | 97be157f434f6bc2e9af8279db3d8f83cf16075fc707d2948b2f6e4b7b27cf59 |
| 50 tic-tac-toe / 31415 | 575/191/192 | tr[0:199,1:376]; cal[0:66,1:125]; te[0:67,1:125] | 958/0/0/0 | f5ac9fb0b75645d5bb25a8f9d1ad4b52c560e76706a9cb42a7fd76354efd7d9e |
| 50 tic-tac-toe / 57721 | 575/192/191 | tr[0:199,1:376]; cal[0:67,1:125]; te[0:66,1:125] | 958/0/0/0 | cd5ade2959f535e0966b10fcedd16a627b72a05e4377f54bdcbd2da68e9da452 |
| 50 tic-tac-toe / 161803 | 575/192/191 | tr[0:199,1:376]; cal[0:67,1:125]; te[0:66,1:125] | 958/0/0/0 | 915afbefbe7c52d37fc381ee3d4ee0fe8c6c9ca84dd49ef3c83572888b25aaae |
| 54 vehicle / 1729 | 508/169/169 | tr[0:131,1:128,2:130,3:119]; cal[0:43,1:42,2:44,3:40]; te[0:44,1:42,2:43,3:40] | 846/0/0/0 | d100ad3604782cbab3fdaa1e9e7705b357a6b57340cfd42ada809fb8bf08391c |
| 54 vehicle / 2718 | 508/169/169 | tr[0:131,1:128,2:130,3:119]; cal[0:43,1:42,2:44,3:40]; te[0:44,1:42,2:43,3:40] | 846/0/0/0 | 9b99cf20f32a39ecc51bc0a6846ed1026a0737f4fc1487ec1c290363479b5866 |
| 54 vehicle / 31415 | 508/169/169 | tr[0:131,1:128,2:130,3:119]; cal[0:44,1:42,2:43,3:40]; te[0:43,1:43,2:43,3:40] | 846/0/0/0 | 65f22aaa47e978eea7657e704538b77b131a49fea577d676f9be8299db9c251a |
| 54 vehicle / 57721 | 508/169/169 | tr[0:131,1:128,2:130,3:119]; cal[0:43,1:43,2:43,3:40]; te[0:44,1:42,2:43,3:40] | 846/0/0/0 | c0cfb613853f1ab6937a144877f8e5266adf720454525ab2132726217371fa70 |
| 54 vehicle / 161803 | 508/169/169 | tr[0:131,1:128,2:130,3:119]; cal[0:44,1:42,2:43,3:40]; te[0:43,1:43,2:43,3:40] | 846/0/0/0 | 68e6a092c01fa5607cea6e7ac0fb54e7d16545a5156ca6632a996e294277fc0b |
| 1067 kc1 / 1729 | 1265/422/422 | tr[0:1069,1:196]; cal[0:357,1:65]; te[0:357,1:65] | 1192/177/20/0 | 00a0c5f66c0624a1b2d497d26987826ecf449e8d1d8a6397c9dd70204e395fa3 |
| 1067 kc1 / 2718 | 1265/422/422 | tr[0:1069,1:196]; cal[0:357,1:65]; te[0:357,1:65] | 1192/177/20/0 | b624cadae228875088473e21c307c7191ec15348232a2c4dd1b06ecbe81a13bf |
| 1067 kc1 / 31415 | 1265/422/422 | tr[0:1069,1:196]; cal[0:357,1:65]; te[0:357,1:65] | 1192/177/20/0 | 5ee5ac93805375a7a0554fbd7de5d9cd73d5589ce03b477acd441b8f54a5e658 |
| 1067 kc1 / 57721 | 1265/422/422 | tr[0:1069,1:196]; cal[0:357,1:65]; te[0:357,1:65] | 1192/177/20/0 | 8c7fe98a2f2f416f0149c264b901f4d11467e4be2d2e92e8efcf2c90e637fdb0 |
| 1067 kc1 / 161803 | 1265/422/422 | tr[0:1069,1:196]; cal[0:357,1:65]; te[0:357,1:65] | 1192/177/20/0 | d36cf769a23688fe68f845adf743be8c219aba0d42046d0a00ccab7113cf5235 |
| 1464 blood-transfusion-service-center / 1729 | 449/150/149 | tr[0:342,1:107]; cal[0:114,1:36]; te[0:114,1:35] | 502/69/31/0 | f66b2ab092218813b82e49b6a3c19df98b06840d90b463bd9d09ba1bce385365 |
| 1464 blood-transfusion-service-center / 2718 | 449/150/149 | tr[0:342,1:107]; cal[0:114,1:36]; te[0:114,1:35] | 502/69/31/0 | 7a360922ac055a47f049cf8bd8ba506dfdb16f0d16ce4f225d202e2ccbf1882d |
| 1464 blood-transfusion-service-center / 31415 | 449/150/149 | tr[0:342,1:107]; cal[0:114,1:36]; te[0:114,1:35] | 502/69/31/0 | db15be1e435f161222a6855dab2bd0d002d230d965d77fd90f80f539fd6bd432 |
| 1464 blood-transfusion-service-center / 57721 | 449/150/149 | tr[0:342,1:107]; cal[0:114,1:36]; te[0:114,1:35] | 502/69/31/0 | 321e887033b357aa7de7e91f27180e20e0f4bba3faecfbb88c50f9db6e54a228 |
| 1464 blood-transfusion-service-center / 161803 | 449/150/149 | tr[0:342,1:107]; cal[0:114,1:36]; te[0:114,1:35] | 502/69/31/0 | 9dd50048c9fdf3c063aca3c816b41c4342bad8000ef95cf4da10efe52ba94161 |
| 1489 phoneme / 1729 | 3242/1081/1081 | tr[0:2290,1:952]; cal[0:764,1:317]; te[0:764,1:317] | 5395/9/0/0 | a182a9105ec19ba98204746f4bbd8b3e03d5b226fcded231c928ef98169c2c0e |
| 1489 phoneme / 2718 | 3242/1081/1081 | tr[0:2290,1:952]; cal[0:764,1:317]; te[0:764,1:317] | 5395/9/0/0 | 221e380515832bb7fb80abf31967270f6519d6d3db2838d07f6eeb284bd54c40 |
| 1489 phoneme / 31415 | 3242/1081/1081 | tr[0:2290,1:952]; cal[0:764,1:317]; te[0:764,1:317] | 5395/9/0/0 | 5b70a0f024fa13c81eea17b9a53f2f87b69376d949da86db26cfac4ed3355281 |
| 1489 phoneme / 57721 | 3242/1081/1081 | tr[0:2290,1:952]; cal[0:764,1:317]; te[0:764,1:317] | 5395/9/0/0 | 198dd1b041ef90697a6a9318bed44d1229af2dcda6acccfe76ac9ff6a2a1141e |
| 1489 phoneme / 161803 | 3242/1081/1081 | tr[0:2290,1:952]; cal[0:764,1:317]; te[0:764,1:317] | 5395/9/0/0 | ac033de0557fb7db858d9039a141d932f6db78139ef3f0008dcf28723cf91936 |

## Determinism and cache results

* Same seed and same data: PASS for the forced regeneration checks.
* Shuffled physical input order: PASS in unit and integration tests; logical assignment hashing is physical-order invariant.
* Distinct seed-specific logical hashes: 5 of 5 for each of the 14 datasets.
* Every generated manifest records all 20 ordered calibration/test fold candidates and their diagnostics.
* Cache rerun command exited 0.
* Cache rerun result: 69 validated CACHE_HIT records plus 1 validated PROTECTED_BASELINE record. The protected record is intentionally not relabeled as a normal cache hit.
* Cache rerun did not alter the validated assignment SHA-256 or logical assignment SHA-256 set.
* No active split directory contains a temporary or partial file.

## Phase 01 preservation

The protected snapshot covered 142 files across the data-foundation baseline, all available raw/processed source artifacts, and the protected split. The final comparison is:

* Before snapshot: artifacts/validation/split_generation_hashes_before.json, SHA-256 D23DB83257A924C1D775936E7C488F4115D9EC63C5137A371584590BAA4DDF1C.
* After snapshot: artifacts/validation/split_generation_hashes_after.json, SHA-256 B165E04974EB8C354C872AB91D54881961AFADCAA442811ED326F05506C6C543.
* Comparison: artifacts/validation/split_generation_hash_comparison.json, SHA-256 03AB0AC064A362E13E7D6A09836E2FE2A50849C0A0C8A87EBECF48428540C3F4.
* Comparison status: PASS; changed_files=[]; file count before 142; file count after 142.
* Existing grouped baseline: 748 rows, train 449, calibration 150, test 149, 502 predictor groups, 69 duplicate groups, 31 conflicting-target groups, zero crossing groups.
* The deprecated row-stratified split remains present and excluded from the active count.

## Tests and quality

| Test or check | Result |
| --- | --- |
| Full repository pytest | PASS: 121 passed, 2 skipped; 2 network tests deselected by repository default marker |
| Required unit suite (tests/unit, excluding network/gpu/foundation/evidence) | PASS: 104 passed |
| Required split integration test | PASS: 1 passed |
| Split-focused unit tests | PASS: 18 passed |
| Ruff | PASS |
| Mypy | PASS: 41 source files |
| Artifact schema generation | PASS |
| Repository naming validator | PASS |
| Repository repair validator | PASS: R01-R09 |
| Local evidence validator | PASS |
| Independent 70-split validator | PASS: 70 validated |
| Protected-artifact hash comparison | PASS: 0 changed files |

Warnings were limited to the existing scikit-learn LogisticRegression deprecation warning and the deliberate impossible-class-constraint test warning. No test failure was suppressed.

## Resource use

* Runtime: Python 3.12.14 under P12 on Windows 11.
* CPU: 11th Gen Intel Core i5-11260H, 4 physical cores, 8 logical processors.
* Installed RAM: 31.73 GiB.
* GPU: not used; GPU is not required for this workstream.
* Observed peak Python working set during the final two-worker refresh: approximately 273.9 MB (261.1 MiB). This is an OS working-set observation, not a fabricated child-process metric.
* Final forced refresh wall time: approximately 5 minutes 35 seconds.
* Final independent validation wall time: approximately 1 minute 39 seconds.
* Cache verification wall time: approximately 5 minutes 30 seconds.
* No checkpoint, dataset, or large generated artifact was added to Git.

## Generated evidence

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| artifacts/validation/split_generation_inventory.json | 51,942 bytes | 8F04B2795616F2B7755758ACFFD9D5F4039B5D5C94930878407208F448AB2CE4 |
| artifacts/validation/split_generation_hashes_before.json | 16,647 bytes | D23DB83257A924C1D775936E7C488F4115D9EC63C5137A371584590BAA4DDF1C |
| artifacts/validation/split_generation_hashes_after.json | 16,647 bytes | B165E04974EB8C354C872AB91D54881961AFADCAA442811ED326F05506C6C543 |
| artifacts/validation/split_generation_hash_comparison.json | 124 bytes | 03AB0AC064A362E13E7D6A09836E2FE2A50849C0A0C8A87EBECF48428540C3F4 |

The 70 assignment Parquet files and 70 manifests remain ignored local evidence under data/splits/; the complete record inventory is retained in split_generation_inventory.json, and each manifest contains its source and assignment hashes.

## Files created

* configs/runtime/split_generation.yaml
* scripts/generate_splits.py
* scripts/validate_split_generation.py
* src/schemaguard/splits/__init__.py
* src/schemaguard/splits/contracts.py
* src/schemaguard/splits/grouping.py
* src/schemaguard/splits/generation.py
* src/schemaguard/splits/selection.py
* src/schemaguard/splits/validation.py
* src/schemaguard/splits/caching.py
* src/schemaguard/splits/locking.py
* src/schemaguard/splits/inventory.py
* tests/unit/__init__.py
* tests/unit/test_predictor_grouping.py
* tests/unit/test_split_contracts.py
* tests/unit/test_split_selection.py
* tests/unit/test_split_validation.py
* tests/unit/test_split_caching.py
* tests/unit/test_split_faults.py
* tests/integration/__init__.py
* tests/integration/test_split_generation.py
* artifacts/handoff/split_generation_review.md

## Files modified

* README.md
* SCHEMAGUARD_MASTER_PLAN.md

No transformation, model-adapter, experiment, SCNF, COSA, statistical, figure, or paper-results file was modified.

## Exact commands and exit codes

The required preflight and validation commands were run under P12. Successful commands exited 0.

~~~text
cd D:\DR2\SchemaGuard
git status --short                         # 0; only the permitted untracked .docx
git branch --show-current                  # 0; main
git rev-parse HEAD                         # 0; bbc09c271d4024cd12ab0f84a4912507f46e1203
git log -1 --oneline                       # 0; bbc09c2 Correct handoff verification metadata
git ls-files                               # 0

conda run -n P12 python scripts/validate_repository_repair.py       # 0
conda run -n P12 python scripts/validate_local_evidence.py          # 0
conda run -n P12 python scripts/validate_repository_naming.py        # 0

conda run -n P12 python scripts/generate_splits.py --config configs/runtime/split_generation.yaml --dataset-id 1464 --seed 1729 --offline
# 0; protected baseline PASS
conda run -n P12 python scripts/generate_splits.py --config configs/runtime/split_generation.yaml --dataset-id 1464 --workers 2 --offline
# 0
conda run -n P12 python scripts/generate_splits.py --config configs/runtime/split_generation.yaml --workers 2 --offline
# 0
conda run -n P12 python scripts/generate_splits.py --config configs/runtime/split_generation.yaml --workers 2 --offline --force-recompute
# 0; 69 GENERATED plus protected baseline
conda run -n P12 python scripts/validate_split_generation.py --config configs/runtime/split_generation.yaml --all --offline
# 0; 70 validated
conda run -n P12 python scripts/generate_splits.py --config configs/runtime/split_generation.yaml --workers 2 --offline
# 0; 69 CACHE_HIT plus protected baseline

conda run -n P12 python -m ruff check .                                  # 0
conda run -n P12 python -m mypy src/schemaguard                          # 0
conda run -n P12 python -m pytest -q                                     # 0
conda run -n P12 python -m pytest -q tests/unit -m "not network and not gpu and not foundation_model and not evidence"
# 0; 104 passed
conda run -n P12 python -m pytest -q tests/integration/test_split_generation.py
# 0; 1 passed
conda run -n P12 python scripts/generate_artifact_schemas.py             # 0
conda run -n P12 python scripts/validate_repository_naming.py            # 0
conda run -n P12 python scripts/validate_repository_repair.py            # 0
conda run -n P12 python scripts/validate_local_evidence.py               # 0

git diff --check                                                       # 0
git commit -m "Implement deterministic grouped split generation"       # 0
git commit -m "Record split generation verification"                   # pending handoff commit
~~~

## Deviations and failures

* Deviations: the protected 1464/1729 artifact is reported as PROTECTED_BASELINE rather than a normal CACHE_HIT; this preserves its required classification while proving validated reuse.
* The repository's two foundation-model integration tests remain explicit controlled skips because they belong to the separate model-compatibility workstream.
* Failures: none.
* Blocked conditions: none.

## Next permitted workstream

The split-generation workstream is complete. A later workstream may begin only after independent review accepts this PASS handoff. This review does not authorize transformations, model adapters, experiments, SCNF, COSA, statistics, figures, or paper-result generation.
