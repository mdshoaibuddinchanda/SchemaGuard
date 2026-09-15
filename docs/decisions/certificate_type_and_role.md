# Certificate Type and Scientific Role

The transformation registry previously used `CONTROL` as a certificate type for V00, V08, and V09. That conflated the mathematical proof with the experimental role.

The corrected contract separates them:

| View | Certificate type | Scientific role |
| --- | --- | --- |
| V00 | BIJECTION | CONTROL |
| V01 | BIJECTION | PRIMARY_MIGRATION |
| V02 | BIJECTION | PRIMARY_MIGRATION |
| V03 | BIJECTION | PRIMARY_MIGRATION |
| V04 | BIJECTION | PRIMARY_MIGRATION |
| V05 | PROJECTION | PRIMARY_MIGRATION |
| V06 | PROJECTION | PRIMARY_MIGRATION |
| V07 | BIJECTION | PRIMARY_MIGRATION |
| V08 | PERMUTATION | CONTROL |
| V09 | PERMUTATION | CONTROL |
| V10 | COMPOSITION | PRIMARY_MIGRATION |

View IDs, names, scientific counts, seeds, datasets, and transformation definitions are unchanged. Only certificate metadata and its validation contract are corrected.
