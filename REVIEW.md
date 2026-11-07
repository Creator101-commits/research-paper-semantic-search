# Review and verification

Scope: source modules, CLI behavior, evaluation mathematics, project claims, Kaggle scripts where present, and README figures.

Corrected lexical model labels, isolated synthetic test queries, retrieved depth 100 for depth-100 metrics, rejected duplicate rankings, and validated empty bootstrap inputs.

## Verified locally

- 25 tests passed with the repository's existing Python environment.
- `ruff check .` passed, including tests, figure source, and Kaggle scripts.
- `mypy src` passed.
- The README image was rendered from code using Matplotlib and visually inspected. Its inputs/results and a source fingerprint are in `scripts/figure-data.json`.

## Limits

The loader returns synthetic papers and claims, not official SciFact records. Neural encoder and cross-encoder integrations remain planned.

Kaggle API access failed because the execution environment could not resolve its hostname. The private notebook was also unavailable through the browser. No GPU training or full-scale external benchmark result is claimed by this review. New CI workflows are configuration changes; their hosted execution must be checked after publication.
