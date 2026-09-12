# Effect of observed-only condition transfer versus training fraction

Declared in `EXPLORATORY_FRACTION_SWEEP.md` at commit `2f7c74f`. Exploratory rows carry no verdict and cannot be promoted to a claim. The confirmatory row is the committed preregistered result, included for scale.

| Label | Fraction | Included / degenerate / missing | Mean paired RMSE reduction | 95% CI | Improved / worsened | Accepted rows per unit (observed-only) | Accepted rows per unit (global rule) | Verdict |
| --- | ---: | --- | ---: | --- | --- | --- | --- | --- |
| exploratory | 0.01 | 9 / 0 / 0 | -0.415 | [-0.806, -0.010] (seed 1701) | 3 / 6 | 22–27 | 0–0 | not_applicable_exploratory |
| confirmatory | 0.05 | 9 / 0 / 0 | -0.377 | [-0.746, -0.035] (seed 1601) | 2 / 7 | 101–125 | 0–2 | null |
| exploratory | 0.1 | 9 / 0 / 0 | -0.508 | [-0.707, -0.273] (seed 1701) | 1 / 8 | 185–212 | 0–3 | not_applicable_exploratory |

Positive values favour augmentation. The confirmatory row is decided by its preregistered interpretation table; the exploratory rows are descriptive only.
