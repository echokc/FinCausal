# Recipe Eval Pipeline Summary

This note summarizes the recipe-to-eval pipeline decisions from the May 8,
2026 implementation discussion. It is meant to be the practical handoff doc:
what we built, what the interfaces mean, and how to scale the next scenarios.

## Core Decision

The recipe should remain the source of truth for experimental design, but it
should not contain executable generation or scoring logic.

The scalable split is:

```text
recipe
  -> generated data fixtures
  -> generated prompt
  -> candidate code or raw LLM response
  -> generic recipe scorer
  -> eval record
```

This avoids one scorer and one prompt template per scenario. New scenarios
should normally add a thin recipe plus, only when needed, a small registered data
generator and reusable probe support.

## Package Map

The current recipe pipeline is split by responsibility:

- `eval/framework/schema/`: declarative recipe objects, recipe primitives, probes,
  schema variants, generated-case schema helpers, and case manifest construction.
- `eval/framework/generator/`: recipe-driven prompt and data generation.
- `eval/framework/scorer/`: generic recipe scorer that executes candidate code and
  evaluates recipe probes.
- `eval/smoketest/`: positive/negative control snippets and fixture bindings.
- `eval/runner/`: CLI smoke runners for scorer and prompt generation.
- `notebooks/recipe_eval_pipeline.ipynb`: notebook version of the pipeline.

The older phase-1 `eval/data_factory/` path has been removed. The recipe
pipeline is now the generalized path for the multi-scenario eval work.

## What Is Covered

The current smoke scope covers:

- `s001_global_quantile_leakage`
- `s002_expost_news_contamination`
- `s003_covariance_inversion_stability`
- `s005_microstructure_volatility_scaling`
- `s006_inventory_induced_skew`
- `s007_liquidity_illusion_realizable_pnl`
- `s008_vol_signature_misclassification`
- `s009_volume_lead_lag_causality`
- `s010_fat_tail_fiduciary_discrimination`
- `s011_outlier_robust_breakout_detection`
- `s012_fat_tail_hedge_notional`
- `s013_temporal_rolling_zscore_causal_leakage`
- `s014_temporal_rolling_corr_beta_causal_leakage`
- `s015_temporal_rolling_var_cvar_causal_leakage`
- `s016_temporal_rolling_drawdown_causal_leakage`
- `s017_temporal_rolling_skew_kurtosis_causal_leakage`
- `s018_statistical_spurious_granger_causality`
- `s019_statistical_ignored_cointegration_pairs_trading`
- `s020_statistical_spurious_factor_significance`
- `s021_statistical_non_stationary_residuals`
- `s022_statistical_false_lead_lag_relationships`
- `s023_regime_multiple_structural_breaks_misclassification`
- `s024_regime_slow_drift_misclassification`
- `s025_regime_correlation_break_misclassification`
- `s026_regime_volatility_clustering_vs_regime_shift`
- `s027_regime_persistence_duration_misestimation`

For these scenarios, the pipeline has the minimum end-to-end shape:

```text
recipe
  -> generate real fixture data
  -> build candidate-facing prompt
  -> run positive/negative controls
  -> extract code from raw LLM-shaped responses
  -> execute code in sandbox per universe
  -> score outputs with generic probes
  -> write eval records
  -> optionally write recipe case manifests
```

## Candidate Interface

The scorer supports two candidate entry points:

- Direct Python code from controls.
- Raw LLM response text, where executable code is extracted from a fenced
  Python block before scoring.

The prompt builder asks the LLM to return exactly one Python code block and to
assign the final result to one of the recipe's accepted output variable names.
The scorer appends a small collector that looks for those accepted names and
normalizes scalar, dict, series, or dataframe outputs into a comparable form.

This is the bridge that lets later real LLM answers plug into the same scorer
used by smoke controls.

## Data Generation Contract

Data generation is registry-based:

```text
behavior_key -> generator(seed) -> List[UniverseFixture]
```

Each `UniverseFixture` has:

- `name`: must match a recipe universe name.
- `data`: a pandas DataFrame for that universe.

The quality gate checks that generated universe names match the recipe
universes. This keeps fixture generation small while still preventing silent
drift between recipe and data.

## Prompt Generation Contract

Prompt generation consumes only the common recipe surface:

- schema variant
- prompt variant
- load snippet
- task steps
- output contract
- known traps

It intentionally does not inspect scenario-specific scorer logic. This is the
main generalization point: if a new scenario cannot be prompted from these
fields, first ask whether the recipe is missing a reusable concept before adding
bespoke prompt code.

## Scorer Contract

The generic scorer executes the same candidate code against every universe in
the recipe, collects outputs, then evaluates recipe probes.

Currently supported probe families include:

- predicate matrix checks
- output bounds
- monotonic response
- time scaling
- output stability
- leverage bounds
- dict field bounds
- dict field monotonicity
- derived metric monotonicity
- leakage sentinel checks

The intended scaling rule is simple: add a new probe type only when multiple
scenarios need the same behavioral assertion. Otherwise prefer an existing
probe, especially `predicate_matrix`, for compact smoke coverage.

## Case Manifest

The recipe case manifest is a multi-universe manifest, not the old A/B-only
`GeneratedCase` shape.

It records:

- protocol version and case id
- behavior key, difficulty, mechanism variant
- data paths for all universes
- candidate-facing prompt and output contract
- schema manifest and universe metadata
- judge config with recipe probes
- reference behavior such as known traps and task steps
- quality checks and provenance

This manifest is the replayable artifact that ties together generated data,
prompt text, and judge configuration.

## Notebook Entry Point

Use:

```text
notebooks/recipe_eval_pipeline.ipynb
```

The notebook runs the practical sequence:

1. prompt smoke
2. case manifest generation
3. controls end-to-end eval
4. LLM-shaped raw response extraction check
5. optional real LLM eval, gated by `RUN_REAL_LLM = False`
6. reload JSONL outputs for inspection

Default notebook outputs go under:

```text
results/recipe_pipeline/
```

## CLI Entry Points

Useful commands:

```bash
venv/bin/python -m eval.smoketest.run_recipe_prompt_smoke
venv/bin/python -m eval.smoketest.run_recipe_smoke
venv/bin/python -m eval.generation.data.fixture_generation_cli --output-root /tmp/fincausal_recipe_data
venv/bin/python -m eval.runners.recipe_case_manifest --output-root /tmp/fincausal_recipe_cases
venv/bin/python -m eval.runners.recipe_eval_pipeline --candidate-source controls --output-path /tmp/fincausal_recipe_eval_controls.jsonl
```

For real LLM generation:

```bash
venv/bin/python -m eval.runners.recipe_eval_pipeline \
  --candidate-source llm \
  --llm-samples 1 \
  --config-path config.yaml \
  --output-path /tmp/fincausal_recipe_eval_llm.jsonl
```

## What Counts As End-To-End Complete

The smoke-level end-to-end loop is complete when a scenario can do all of this:

1. recipe exists and declares universes, output, probes, task steps, traps.
2. data generator emits fixtures whose names match recipe universes.
3. prompt builder renders a candidate-facing prompt from generic recipe fields.
4. candidate code or raw LLM response can be normalized into executable code.
5. sandbox executes the candidate per universe.
6. scorer evaluates recipe probes.
7. orchestrator writes an eval record.
8. optional case manifest records prompt, data paths, judge config, and
   provenance.

The production loop still needs larger-scale validation with real LLM samples,
artifact versioning policy, and broader probe support as new recipe families are
added.

## Current Hybrid Scorer Assessment

As of the May 8, 2026 iteration, the scorer is best described as a v0.5 hybrid
eval skeleton. It is good enough for iteration, debugging, and collecting real
LLM failure samples, but it is not yet stable enough to treat aggregate results
as final benchmark numbers.

Update: the scorer has been hardened so deterministic probes are explicitly
treated as evidence collectors, not final semantic judges. Failed probes become
`suspected_charges`; the LLM judge adjudicates causal validity; the decision
policy records the final triage plus diagnostics.

The current architecture is directionally right:

```text
contract gate
  -> deterministic probes produce observations and suspected charges
  -> optional LLM judge adjudicates causal verdict and failure origin
  -> decision policy returns PASS / FAIL / QUARANTINE plus diagnostics
```

Important design decisions:

- Scores are binary/triage, not arbitrary 0-100 partial credit.
- `FAIL` means a contract, runtime, or semantic failure was confirmed.
- `QUARANTINE` means the system cannot reliably attribute the failure.
- Deterministic hard probes do not directly issue final failure. They create
  hard charges. If a hard charge is adjudicated as a true failure, the final
  decision is `FAIL`.
- LLM judge may dismiss a deterministic charge as `scorer_mismatch`, but only
  when the candidate appears substantively correct and the probe is too narrow,
  checking the wrong field, or otherwise falsely accusing a correct solution.
- Judge output now uses a structured causal schema: `causal_verdict`,
  `failure_origin`, `probe_assessment`, `confidence`, and
  `required_evidence_missing`. Legacy `charge_verdicts` and `holistic_vote` are
  still emitted for compatibility.
- Invalid judge JSON is handled with JSON extraction and one JSON-only repair
  prompt. If repair fails, unresolved charges become `QUARANTINE`.
- Contradictory judge rationales are surfaced as diagnostics. If deterministic
  evidence is strong and the judge top-level verdict confirms a causal failure,
  the result remains `FAIL` with a warning, not `QUARANTINE`.

What is working well:

- Contract gate, deterministic probes, evidence pack, LLM adjudication, and
  binary decisions are now separated.
- Probe failures are represented as charges with `vote`, `severity`,
  `requires_adjudication`, and `charge_id`.
- Eval records now include `deterministic_observations`, `suspected_charges`,
  `failure_origin`, and `diagnostics`.
- `QUARANTINE` is a first-class outcome instead of being folded into `FAIL`.
- The generic scorer now covers scalar, dict, series, dataframe, multi-universe,
  and directory-backed multi-file fixtures.
- `s001_global_quantile_leakage` and `s002_expost_news_contamination` have been
  added to the new recipe pipeline alongside the later smoke scenarios.

Main weaknesses:

- LLM judge consistency is still the largest risk. The scorer now validates and
  warns on contradictory verdict/reason pairs, but calibration is still needed.
- Evidence packs are still uneven. Some scenarios provide useful row-level
  evidence, while others only expose output summaries and probe metrics.
- Pass-side blind spots remain: by default the LLM judge is called only when
  deterministic probes produce suspected charges. A future `sample_passes` or
  `judge_policy` mode should audit deterministic passes.
- Contract checks are improving but incomplete. Missing top-level assignment,
  forbidden imports, printed-only results, required dataframe fields, null
  policies, and dtype policies should become systematic contract checks.

Observed LLM judge issues from the latest sample run:

- `s010_fat_tail_fiduciary_discrimination`: the candidate endorsed all
  universes, including fat-tail failures. The LLM reason correctly described a
  candidate failure but mislabeled the charge as `scorer_mismatch`. The judge
  prompt and verdict validator were tightened so candidate-failure language is
  normalized toward `true_failure` / `semantic_failure`.
- `s006_inventory_induced_skew`: the judge treated some candidate failures as
  scorer mismatch or inconclusive. This needs better probe evidence and stricter
  scorer-mismatch semantics.
- `s008_vol_signature_misclassification`: the judge reason indicated a semantic
  failure, but the label leaned toward scorer mismatch. This is another symptom
  of loose judge taxonomy.
- `s003_covariance_inversion_stability`: the judge omitted a hard charge verdict,
  producing `QUARANTINE`. Missing hard-charge verdicts need a deterministic or
  evidence-strength-aware fallback.

Recommended next priorities:

1. Add a formal `JudgeVerdict` dataclass or pydantic schema around the current
   structured judge fields.
2. Refine the decision policy into ordered rules:

   ```text
   contract fail -> FAIL
   runtime ambiguous -> QUARANTINE
   confirmed hard causal failure -> FAIL
   unresolved hard charge -> QUARANTINE
   scorer mismatch -> QUARANTINE / scorer_mismatch bucket
   contradictory judge rationale + strong evidence -> FAIL with warning
   all charges dismissed + deterministic support -> PASS or audited PASS
   ```

3. Add probe-specific evidence:
   - `s010`: worst day, drawdown, Sharpe, tail rows, and endorsement rationale.
   - `s003`: leverage, weight distribution, and stability deltas.
   - `s006`: baseline/treatment quote states and derived mid-price movement.
   - `s008`: persistent trend vs transient shock output comparisons.
4. Make `scorer_mismatch` harder to claim. The judge should only use it when it
   can state why the candidate satisfies the causal requirement and why the
   deterministic probe is wrong.
5. Build a small judge calibration set per scenario:
   - known positive
   - known negative
   - plausible-looking but wrong candidate
   - scorer-mismatch candidate
   - contradictory-judge-output fixture

In short: the pipeline has the right engineering shape and is now useful for
scenario expansion and failure analysis. The next hardening pass should focus on
judge calibration, evidence completeness, and pass-side audit policy.

## May 9 Scenario Expansion Handoff

Added two new scenario families after `s013`-`s017`, using the same recipe-first
pattern:

- `s018`-`s022`: non-stationary spurious regression variants. These share
  `nonstationary_spurious_regression` as the mechanism and cover raw-level
  Granger tests, ignored cointegration in pairs trading, spurious factor
  significance, unchecked non-stationary residuals, and false lead-lag
  relationships.
- `s023`-`s027`: regime structural-break variants. These share
  `multi_break_regime_misclassification` as the mechanism and cover multiple
  structural breaks, slow drift, correlation breaks, volatility clustering vs
  durable shifts, and persistence/duration misestimation.

Implementation notes:

- Recipes live in `eval/recipes/scenario_recipes.py`.
- Fixture generators are registered in `eval/framework/generator/recipe_data_generator.py`.
- Positive/negative controls are registered in `eval/smoketest/smoke_controls.py`.
- The generic scorer now supports a reusable `code_pattern_present` probe in
  `eval/framework/scorer/generic_recipe_scorer.py`.
- The non-stationarity family uses clean stationary panels and shock panels with
  post-T unit-root injection. Controls require either stationarity-style tests or
  returns/differencing before reporting relationship diagnostics.
- The regime family uses clean stable panels and shock panels with three
  expected break windows: an abrupt high-volatility/trend break, a gradual drift
  window, and a later persistence/correlation break. Controls require multiple
  regime labels, change-point flags in each expected window, and no major false
  change points in the clean universe.

Verification run with the project venv:

```bash
venv/bin/python -m compileall eval
venv/bin/python -c "from eval.smoketest.run_recipe_smoke import run_recipe_smoke; keys=['s018_statistical_spurious_granger_causality','s019_statistical_ignored_cointegration_pairs_trading','s020_statistical_spurious_factor_significance','s021_statistical_non_stationary_residuals','s022_statistical_false_lead_lag_relationships','s023_regime_multiple_structural_breaks_misclassification','s024_regime_slow_drift_misclassification','s025_regime_correlation_break_misclassification','s026_regime_volatility_clustering_vs_regime_shift','s027_regime_persistence_duration_misestimation']; print({k: run_recipe_smoke(k)[k]['ok'] for k in keys})"
venv/bin/python -m eval.generation.data.fixture_generation_cli --behavior-key s023_regime_multiple_structural_breaks_misclassification
venv/bin/python -m eval.generation.data.fixture_generation_cli --behavior-key s027_regime_persistence_duration_misestimation
```

All compact smoke checks for `s018`-`s027` returned `True`. Data generation spot
checks for `s023` and `s027` produced the expected `clean` and `shock`
universes, each with 480 rows and no missing or extra universes.

## How To Add The Next Scenario

Use this order:

1. Add or update the recipe first.
2. Try to express the score with existing probes.
3. Add a small data generator registered by `behavior_key`.
4. Add one positive and one negative smoke control.
5. Run prompt smoke and scorer smoke.
6. Generate a case manifest.
7. Run the orchestrator with controls.
8. Only then run real LLM samples.

If step 2 requires bespoke scoring, pause and decide whether the new behavior is
actually a reusable probe family.
