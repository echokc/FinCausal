# Recipe Generation Principles

This document captures the rules we use when adding temporal scenario recipes.
The goal is to make new scenarios fast to write, easy to compare, and eventually
usable as a source of truth for prompt/data/scorer generation.

## Goal

A recipe should describe the experimental design of a scenario, not reimplement
the scenario.

Good recipes answer:

- What behavior is being tested?
- What data schema does the candidate see?
- What universes are compared?
- What output is expected?
- What probes decide whether the behavior passed?
- What traps should the candidate avoid?

Recipes should stay thin. If a recipe starts containing bespoke scoring logic,
data construction code, or long behavior-specific branching, it is probably
recreating the old large-file problem.

## Preferred Shape

Use the generic recipe primitives first:

- `multi_universe_output_recipe(...)` for most legacy scenarios.
- `output(...)` for scalar, dict, dataframe, series, or vector outputs.
- `universe(...)` for each baseline/treatment/stress/leakage/trap case.
- schema helpers such as `price_series_schema_variants(...)`,
  `daily_return_schema_variants(...)`, `trade_tape_schema_variants(...)`.
- probe helpers such as `output_bounds_probe(...)`,
  `monotonic_response_probe(...)`, `field_bounds_probe(...)`,
  `field_monotonic_probe(...)`, `leakage_sentinel_probe(...)`,
  `predicate_matrix_probe(...)`.

Only add a new primitive when several scenarios need the same idea and existing
primitives would force awkward free-form dictionaries.

## Recipe Fields

Every recipe should include these concepts:

- `behavior_key`: stable scenario identifier. It should name the causal or
  robustness behavior, not just the file number.
- `pillar`: one of the benchmark causal pillars, such as `Temporal Causality`,
  `Statistical Causality`, `System Feedback Causality`, `Regime Causality`, or
  `Risk Causality`.
- `difficulty`: coarse difficulty label.
- `default_schema_variant`: the canonical schema key.
- `default_prompt_variant`: the canonical prompt framing key.
- `schema_variants`: structured column/file roles.
- `prompt_variants`: role, causal requirement, difficulty, and surface domains.
- `distribution_variants`: short names for the data regimes.
- `mechanism_variant`: the causal failure mode or mechanism under test.
- `output`: expected candidate output variable and semantic.
- `universes`: all baseline/treatment/stress/leakage/trap cases.
- `probes`: behavioral assertions over outputs.
- `task_steps`: concise implementation obligations for the candidate.
- `load_snippet`: the ordinary way the candidate loads the provided data.
- `known_traps`: failure modes the scenario is designed to expose.

## Naming

Names should be stable and semantic.

Prefer:

- `fat_tail_shock`
- `leakage_sentinel`
- `contemporaneous_trap`
- `leading_signal`
- `micro_noise_weight_stability`

Avoid names that only describe implementation accidents:

- `df1`
- `case_a`
- `weird_variant`
- `scenario_thing`

The numeric scenario id can remain in the Python constant name, but the recipe
content should describe the behavior itself.

## Universes

A universe is one member of the behavioral comparison set.

Use roles consistently:

- `baseline`: ordinary reference case.
- `treatment`: causally changed case.
- `stress`: harder version of the same behavior.
- `leakage`: sentinel case designed to catch look-ahead or future contamination.
- `trap`: case that attracts a naive but wrong strategy.
- `positive`: case that should produce an approving/high/active output.
- `negative`: case that should produce a rejecting/low/inactive output.

Each universe should contain:

- a short description of what is different,
- expected observable behavior,
- optional intervention metadata if it is a transformed version of another
  universe.

Do not bury scorer thresholds only in prose. Put them in `expected` or probes.

## Outputs

The output declaration should describe the candidate-facing contract.

Use:

- `kind="scalar"` for a single number or binary decision.
- `kind="dict"` for named metrics like `bid_price`, `ask_price`, `net_position`.
- `kind="dataframe"` for tabular outputs like portfolio weights.
- `kind="series"` for row-aligned outputs like protection flags.
- `kind="vector"` for ordered numeric arrays.

Set `semantic` to the behavioral meaning, not only the variable name.

Examples:

- variable `position`, semantic `position`
- variable `hedge_notional`, semantic `hedge_notional`
- variable `quote_state`, semantic `quote_state`
- variable `portfolio_weights`, semantic `portfolio_weights`

Use `accepted_names` when legacy scorers or generated code may expose equivalent
names.

## Probes

Probes are the most important part of the recipe. They describe pass/fail
behavior independent of how the candidate implemented the solution.

Use the narrowest existing probe that expresses the behavioral requirement:

- `output_bounds_probe`: one universe should produce a value within bounds.
- `monotonic_response_probe`: treatment should increase/decrease relative to
  baseline.
- `field_bounds_probe`: one field in a dict output should be bounded.
- `field_monotonic_probe`: one dict field should move between universes.
- `derived_metric_monotonic_probe`: an evaluator-derived metric should move.
- `output_stability_probe`: small perturbation should not materially change the
  output.
- `leverage_bounds_probe`: portfolio weights should obey leverage/investment
  constraints.
- `config_counterfactual_probe`: config changes should alter behavior in the
  expected direction.
- `time_scaling_probe`: frequency-to-horizon scaling should match the intended
  rule.
- `leakage_sentinel_probe`: a sentinel future change should not alter the clean
  reference output.
- `predicate_matrix_probe`: several universes each have simple predicate
  expectations.

Use `predicate_matrix_probe` for compact case tables, especially binary
decisions, flag checks, and mixed predicates across many universes.

Do not add a bespoke probe just because one scenario is unusual. Add a bespoke
probe only when the same pattern will recur.

## Predicate Matrix

`predicate_matrix_probe` is intentionally a bridge primitive. It lets us encode
case-specific expectations before we have a fully typed predicate DSL.

Good uses:

```python
predicate_matrix_probe(
    name="fiduciary_tail_risk_matrix",
    output_semantic="endorsement",
    expectations={
        "fat_tail": {"equals": 0},
        "clean": {"equals": 1},
        "mild_tail": {"equals": 0},
        "sharpe_hack": {"equals": 0},
        "vol_spike": {"equals": 1},
    },
)
```

```python
predicate_matrix_probe(
    name="robust_breakout_flag_matrix",
    output_semantic="protection_flags",
    expectations={
        "poisoned_breakout": {"flag_at": {140: True, 80: False}},
        "poison_without_signal": {"sum_equals": 0},
    },
)
```

If predicate dictionaries start growing complex, that is a signal to introduce
typed predicate specs such as `Equals`, `Range`, `AbsMax`, `FlagAt`, `SumEquals`,
`Delta`, or `RatioRange`.

## Schemas

Schemas should describe what the candidate sees, not how the data was generated.

Add a schema helper when a column layout is likely to recur:

- price series,
- price/volume series,
- daily returns,
- trade tape,
- execution log,
- return panel,
- trade/news joins.

Do not create one-off schema helpers for one scenario unless it clarifies a
family of future scenarios.

## Prompt Variants

Prompt variants should be short but explicit:

- `framing`: who the candidate is acting as.
- `causal_requirement`: the core behavior being tested.
- `difficulty_level`: numeric coarse level.
- `surface_domains`: domain labels that can vary without changing the mechanism.

The prompt variant should not contain long scenario-specific data explanations.
Those belong in universes, task steps, and scorer/data generation.

## Task Steps

Task steps should be candidate obligations, not an answer key.

Good:

- "Load the price history from DATA_PATH."
- "Estimate whether volume spikes lead future returns rather than merely
  coincide with current price moves."
- "Return a boolean pandas Series aligned to the input rows."

Avoid:

- long algorithm recipes that overfit the scorer,
- exact hidden constants unless they are part of the public task,
- scorer implementation details.

## Known Traps

Known traps should explain why naive solutions fail.

They are not just documentation. They help us:

- audit whether probes cover the intended failure modes,
- generate adversarial variants,
- compare scenarios by mechanism,
- avoid writing many scenarios that test the same thing under different names.

Each trap should have:

- `type`: short stable key,
- `description`: one-sentence explanation.

## When To Generalize

Generalize when at least two scenarios share a shape.

Good generalization:

- extract a schema helper used by several recipes,
- extract a probe helper used by several recipes,
- extract a recipe factory for a recurring scenario pattern,
- type a predicate family that appears repeatedly.

Bad generalization:

- moving one large behavior file into a second large helper file,
- hiding scenario-specific logic behind a generic-sounding class,
- adding abstractions before the repeated pattern is visible,
- turning every unusual scenario into a new framework.

The recipe layer should make new examples faster to write and easier to review.
If an abstraction makes a new scenario harder to understand, it is probably the
wrong abstraction.

## Source Of Truth Direction

Current recipes are structured descriptions. The next scalability step is to
make them executable specifications.

The intended direction:

1. recipe describes schema, universes, outputs, probes, and traps.
2. data builders consume universe/intervention specs.
3. prompt builders consume schema, task steps, and prompt variants.
4. scorer adapters consume probes.
5. scenario files become thin declarations plus data-generation hooks only when
   genuinely necessary.

Until then, keep scorer thresholds and recipe probes aligned manually.

## Checklist For A New Scenario

Before adding a new scenario, answer:

- What is the causal or robustness mechanism?
- What is the naive failure mode?
- What are the minimum universes needed to expose it?
- Is there a leakage sentinel?
- What output shape does the candidate produce?
- Which existing probe expresses the pass/fail condition?
- Is a new schema helper genuinely reusable?
- Are thresholds represented in probes, not only prose?
- Can the recipe be read without opening the scorer?
- Would this recipe still make sense if we generated ten variants of the same
  mechanism?

If the answer to the last question is no, the recipe is probably too
scenario-specific.

## Implementation Notes

For the current implementation summary, package map, notebook entry point, and
recipe-to-eval handoff contract, see
`docs/recipe_eval_pipeline_summary.md`.
