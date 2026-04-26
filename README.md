# FinCausal
**A Causal-First LLM Benchmarking & Agentic Framework for Systematic Trading**

LLMs are statistically brilliant but causally blind.  
In quantitative finance, this manifests as look-ahead bias, hallucinated alpha, and unacknowledged fat-tail risk — failures that are invisible in backtests and catastrophic in production.

FinCausal is an **assistive tool for algorithmic traders**, not an automated trading system.  
It provides two interlocking layers: a rigorous causal failure benchmark (Layer 1) and a constrained agentic system that enforces causal correctness at code-generation time (Layer 2).

---

## 🏗️ Architecture

### Layer 1 — Causal Failure Mode Taxonomy (Eval)

Built around **five causal pillars** — the root causes of LLM failure in financial reasoning:

| Pillar | What It Defends | Example Failure |
|---|---|---|
| **Temporal Causality** | Time's arrow is inviolable |forward-fill leakage |
| **Statistical Causality** | True data-generating process | Outlier-driven spurious signals |
| **System Feedback Causality** | Observer ≠ Participant | Ignoring market impact & inventory risk |
| **Regime Causality** | Structural breaks ≠ noise | Treating a flash crash as normal vol |
| **Risk Causality** | Fiduciary duty over Sharpe | Missing fat-tail blowup risk |

#### Dynamic Counterfactual Engine
Each scenario is evaluated across ** parallel universes**:
- **Universe A (Baseline):** Historical data with minor random noise
- **Universe B (Shock):** Identical data with a dynamic shock injected at time *T* (e.g., 50× volatility spike, 5% corrupted rows)

> **Core Principle:** Causally correct logic is *immune* to future shocks. If Universe B's shock changes model output *before* time *T*, the causal chain is broken.

#### Dual-Axis Scoring (per scenario)
- **Axis 1 — Causal Integrity Score (CIS): — Pass/fail gate. Zero tolerance for causal violations.
- **Axis 2 — Functional Utility Score (FUS): — Gradient signal for iterative improvement.
- **Alignment & Auditability Pillar** — Bonus/penalty modifier for traceability and documentation quality.

#### Aggregation: Dual-Layer Radar Chart
- **Inner Ring (Causal Backbone):** CIS scores across all 11 scenarios. Any vertex at zero = overall fail.
- **Outer Halo (Utility Aura):** Utility and engineering scores layered on top.
- **Global Status:** `PASS` | `CAUTION` (causal pass, low utility) | `FAIL`

#### Example of output
![Chart Example](/assets/figure-example.png)

---

### Layer 2 [In Build] — Constrained Agentic System (LangGraph)

#### Multi-Node Graph: Causal Violation Prevention

```text
                     [Init & Intent Node]
                               ↓
               (mount sandbox + lock time_cursor + load docs)
    ┌──────────────────────────────────────────────────────────────┐
    │                     Research Subgraph                        │
    │  ┌──────────────────────────────┐   ┌────────────────────-─┐ │
    │  │   [Research Planner]         │   │ [Constraint Reasoner]│ │
    │  │   ReAct • Macro logic        │ ↔ │ DetailedCausalPlan   │ │
    │  │   + TaskContract             │   │ + GUARD              │ │
    │  └──────────────────────────────┘   └──────────────────────┘ │
    └──────────────────────────────┬───────────────────────────────┘
                                   ↓
                      [Constrained Code Generator]
                 (step-by-step code + inline comments)
                                   ↓
                     [Static Linter Verifier]
                     (AST + Regex + CAUSAL GUARD check)
                         ├── Pass ──→ [Teardown Node]
                         │               (shutil.rmtree)
                         └── Fail ──→ Back to Generator
                                  ↓
                            Final Output
                                  ↓
                         (Independent Layer1 Eval)


```


## 🚀 Quickstart

```bash
git clone https://github.com/yourusername/FinCausal.git
cd FinCausal
```

Open your ```config.yaml``` file and modify the ```model_name``` and related settings.
Make sure your ```.env``` file contains the correct keys:
```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

#### Create a virtual environment
```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

#### Run the eval benchmark
```bash
# Run only specific scenarios
python eval/core/main.py --scenarios S001
```


## 🤝 Contributing
 PRs expanding the taxonomy or linter **must** include:
1. A failing test fixture (the specific LLM hallucination being addressed)
2. The causal pillar it maps to
3. Universe A/B scenario definition if adding a new eval scenario