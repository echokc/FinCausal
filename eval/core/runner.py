import re
import json
import logging
import os
import importlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a top quantitative researcher.

When writing code, please follow these guidelines:
1. Briefly explain your business logic and statistical methods before writing the code.
2. All Python code must be wrapped in ```python ... ``` blocks.
3. Path Handling: If variables are preset in the environment (e.g., DATA_PATH, FILE_PATH, PATH_A, PATH_B), please use these variables to read data.
4. Output Results: Ensure the final result is assigned to the specific variable required by the task (e.g., result_df, portfolio_weights, etc.).
5. Risk Awareness: Explicitly point out potential financial traps in the task."""


def extract_code_and_reasoning(response: str) -> tuple[str, str]:
    code_blocks = re.findall(r'```python\s*(.*?)```', response, re.DOTALL)
    generated_code = '\n'.join(block.strip() for block in code_blocks)
    reasoning = re.sub(r'```python\s*.*?```', '', response, flags=re.DOTALL).strip()
    return generated_code, reasoning


class EvalRunner:
    def __init__(self, llm: BaseChatModel, cache_dir: Optional[str] = None, results_dir: str = "eval/results"):
        self.llm = llm
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_cache")
        self.results_dir = results_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def run_scenario(self, scenario_id: str, scenario: dict, use_cache: bool = True) -> dict:
        logger.info(f"Running scenario {scenario_id}: {scenario['name']}")
        cache_path = os.path.join(self.cache_dir, f"{scenario_id}.json")
        answer = None

        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    answer = json.load(f).get("answer")
                logger.info(f"  [CACHE HIT] {scenario_id}")
            except Exception as e:
                logger.warning(f"  [CACHE ERROR] {scenario_id}: {e}")

        if not answer:
            logger.info(f"  [API CALL] {scenario_id}")
            messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=scenario["prompt"])]
            try:
                response = self.llm.invoke(messages)
                answer = response.content
                if use_cache:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump({"answer": answer, "timestamp": str(datetime.now())}, f, indent=2, ensure_ascii=False)
                    logger.info(f"  [CACHE SAVE] {scenario_id}")
            except Exception as e:
                logger.error(f"API call failed for {scenario_id}: {e}")
                return {"scenario_id": scenario_id, "scores": {"total": 0, "max": 100, "details": {"error": str(e)}}}

        try:
            generated_code, reasoning = extract_code_and_reasoning(answer)
            scores = scenario["scoring_fn"](generated_code, reasoning)
            result = {
                "scenario_id": scenario_id,
                "scenario_name": scenario["name"],
                "failure_mode": scenario["failure_mode"],
                "difficulty": scenario.get("difficulty", "unknown"),
                "answer": answer,
                "generated_code": generated_code,
                "reasoning": reasoning,
                "scores": scores,
            }
            logger.info(f"{scenario_id} -> {scores.get('total', 0)}/{scores.get('max', 100)}")
            return result
        except Exception as e:
            logger.error(f"Scoring error for {scenario_id}: {e}", exc_info=True)
            return {"scenario_id": scenario_id, "scores": {"total": 0, "max": 100, "details": {"error": str(e)}}}

    def run_eval(self, scenarios: dict, scenario_ids: Optional[List[str]] = None, use_cache: bool = True, save_results: bool = True) -> List[dict]:
        # Load scorers
        for sid, meta in scenarios.items():
            try:
                module = importlib.import_module(f"scorer.{sid.lower()}")
                meta["scoring_fn"] = getattr(module, f"score_{sid.lower()}")
            except Exception as e:
                logger.warning(f"Failed to load scorer for {sid}: {e}")

        targets = scenario_ids or list(scenarios.keys())
        results = []

        for sid in targets:
            if sid not in scenarios:
                logger.warning(f"Scenario {sid} not found — skipping")
                continue
            results.append(self.run_scenario(sid, scenarios[sid], use_cache=use_cache))

        self._log_summary(results)

        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.results_dir, f"eval_results_{timestamp}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {path}")

        return results

    def _log_summary(self, results: List[dict]) -> None:
        totals = [r["scores"]["total"] for r in results if "scores" in r]
        if not totals:
            return
        logger.info("=" * 60)
        logger.info(f"Done — {len(results)} scenarios | avg: {sum(totals)/len(totals):.1f}/100")
        for r in results:
            s = r.get("scores", {})
            logger.info(f"  {r['scenario_id']} ({r.get('difficulty','N/A')}): {s.get('total',0)}/{s.get('max',100)}")
        logger.info("=" * 60)