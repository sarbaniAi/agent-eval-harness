"""
Evaluation Runner — Config-driven evaluation execution engine.
Reads YAML config → builds scorers → runs evaluation → reports results.
"""
import mlflow
import json
from typing import Callable, List, Dict, Any, Optional
from .eval_config import EvalHarnessConfig, load_config
from .scorer_registry import build_scorer_list
from .optimizer import PromptOptimizer


class EvalRunner:
    """
    Main entry point for the evaluation harness.

    Usage:
        config = load_config("config/customer_support.yaml")
        runner = EvalRunner(config)
        results = runner.run(eval_data=data, predict_fn=my_agent)
    """

    def __init__(self, config: EvalHarnessConfig):
        self.config = config
        self.scorers = build_scorer_list(config.scorers, config.llm_endpoint)
        print(f"✅ EvalRunner initialized: {config.name}")
        print(f"   Agent type: {config.agent_type}")
        print(f"   Scorers: {len(self.scorers)} active")
        print(f"   Thresholds: pass_rate={config.thresholds.overall_pass_rate}, security={config.thresholds.security_pass_rate}")

    def run(self, eval_data: List[Dict], predict_fn: Callable) -> Dict[str, Any]:
        """
        Run evaluation against the dataset.

        Args:
            eval_data: List of {inputs, expectations} dicts
            predict_fn: Function that takes inputs dict and returns outputs dict

        Returns:
            Dict with run_id, metrics, pass_rate, failures
        """
        print(f"\n🚀 Running evaluation: {len(eval_data)} test cases × {len(self.scorers)} scorers")

        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=self.scorers,
        )

        # Parse results
        traces = mlflow.search_traces(run_id=results.run_id)
        metrics = self._compute_metrics(traces)

        print(f"\n📊 Results:")
        print(f"   Overall pass rate: {metrics['overall_pass_rate']:.1%}")
        print(f"   Security pass rate: {metrics['security_pass_rate']:.1%}")
        print(f"   Meets threshold: {'✅' if metrics['meets_threshold'] else '🚨 NO'}")

        return {
            "run_id": results.run_id,
            "metrics": metrics,
            "traces": traces,
            "config": self.config,
        }

    def run_with_optimization(
        self, eval_data: List[Dict], predict_fn_factory: Callable,
        current_prompt: str, client
    ) -> Dict[str, Any]:
        """
        Run evaluation → if below threshold → optimize → re-evaluate.

        Args:
            eval_data: Evaluation dataset
            predict_fn_factory: Function that takes (prompt) and returns a predict_fn
            current_prompt: Current system prompt
            client: OpenAI client for optimization LLM calls
        """
        # Step 1: Evaluate baseline
        print(f"\n{'═'*60}")
        print(f"📊 Evaluating BASELINE")
        baseline_predict = predict_fn_factory(current_prompt)
        baseline = self.run(eval_data, baseline_predict)

        if baseline["metrics"]["meets_threshold"]:
            print(f"✅ Baseline meets threshold — no optimization needed")
            return {"baseline": baseline, "optimized": None, "winner": "baseline"}

        # Step 2: Identify failures
        failures = self._extract_failures(baseline["traces"])
        print(f"\n⚠️  {len(failures)} failures found — running optimization strategies...")

        # Step 3: Run optimization strategies
        optimizer = PromptOptimizer(self.config, client)
        optimized_prompts = optimizer.optimize(current_prompt, failures)

        # Step 4: Evaluate each optimized version
        best = {"name": "baseline", "pass_rate": baseline["metrics"]["overall_pass_rate"], "prompt": current_prompt}

        for strategy_name, optimized_prompt in optimized_prompts.items():
            print(f"\n{'─'*40}")
            print(f"📊 Evaluating: {strategy_name}")
            opt_predict = predict_fn_factory(optimized_prompt)
            opt_result = self.run(eval_data, opt_predict)
            opt_rate = opt_result["metrics"]["overall_pass_rate"]

            if opt_rate > best["pass_rate"]:
                best = {"name": strategy_name, "pass_rate": opt_rate, "prompt": optimized_prompt, "result": opt_result}

        # Step 5: Report winner
        print(f"\n{'═'*60}")
        print(f"🏆 Winner: {best['name']} ({best['pass_rate']:.1%})")
        improvement = best["pass_rate"] - baseline["metrics"]["overall_pass_rate"]
        print(f"   Improvement: {improvement:+.1%}")

        return {
            "baseline": baseline,
            "winner": best["name"],
            "winner_pass_rate": best["pass_rate"],
            "winner_prompt": best["prompt"],
            "improvement": improvement,
        }

    def _compute_metrics(self, traces) -> Dict[str, Any]:
        """Compute pass rates from evaluation traces."""
        scorer_stats = {}

        for _, trace in traces.iterrows():
            assessments = trace.get("assessments", []) or []
            if isinstance(assessments, list):
                for a in assessments:
                    name = getattr(a, 'name', 'unknown')
                    value = getattr(a, 'value', None)
                    if name not in scorer_stats:
                        scorer_stats[name] = {"total": 0, "passed": 0}
                    scorer_stats[name]["total"] += 1
                    if value == True or value == "pass":
                        scorer_stats[name]["passed"] += 1

        total_assessments = sum(s["total"] for s in scorer_stats.values())
        total_passed = sum(s["passed"] for s in scorer_stats.values())
        overall_pass_rate = total_passed / max(total_assessments, 1)

        # Security-specific pass rate
        security_names = {"pii_leakage", "prompt_injection_detection", "cross_customer_data",
                          "unauthorized_actions", "safety", "error_propagation"}
        sec_total = sum(s["total"] for n, s in scorer_stats.items() if n in security_names)
        sec_passed = sum(s["passed"] for n, s in scorer_stats.items() if n in security_names)
        security_pass_rate = sec_passed / max(sec_total, 1)

        return {
            "overall_pass_rate": overall_pass_rate,
            "security_pass_rate": security_pass_rate,
            "per_scorer": scorer_stats,
            "total_assessments": total_assessments,
            "meets_threshold": overall_pass_rate >= self.config.thresholds.overall_pass_rate,
            "meets_security_threshold": security_pass_rate >= self.config.thresholds.security_pass_rate,
        }

    def _extract_failures(self, traces) -> List[Dict]:
        """Extract failed assessments from traces."""
        failures = []
        for _, trace in traces.iterrows():
            assessments = trace.get("assessments", []) or []
            if isinstance(assessments, list):
                for a in assessments:
                    if getattr(a, 'value', None) == False:
                        failures.append({
                            "type": getattr(a, 'name', 'unknown'),
                            "rationale": getattr(a, 'rationale', ''),
                        })
        return failures
