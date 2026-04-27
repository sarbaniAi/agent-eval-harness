"""
Configuration loader for the Agent Evaluation Harness.
Reads YAML config and provides typed access to all settings.
"""
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class ScorerConfig:
    name: str
    type: str  # builtin, llm_judge, code, guidelines
    enabled: bool = True
    layer: str = ""  # orchestrator, sub_agent, workflow, safety
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdConfig:
    overall_pass_rate: float = 0.90
    security_pass_rate: float = 0.99
    orchestrator_pass_rate: float = 0.95
    regression_tolerance: float = 0.02


@dataclass
class MonitoringScorerConfig:
    name: str
    sample_rate: float = 1.0


@dataclass
class OptimizationStrategy:
    name: str
    enabled: bool = True
    scope: str = "agent"  # agent, per_agent, orchestrator
    description: str = ""


@dataclass
class EvalHarnessConfig:
    """Complete configuration for an evaluation harness run."""
    # Metadata
    name: str
    version: str
    description: str
    agent_type: str  # single_agent | multi_agent

    # Environment
    catalog: str
    schema: str
    llm_endpoint: str
    strong_llm_endpoint: str
    embedding_endpoint: str

    # Scorers
    scorers: List[ScorerConfig]

    # Thresholds
    thresholds: ThresholdConfig

    # Monitoring
    monitoring_scorers: List[MonitoringScorerConfig]

    # Optimization
    optimization_strategies: List[OptimizationStrategy]
    auto_optimize_trigger: str = ""
    max_iterations: int = 3
    require_human_approval: bool = True

    # Agent config (raw — agent-specific)
    agent_config: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    dataset_config: Dict[str, Any] = field(default_factory=dict)
    feedback_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_schema(self) -> str:
        return f"{self.catalog}.{self.schema}"


def load_config(yaml_path: str) -> EvalHarnessConfig:
    """Load and parse a YAML configuration file."""
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    meta = raw.get("metadata", {})
    env = raw.get("environment", {})
    eval_cfg = raw.get("evaluation", {})
    mon = raw.get("monitoring", {})
    opt = raw.get("optimization", {})

    scorers = [
        ScorerConfig(
            name=s["name"], type=s["type"], enabled=s.get("enabled", True),
            layer=s.get("layer", ""), description=s.get("description", ""),
            config=s.get("config", {})
        )
        for s in eval_cfg.get("scorers", [])
    ]

    thresholds = ThresholdConfig(**{
        k: v for k, v in eval_cfg.get("thresholds", {}).items()
        if k in ThresholdConfig.__dataclass_fields__
    })

    monitoring_scorers = [
        MonitoringScorerConfig(name=m["name"], sample_rate=m.get("sample_rate", 1.0))
        for m in mon.get("scorers", [])
    ]

    strategies = [
        OptimizationStrategy(
            name=s["name"], enabled=s.get("enabled", True),
            scope=s.get("scope", "agent"), description=s.get("description", "")
        )
        for s in opt.get("strategies", [])
    ]

    auto_opt = opt.get("auto_optimize", {})

    return EvalHarnessConfig(
        name=meta.get("name", ""),
        version=meta.get("version", "1.0"),
        description=meta.get("description", ""),
        agent_type=meta.get("agent_type", "single_agent"),
        catalog=env.get("catalog", ""),
        schema=env.get("schema", ""),
        llm_endpoint=env.get("llm_endpoint", ""),
        strong_llm_endpoint=env.get("strong_llm_endpoint", ""),
        embedding_endpoint=env.get("embedding_endpoint", ""),
        scorers=scorers,
        thresholds=thresholds,
        monitoring_scorers=monitoring_scorers,
        optimization_strategies=strategies,
        auto_optimize_trigger=auto_opt.get("trigger", ""),
        max_iterations=auto_opt.get("max_iterations", 3),
        require_human_approval=auto_opt.get("require_human_approval", True),
        agent_config=raw.get("agent", {}),
        data_config=raw.get("data", {}),
        dataset_config=eval_cfg.get("dataset", {}),
        feedback_config=raw.get("feedback", {}),
    )
