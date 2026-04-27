# Agent Evaluation Harness for Databricks

**Configuration-driven evaluation, monitoring, and optimization for AI agents on Databricks with MLflow 3.0+**

Drop in your agent, pick a config, and get production-grade evaluation in 5 minutes.

## Quick Start

```
1. Run notebooks/00_install_setup        → Creates sample data + tables
2. Run notebooks/01_quickstart_customer_support   → Single-agent eval (RAG + tools)
3. Run notebooks/02_quickstart_document_processing → Multi-agent eval (4-layer)
```

## What's Included

### Two Example Use Cases

| Use Case | Agent Type | Scorers | Config |
|----------|-----------|---------|--------|
| **Customer Support** | Single-agent (RAG + tools) | PII, injection, tool usage, KB grounding, guidelines, safety | `config/customer_support.yaml` |
| **Document Processing** | Multi-agent (5 sub-agents) | Routing, sequencing, sub-agent quality, handoffs, decisions, efficiency | `config/document_processing.yaml` |

### Reusable Harness Framework

```
harness/
├── eval_config.py      # YAML config loader with typed access
├── eval_runner.py      # Config-driven evaluation engine
├── scorer_registry.py  # 15+ pluggable scorers (code + LLM)
├── optimizer.py        # 3 prompt optimization strategies
└── __init__.py
```

### How to Add Your Own Agent

1. **Copy a config**: `cp config/customer_support.yaml config/my_agent.yaml`
2. **Edit the config**: Change scorers, thresholds, guidelines, tools
3. **Write your predict function**:
```python
def my_predict(inputs: dict) -> dict:
    return my_agent.run(inputs["question"])
```
4. **Run evaluation**:
```python
from harness import load_config, EvalRunner
config = load_config("config/my_agent.yaml")
runner = EvalRunner(config)
results = runner.run(eval_data=my_data, predict_fn=my_predict)
```

## Scorer Library

### Code-Based (Free, Deterministic)
| Scorer | What It Checks |
|--------|---------------|
| `pii_leakage` | Email, phone, credit card, Aadhaar patterns in response |
| `prompt_injection_detection` | Known injection patterns in input |
| `cross_customer_data` | Response mentions unauthorized entities |
| `routing_correctness` | Orchestrator called correct agents |
| `workflow_sequencing` | Agents called in correct order |
| `handoff_integrity` | Context preserved between agents |
| `path_efficiency` | Minimum agents called |
| `end_to_end_decision` | Final decision matches expected |
| `extraction_accuracy` | Field-level extraction F1 |
| `unauthorized_actions` | No approvals without validation |
| `error_propagation` | Sub-agent failures handled gracefully |

### LLM Judges
| Scorer | What It Checks |
|--------|---------------|
| `Guidelines` | Custom natural-language rules (configurable in YAML) |
| `Safety` | MLflow built-in safety judge |
| `RetrievalGroundedness` | Response grounded in retrieved context |
| `Correctness` | Response accuracy vs ground truth |

## Prompt Optimization Strategies

| Strategy | How It Works |
|----------|-------------|
| **Failure-Targeted Patching** | Analyzes failures → adds specific rules |
| **Few-Shot Injection** | Adds attack examples + correct responses |
| **Constitutional Rewrite** | Rewrites with principles + self-check |
| **Routing Rule Refinement** | Optimizes orchestrator logic (multi-agent) |

## Requirements

- Databricks workspace with Unity Catalog
- Foundation Model API (Claude or GPT endpoints)
- MLflow 3.0+
- Serverless compute

## Author

**Sarbani Maiti** — Solutions Architect, Databricks
