"""Shared paths, the prompt/config registry, and cost accounting."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
REPORTS_DIR = ROOT / "reports"
TRACES_DIR = ROOT / "traces"
REGRESSION_DIR = ROOT / "regression"
GOLDEN_PATH = REGRESSION_DIR / "golden.jsonl"
PRODUCTION_PATH = RESULTS_DIR / "production.json"

# Local, file-based tracking: a SQLite backend store plus an artifact directory,
# both inside track_b_agentic so nothing depends on a tracking server.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
ARTIFACT_ROOT = ROOT / "mlartifacts"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "track-b-agentic-prompts")

# The model is fixed for every tracked run (see README); what varies is the
# prompt version and the loop/retrieval configuration.
PROVIDER = "groq"
MODEL = "openai/gpt-oss-120b"
JUDGE_PROVIDER = "groq"
JUDGE_MODEL = "openai/gpt-oss-120b"

# USD per million tokens, from Groq's model page
# https://console.groq.com/docs/model/openai/gpt-oss-120b (read 2026-09-26):
# input $0.15, cached input $0.075, output $0.60. Cached-input discounts are not
# visible per call here, so every input token is costed at the full rate - an
# upper bound.
PRICING_USD_PER_M = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}

# The fixed experiment / regression query set: 8 of the 20 W16 harness cases,
# covering every kind (single hop, multi hop, project notes, arithmetic,
# unanswerable) and including three the W16 single loop failed (read-lines,
# tz-convert, dataclass-json). Eight, not more, because of Groq's free-tier
# budget for gpt-oss-120b: 8K tokens/minute and 200K tokens/day, shared by the
# assistant and the judge, and one W15-loop query measured up to ~15K tokens.
CASE_IDS = (
    "json-indent",
    "int-error",
    "read-lines",
    "tz-convert",
    "dataclass-json",
    "project-serving",
    "arith-units",
    "out-of-corpus-pricing",
)


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = PRICING_USD_PER_M.get(model)
    if not price:
        return 0.0
    return prompt_tokens / 1e6 * price["input"] + completion_tokens / 1e6 * price["output"]


@dataclass
class RunConfig:
    """Everything that defines one tracked run. Logged verbatim as MLflow params."""

    prompt_version: str
    prompt_file: str
    agent_mode: str = "single"
    provider: str = PROVIDER
    model: str = MODEL
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 4
    max_iterations: int = 4
    reasoning_effort: str = "low"
    # Accept a final answer gpt-oss wrapped in a pseudo-tool call (see app/llm.py).
    recover_tool_answers: bool = False
    note: str = ""

    def settings_overrides(self, system_prompt: str) -> dict[str, Any]:
        overrides: dict[str, Any] = {
            "provider": self.provider,
            "top_k": self.top_k,
            "max_tool_iterations": self.max_iterations,
            "system_prompt": system_prompt,
            "reasoning_effort": self.reasoning_effort,
            "recover_tool_answers": self.recover_tool_answers,
            # One model per run: a silent fail-over would mix two models' numbers.
            "fallback_provider": "none",
            # Hosted free tiers rate-limit by tokens per minute; retries honour
            # Retry-After (app.llm) and need room to wait it out.
            "max_retries": 8,
            "request_timeout_seconds": 120.0,
            # Absolute, so a scheduler whose cwd is elsewhere finds the same index.
            "data_dir": str(ROOT / "data"),
            "chroma_dir": str(ROOT / "data" / "chroma"),
            "embedding_cache_dir": str(ROOT / ".cache" / "fastembed"),
        }
        model_field = {"groq": "groq_model", "openai": "openai_model", "ollama": "ollama_model", "vllm": "vllm_model"}
        overrides[model_field.get(self.provider, "groq_model")] = self.model
        return overrides

    def as_params(self) -> dict[str, Any]:
        return asdict(self)

    def default_run_name(self) -> str:
        return (
            f"{self.prompt_version}-{self.agent_mode}-t{self.temperature:g}-k{self.top_k}-i{self.max_iterations}"
        )


def load_versions() -> dict[str, dict[str, Any]]:
    return json.loads((PROMPTS_DIR / "versions.json").read_text(encoding="utf-8"))


def load_config(version: str, **overrides: Any) -> tuple[RunConfig, str]:
    """The registered config for a prompt version, with CLI overrides applied."""
    versions = load_versions()
    if version not in versions:
        raise SystemExit(f"unknown prompt version {version!r}; registered: {sorted(versions)}")
    spec = dict(versions[version])
    spec.update({key: value for key, value in overrides.items() if value is not None})
    config = RunConfig(prompt_version=version, **spec)
    prompt_text = (PROMPTS_DIR / config.prompt_file).read_text(encoding="utf-8").strip()
    return config, prompt_text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")
