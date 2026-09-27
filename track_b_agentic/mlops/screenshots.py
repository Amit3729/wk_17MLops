"""Capture MLflow UI screenshots for the README (results/screenshots/*.png).

Needs a running UI (`make mlflow-ui`) and Playwright, which is deliberately not a
project dependency:

    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 &
    uv run --no-project --with playwright --with mlflow-skinny python mlops/screenshots.py

It drives the installed Google Chrome (channel="chrome"), so no browser download
is needed. MLflow 3.16 opens an assistant side panel and dark mode by default;
the init script turns both off before the app loads.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "screenshots"
UI = os.getenv("MLFLOW_UI", "http://127.0.0.1:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "track-b-agentic-prompts")

INIT = """
localStorage.setItem('_mlflow_dark_mode_toggle_enabled', 'false');
localStorage.setItem('mlflow.assistant.panelOpen_v1', 'false');
localStorage.setItem('mlflow.assistant.fab.autoOpenedOnFirstLoad_v1', 'true');
"""


def api(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{UI}/api/2.0/mlflow/{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def main() -> None:
    with urllib.request.urlopen(f"{UI}/api/2.0/mlflow/experiments/get-by-name?experiment_name={EXPERIMENT}") as r:
        experiment = json.load(r)["experiment"]
    exp_id = experiment["experiment_id"]
    runs = api("runs/search", {"experiment_ids": [exp_id], "max_results": 50,
                               "order_by": ["attributes.start_time ASC"]}).get("runs", [])
    runs = [run for run in runs if run["info"]["status"] == "FINISHED"]
    names = {run["info"]["run_id"]: run["info"].get("run_name", "") for run in runs}
    ids = [run["info"]["run_id"] for run in runs]
    wanted = sys.argv[1:]  # optional run names to show on the compare page
    if wanted:
        ids = [run_id for run_id in ids if names[run_id] in wanted]
    OUT.mkdir(parents=True, exist_ok=True)

    shots = [
        ("00_metric_charts.png",
         f"{UI}/#/experiments/{exp_id}/runs?workflowType=machine_learning&compareRunsMode=CHART", 2100, 2400),
        ("01_runs_table.png", f"{UI}/#/experiments/{exp_id}/runs", 1800, 1000),
        ("02_compare_runs.png",
         f"{UI}/#/compare-runs?runs={json.dumps(ids)}&experiments={json.dumps([exp_id])}", 1800, 3400),
        ("03_traces.png", f"{UI}/#/experiments/{exp_id}/traces", 1800, 1100),
    ]
    best = os.getenv("SHOT_RUN")  # run name whose detail + trace to capture
    best_id = next((run_id for run_id, name in names.items() if name == best), ids[-1] if ids else None)
    if best_id:
        shots.append(("04_run_detail.png", f"{UI}/#/experiments/{exp_id}/runs/{best_id}", 1800, 1600))
        shots.append(("05_run_artifacts.png", f"{UI}/#/experiments/{exp_id}/runs/{best_id}/artifacts", 1800, 1100))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        for name, url, width, height in shots:
            context = browser.new_context(viewport={"width": width, "height": height}, color_scheme="light")
            context.add_init_script(INIT)
            page = context.new_page()
            page.goto(url, wait_until="load")
            page.wait_for_timeout(4000)
            page.screenshot(path=str(OUT / name))
            print(f"wrote {OUT / name}")
            context.close()
        # One trace opened: the span tree with tool calls under the AGENT span.
        context = browser.new_context(viewport={"width": 1800, "height": 1200}, color_scheme="light")
        context.add_init_script(INIT)
        page = context.new_page()
        page.goto(f"{UI}/#/experiments/{exp_id}/traces", wait_until="load")
        page.wait_for_timeout(4000)
        got_it = page.get_by_role("button", name="Got it")  # MLflow's "Detect Issues" tip covers the list
        if got_it.count():
            got_it.first.click()
            page.wait_for_timeout(1000)
        target = os.getenv("SHOT_TRACE", "Compute 16535")
        link = page.get_by_text(target, exact=False).first
        if link.count():
            link.click(force=True)
            page.wait_for_timeout(4000)
            page.screenshot(path=str(OUT / "06_trace_detail.png"))
            print(f"wrote {OUT / '06_trace_detail.png'}")
        context.close()
        # The Evidently regression report, opened inside the run's artifacts.
        if best_id:
            context = browser.new_context(viewport={"width": 2000, "height": 1300}, color_scheme="light")
            context.add_init_script(INIT)
            page = context.new_page()
            page.goto(f"{UI}/#/experiments/{exp_id}/runs/{best_id}/artifacts", wait_until="load")
            page.wait_for_timeout(4000)
            page.get_by_text("regression", exact=True).first.click()
            page.wait_for_timeout(1500)
            page.get_by_text(".html", exact=False).first.click()
            page.wait_for_timeout(9000)
            page.screenshot(path=str(OUT / "07_evidently_in_mlflow.png"))
            print(f"wrote {OUT / '07_evidently_in_mlflow.png'}")
        browser.close()


if __name__ == "__main__":
    main()
