"""Index archived append stages and their construction costs, never live files.

The cost scope is the development episodes that supplied these additions and
their rule-writing calls, not every exploratory call made by the whole project.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from runtime import digest, summarize_usage


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def bounds(usage, prices):
    return {period: sum(usage[token] * rates[rate] for token, rate in [
        ("uncached_input_tokens", "uncached_input"),
        ("cached_input_tokens", "cached_input"), ("output_tokens", "output")]) / prices["per_tokens"]
        for period, rates in [("off_peak_usd", prices["off_peak"]), ("peak_usd", prices["peak"])]}


def main():
    price_path = ROOT / "sources/deepseek-pricing-2026-10-19.json"
    prices = read(price_path)
    index = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "scope": __doc__.strip(), "price_source": prices["source"],
             "price_sha256": digest(price_path),
             "cost_interpretation": "Published all-off-peak/all-peak bounds, not a bill. Reasoning is included in output and is not charged again.",
             "stages": []}
    previous = (ROOT / "sources/CUDA-Agent.SKILL.md").read_text(encoding="utf-8")
    total = dict.fromkeys(["input_tokens", "cached_input_tokens", "uncached_input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"], 0)
    for number, directory in enumerate(sorted((ROOT / "evidence").glob("growth-[0-9][0-9]")), 1):
        assert directory.name == f"growth-{number:02}"
        provenance = read(directory / "provenance.json")
        assert provenance["status"] == "complete"
        for name, key in [("addition.md", "addition_sha256"), ("cumulative.md", "cumulative_sha256")]:
            assert digest(directory / name) == provenance[key]
        current = (directory / "cumulative.md").read_text(encoding="utf-8")
        assert current.startswith(previous.rstrip() + "\n\n")
        previous = current
        trial = ROOT / Path(provenance["trial"].replace("\\", "/"))
        result = read(trial / "result.json")
        config = read(trial / "config.json")
        assert config["purpose"] == "development"
        responses = sorted(directory.glob("attempt-*-response.json")) or [directory / "response.json"]
        maintenance = summarize_usage([read(path)["usage"] for path in responses])
        actor = result["usage"]
        assert actor and not result.get("usage_error")
        actor_responses = sorted(trial.glob("response-*.json"))
        actor_rows = [read(path)["usage"] for path in actor_responses]
        assert len(actor_rows) == result["requests"]
        assert summarize_usage(actor_rows) == actor, "Trial total disagrees with actual API responses"
        # Retain an archived usage copy; later reports need not depend on runs/.
        stage = {"stage": number, "directory": directory.relative_to(ROOT).as_posix(),
                 "provenance_sha256": digest(directory / "provenance.json"),
                 "lines": len(current.splitlines()), "source_bullets_added": sum(line.startswith("- ") for line in (directory / "addition.md").read_text(encoding="utf-8").splitlines()),
                 "trial": provenance["trial"], "task": config["task"],
                 "trial_state": result["state"], "actor_requests": result["requests"],
                 "actor_usage": actor, "maintenance_attempts": len(responses), "maintenance_usage": maintenance,
                 "final_evaluation": {name: bool(result["final_evaluation"].get(name, {}).get("ok")) for name in ["compile", "verify", "benchmark"]},
                 "construction_cost_bounds": {"actor": bounds(actor, prices), "maintenance": bounds(maintenance, prices)}}
        for usage in [actor, maintenance]:
            for key in total:
                assert usage[key] is not None, "Do not silently invent a missing reasoning breakdown"
                total[key] += usage[key]
        index["stages"].append(stage)
    index["construction_usage"] = total
    index["construction_cost_bounds"] = bounds(total, prices)
    out = ROOT / "materials/growth-index.json"
    out.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stages": len(index["stages"]), "usage": total, "cost_bounds": index["construction_cost_bounds"]}))


if __name__ == "__main__":
    main()
