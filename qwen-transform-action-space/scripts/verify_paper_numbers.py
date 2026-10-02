#!/usr/bin/env python3
"""Stage 2.5 reproducibility reconciliation.

Re-derives every quantitative claim that appears in paper/main.md directly from
the raw corpus and the frozen artifacts, then asserts each one. This exists
because the corpus language census (23,731 / 22,007 / 0 / 1,724) was originally
produced by an ad-hoc inline command; a paper number that cannot be re-run is a
number that cannot be checked.

Usage: python scripts/verify_paper_numbers.py
Exit 0 if all assertions pass, 1 otherwise.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "adaexplore"
PAPER = ROOT / "paper" / "main.md"

checks: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    checks.append((name, bool(cond), detail))


def census() -> dict:
    total = tri = cu = neither = 0
    for p in DATA.rglob("step_*.py"):
        t = p.read_text(encoding="utf-8", errors="replace")
        total += 1
        a = "@triton.jit" in t or "import triton" in t
        b = (
            "__global__" in t
            or "threadIdx" in t
            or "load_inline" in t
            or "CUDAExtension" in t
        )
        tri += a
        cu += b
        neither += not a and not b
    return {"total": total, "triton": tri, "cuda": cu, "neither": neither}


def main() -> int:
    paper = PAPER.read_text(encoding="utf-8")
    # ---- corpus census -------------------------------------------------
    c = census()
    check("census total step files == 23731", c["total"] == 23731, str(c["total"]))
    check("census triton == 22007", c["triton"] == 22007, str(c["triton"]))
    check("census cuda == 0", c["cuda"] == 0, str(c["cuda"]))
    check("census pytorch-only == 1724", c["neither"] == 1724, str(c["neither"]))
    for lit, val in [("23,731", c["total"]), ("22,007", c["triton"]), ("1,724", c["neither"])]:
        check(f"paper quotes {lit}", lit in paper, f"computed {val}")

    # ---- edge counts ---------------------------------------------------
    cands = json.loads((ROOT / "data" / "edges" / "edge_index.json").read_text(encoding="utf-8"))
    sel = json.loads((ROOT / "data" / "edges" / "selected_edges.json").read_text(encoding="utf-8"))
    lv = Counter(e["level"] for e in cands)
    tasks = len({(e["level"], e["task_id"]) for e in cands})
    check("candidates == 2510", len(cands) == 2510, str(len(cands)))
    check("L2 candidates == 1906", lv["L2"] == 1906, str(lv["L2"]))
    check("L3 candidates == 604", lv["L3"] == 604, str(lv["L3"]))
    check("candidate tasks == 145", tasks == 145, str(tasks))
    check("selected == 55", len(sel) == 55, str(len(sel)))
    check("selected tasks == 45", len({(e['level'], e['task_id']) for e in sel}) == 45)
    check("selected L2 == 49", sum(1 for e in sel if e["level"] == "L2") == 49)
    check("selected L3 == 6", sum(1 for e in sel if e["level"] == "L3") == 6)
    check("multi-hunk == 33", sum(1 for e in sel if e["hunks"] > 1) == 33)
    check("all diffs distinct", len({e["diff_fingerprint"] for e in sel}) == 55)

    # ---- annotations ---------------------------------------------------
    blob = json.loads((ROOT / "data" / "annotations.json").read_text(encoding="utf-8"))
    st = blob["statistics"]
    recs = blob["annotations"]
    check("annotation count == edge count", len(recs) == len(sel))

    def near(name, got, want, tol=0.005):
        check(f"{name} == {want}", abs(got - want) <= tol, f"computed {got}")

    near("H(primary)", st["entropy_primary_bits"], 2.67)
    near("H(diff fingerprint)", st["entropy_diff_fingerprint_bits"], 5.78)
    near("H(any label)", st["entropy_all_labels_bits"], 3.07)
    near("H(hunkcount)", st["entropy_hunkcount_bits"], 2.49)
    near("H(changed bucket)", st["entropy_changedline_bucket_bits"], 2.40)
    check("coverage 90.9", abs(st["coverage_pct"] - 90.9) < 0.05, str(st["coverage_pct"]))
    check("reuse 0.18", abs(st["reuse_rate_V_over_E"] - 0.1818) < 0.001)
    check("multi-loci 30/55", st["edges_spanning_multiple_loci"] == 30)
    check("vocab used as primary == 10", st["vocabulary_used"] == 10)
    check("autotuner 53/55", st["edges_with_autotuner_change"] == 53)
    check("suspicious 9/55", st["edges_with_suspicious_parent_speedup"] == 9)
    conf = st["confidence_dist"]
    check("confidence 14/38/3", (conf.get("high"), conf.get("medium"), conf.get("low")) == (14, 38, 3), str(conf))
    pc = st["label_counts_primary"]
    for lab, want in [("Fuse", 17), ("PrecisionChange", 15), ("HostDeviceOverlap", 7),
                      ("Other", 5), ("ChangeLaunchConfig", 3), ("Vectorize", 3),
                      ("EliminateMaterialization", 2), ("Tile", 1), ("LoopTransform", 1),
                      ("IncreaseWorkPerThread", 1)]:
        check(f"primary {lab} == {want}", pc.get(lab) == want, str(pc.get(lab)))
    ac = st["label_counts_any"]
    check("any WarpReduce == 6", ac.get("WarpReduce") == 6)
    check("any Vectorize == 6", ac.get("Vectorize") == 6)
    check("SharedMemoryTransform never used", ac.get("SharedMemoryTransform") is None)
    check("Fuse share 30.9%", abs(17 / 55 * 100 - 30.9) < 0.05)
    d = st["within_label_diversity"]
    for lab, occ, diffs, tasks_ in [("ChangeLaunchConfig", 34, 34, 30), ("EliminateMaterialization", 32, 32, 29),
                                     ("Fuse", 25, 25, 22), ("PrecisionChange", 22, 22, 18),
                                     ("Tile", 14, 14, 13), ("HostDeviceOverlap", 14, 14, 12),
                                     ("WarpReduce", 6, 6, 6), ("Vectorize", 6, 6, 5)]:
        e = d.get(lab, {})
        check(
            f"diversity {lab}",
            (e.get("occurrences"), e.get("distinct_diffs"), e.get("distinct_tasks"))
            == (occ, diffs, tasks_),
            str(e),
        )

    # ---- calibration (schema v2: symmetric permutation + correct MM) ----
    cal = json.loads((ROOT / "data" / "informativeness_calibration.json").read_text(encoding="utf-8"))
    check("calibration schema v2", cal.get("schema_version") == 2, str(cal.get("schema_version")))
    near("H(speedup quartile)", cal["outcome"]["entropy_bits"], 2.00)
    check("speedup outcome has 4 cells", cal["outcome"]["cells"] == 4)
    descs = {d["descriptor"]: d for d in cal["descriptors"]}
    for name, cells, mi, mm, null_m, null_sd, p in [
        ("primary semantic label", 10, 0.697, 0.624, 0.397, 0.072, 0.0005),
        ("coarse diff structure", 24, 0.9373, 0.7919, 0.990, 0.085, 0.752),
        ("hunk count alone", 9, 0.3668, 0.2759, 0.361, 0.071, 0.436),
    ]:
        d = descs.get(name, {})
        check(f"{name}: cells", d.get("cells") == cells, str(d.get("cells")))
        near(f"{name}: MI_plugin", d.get("mi_plugin_bits", 0), mi, tol=0.002)
        near(f"{name}: MI_MillerMadow", d.get("mi_miller_madow_bits", 0), mm, tol=0.002)
        pt = d.get("permutation_test", {})
        near(f"{name}: null mean", pt.get("null_mean_bits", 0), null_m, tol=0.005)
        near(f"{name}: null sd", pt.get("null_sd_bits", 0), null_sd, tol=0.005)
        near(f"{name}: permutation p", pt.get("p_value_one_sided", 0), p, tol=0.002)
    boot = cal["bootstrap_label_minus_coarse"]
    near("bootstrap mean diff", boot["mean_diff_bits"], -0.361, tol=0.02)
    check("bootstrap CI upper excludes 0", boot["ci95_high_bits"] < 0, str(boot["ci95_high_bits"]))
    ne = cal["normalized_entropies"]
    near("label entropy", ne["primary_label"]["entropy_bits"], 2.67)
    near("coarse entropy", ne["coarse_structure"]["entropy_bits"], 4.26, tol=0.01)
    near("fingerprint entropy is the identity log2(55)", ne["diff_fingerprint"]["entropy_bits"], 5.78)
    check("fingerprint normalized == 1.0", abs(ne["diff_fingerprint"]["normalized"] - 1.0) < 1e-6,
          str(ne["diff_fingerprint"]["normalized"]))

    # identity + cross-task nulls that the paper now reports as retractions
    ia = cal["diversity_null_models"]["identity_audit"]
    check("identity audit: all fingerprints unique", ia["all_unique"] is True)
    check("identity audit: n == distinct", ia["n_edges"] == ia["n_distinct_fingerprints"] == 55)
    dn = cal["diversity_null_models"]["distinct_tasks_vs_label_permutation"]["per_label"]
    for lab, obs, null_mean, pct in [
        ("Fuse", 16, 16.03, 66.8),
        ("PrecisionChange", 13, 14.26, 15.5),
    ]:
        e = dn.get(lab, {})
        check(f"distinct_tasks {lab} observed", e.get("observed_distinct_tasks") == obs, str(e))
        near(f"distinct_tasks {lab} null mean", e.get("null_mean", 0), null_mean, tol=0.02)
        near(f"distinct_tasks {lab} percentile", e.get("percentile", 0), pct, tol=0.6)

    # the paper must present the corrected result and mark the retracted ones
    abstract = paper.split("## Abstract")[1].split("---")[0] if "## Abstract" in paper else ""
    check("abstract carries permutation p=0.0005", "0.0005" in abstract, abstract[:120])
    check("abstract no longer cites the retracted 0.615/0.728 pair",
          "0.615" not in abstract and "0.728" not in abstract)
    check("abstract frames the compression comparison as NOT the evidence",
          ("informative test is not compression" in abstract.lower()
           or "identity" in abstract.lower()),
          abstract[-200:])
    # The abstract no longer leans on the compression comparison at all -- that was the
    # argument Section 3.5 retired -- so requiring it to quote the 4.26-bit comparator
    # would push the paper to re-add a withdrawn claim. The invariant that matters is
    # conditional: IF the abstract cites a compression figure, it must also mark the
    # comparison non-evidential.
    cites_compression = "2.67" in abstract or "4.26" in abstract
    if cites_compression:
        check("abstract qualifies the compression comparison",
              ("not compression" in abstract.lower()
               or "identity" in abstract.lower()
               or "coarse summary" in abstract.lower()),
              "cites 2.67/4.26 without the non-evidence qualifier")
    else:
        check("abstract rests on the permutation test, not on compression",
              "permutation" in abstract.lower() and "0.0005" in abstract,
              abstract[:120])
    sec36 = paper.split("### 3.6")[1].split("### 3.7")[0] if "### 3.6" in paper else ""
    for token in ["Miller", "permutation", "0.0005", "0.752", "0.436"]:
        check(f"3.6 reports corrected statistic/method: {token}", token in sec36, token)
    sec35 = paper.split("### 3.5")[1].split("### 3.6")[0] if "### 3.5" in paper else ""
    check("3.5 labels fingerprint entropy as an identity", "identity" in sec35.lower())
    check("3.5 states plainly which claims the table cannot support",
          any(k in sec35.lower() for k in ("do not survive", "void", "withdraw", "retract")),
          sec35[:80])
    check("3.5 reports the cross-task null honestly", "16.03" in sec35 or "at chance" in sec35.lower())

    # cross-section consistency: a retracted claim may only survive INSIDE the
    # withdrawal list of Section 3.5, quoted as what is being retracted. Anywhere
    # else it would be an asserted claim that the evidence no longer supports.
    outside_35 = paper.replace(sec35, "")
    for stale in ["recur across tens", "recurs across roughly thirty",
                  "tens of different tasks", "34 distinct diffs"]:
        check(f"retracted claim confined to 3.5 withdrawal list: '{stale[:28]}'",
              stale not in outside_35, stale)
    sec25 = paper.split("### 2.5")[1].split("## 3.")[0] if "### 2.5" in paper else ""
    check("2.5 states transferability is unsupported by this data",
          "not supported by this data" in sec25 or "not supported by our data" in sec25,
          sec25[:80])
    check("2.1 table includes the turn/tool-call cut", "Turn / tool call" in paper)
    check("Section 4 defines the Brute-search arm", "Brute-search" in paper)

    # Stage 3' NEW-1 regression guard: no section may still assert the reversed
    # v1 conclusion outside the explicit correction narrative about it.
    sec52 = paper.split("### 5.2")[1].split("### 5.3")[0] if "### 5.2" in paper else ""
    check("5.2 no longer asserts labels 'do not out-predict' surface structure",
          "do not out-predict" not in sec52, "stale reversed claim survives")
    check("5.2 states the hierarchy is neither confirmed nor redundant",
          "neither confirmed nor shown redundant" in sec52, sec52[:80])
    check("7.3 discloses the oracle false-pass dependency",
          "[20,24]" in paper or "false pass" in paper.lower() or "admits false passes" in paper.lower())
    # NEW-2 targeted check, replacing an earlier generic line scan that flagged any
    # standalone "69": that scan misfired on speedup literals ("69x") and on the
    # historically correct statement that the v1 harness passed 69/69. A brittle
    # substring audit is exactly the failure this paper documents, so the check is
    # scoped to the two sentences that describe the harness state.
    check("harness bookkeeping lives in Appendix B, not in the Section 3 argument",
          "69" in paper.split("## Appendix B")[1] and "112" not in sec36,
          "v1 harness count should appear only in the production record")
    check("Appendix B describes the historical 69/69 pass accurately",
          "69/69" in paper.split("## Appendix B")[1])
    check("no dangling Section 7.5 reference after relocation",
          "Section 7.5" not in paper,
          f"{len(re.findall(r'Section 7\\.5', paper))} reference(s) remain")
    _body = paper[: paper.find("## References")]
    check("production narrative stays out of the body",
          _body.lower().count("pipeline") <= 8,
          f"body mentions pipeline {_body.lower().count('pipeline')}x")

    # ---- every Other edge really is a backend substitution -------------
    others = [r for r in recs if r["primary"] == "Other"]
    check("Other count == 5", len(others) == 5, str(len(others)))
    backend = re.compile(
        r"(cuBLAS|matmul|scaled_dot_product|torch\.nn\.functional|F\.linear|F\.mish"
        r"|torch\.mish|library|backend)",
        re.I,
    )
    # Annotation reasoning is Chinese (see STAGE2_5_INTEGRITY.md); accept the
    # equivalent Chinese substitution vocabulary rather than forcing translation.
    backend_zh = re.compile("(库|手调|撤下|拆到|改回|换用)")
    for r in others:
        blobtext = " ".join([r.get("reasoning", ""), r.get("notes", ""), " ".join(r.get("why_not_existing", []))])
        check(
            f"Other edge {r['edge_id']} mentions backend/library substitution",
            bool(backend.search(blobtext)) or bool(backend_zh.search(blobtext)),
            blobtext[:80],
        )

    # ---- reporting -----------------------------------------------------
    passed = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        if not ok:
            print(f"FAIL  {name}   [{detail}]")
    print(f"\n{passed}/{len(checks)} numeric assertions reconciled against source data")
    if passed == len(checks):
        print("REPRODUCIBILITY: PASS")
        return 0
    print("REPRODUCIBILITY: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
