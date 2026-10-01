import {
  Callout,
  canvasImage,
  Code,
  H1,
  MetricsGrid,
  ReportSection,
  ReportShell,
  Stack,
  Table,
  Tag,
  Text,
  Timeline,
} from "qoder/canvas";

const fig5 = canvasImage("./fig5_permutation_calibration.png");
const fig1 = canvasImage("./fig1_entropy_comparison.png");

const headlineMetrics = [
  { label: "Pipeline stages", value: "10", description: "Stage 1 → Stage 5, auto-advanced" },
  { label: "Commits on branch", value: "20", description: "each one a rollback point" },
  { label: "References verified", value: "36/36", tone: "success" as const, description: "arXiv 33 · DOI 2 · URL 1" },
  { label: "Annotated edges", value: "55", description: "from 2,510 candidates / 145 trees" },
];

const stageEvents = [
  { id: "s1", timestamp: "Stage 1", title: "RESEARCH", state: "completed" as const, tone: "success" as const, description: "RQ brief + per-source ledger. Two spec-named titles unlocatable, barred from the bibliography. All seven keyword groups audited later." },
  { id: "s15", timestamp: "Stage 1.5", title: "EXPERIMENT", state: "completed" as const, tone: "success" as const, description: "Vocabulary frozen before any diff was read; 55 MCTS parent→child edges annotated by five blinded passes." },
  { id: "s2", timestamp: "Stage 2", title: "WRITE", state: "completed" as const, tone: "success" as const, description: "English draft, 8,825 words, 10 sections." },
  { id: "s25", timestamp: "Stage 2.5", title: "INTEGRITY", state: "completed" as const, tone: "success" as const, description: "28/28 citations re-resolved; 3 unverifiable assertions from the spec narrative removed." },
  { id: "s3", timestamp: "Stage 3", title: "REVIEW", state: "completed" as const, tone: "warning" as const, description: "Five blinded reviewers; decision Major Revision (48–57). Caught a statistical error four self-checks had certified." },
  { id: "s4", timestamp: "Stage 4", title: "REVISE", state: "completed" as const, tone: "success" as const, description: "R-1…R-8; headline conclusion reversed; four verified prior works added." },
  { id: "s3p", timestamp: "Stage 3′/4′", title: "RE-REVIEW + RE-REVISE", state: "completed" as const, tone: "success" as const, description: "R-1…R-6 MET; found one stale contradictory sentence and an inaccurate self-report." },
  { id: "s45", timestamp: "Stage 4.5", title: "FINAL INTEGRITY", state: "completed" as const, tone: "success" as const, description: "From-scratch re-verification: 36/36 references, 117/117 assertions, 3 field-level citation errors fixed." },
  { id: "s5", timestamp: "Stage 5", title: "FINALIZE", state: "completed" as const, tone: "success" as const, description: "LaTeX project + mechanically generated .bib + Chinese parallel translation." },
];

const reversalRows = [
  ["v1 claim", "Semantic labels yield less information about realised speedup than coarse diff structure (0.615 vs 0.728 bits)", "Headline of the draft"],
  ["Root cause", "Marginal-entropy Miller–Madow term (k_X−1)/2n applied to a mutual information; (k_Y−1) never multiplied in, and only one descriptor was permutation-calibrated", "Found independently by 2 reviewers"],
  ["v2 result", "Semantic label beats its own null (p = 0.0005); coarse structure (p = 0.752) and hunk count (p = 0.436) do not", "Direction reversed"],
  ["Also withdrawn", "H(diff fingerprint) = 5.78 bits as evidence — identically log₂(55); cross-task recurrence — at chance under label permutation", "Both were load-bearing in v1"],
];

const claimRows = [
  ["A frozen 12-category vocabulary covers real successes without inflation", "Supported", "10 categories, 90.9% coverage, Other actively invited"],
  ["Semantic labels carry information beyond their own granularity", "Supported", "permutation p = 0.0005 vs 0.752 / 0.436"],
  ["Low description length", "Qualified", "2.67 bits vs 4.26 for a faithful coarse description; 5.78 is an identity"],
  ["Same action recurs across tasks (transferability)", "Not supported", "at chance under label permutation; sampling caps edges per task"],
  ["One action spans several code sites", "Unidentifiable", "0 of 55 edges carry a single label"],
  ["Search efficiency / learning efficiency", "Untested", "no intervention experiment exists in this paper"],
];

const gateRows = [
  ["verify_references.py", "36/36 references re-resolved from scratch; title and author surname compared field by field", "PASS"],
  ["verify_paper_numbers.py", "117/117 numeric and prose assertions re-derived from the raw corpus", "PASS"],
  ["verify_zh_alignment.py", "286/286 quantitative tokens match EN↔ZH; symbol parity 8/8; 35/35 identifiers in order", "PASS"],
  ["integrity_scan.py", "Citation closure 36/36 both ways; CJK leakage 0; 11/11 sections", "PASS"],
  ["audit_spec_coverage.py", "Abstract 245 words (spec 200–250); stated marker equals measured; all deliverables present", "PASS"],
];

const caughtRows = [
  ["Method", "Miller–Madow misuse reversing the headline conclusion", "Stage 3 — 2 reviewers, independently"],
  ["Evidence", "2 spec-named works unlocatable; 3 assertions sourced from narrative rather than the paper", "Stage 1 / 2.5"],
  ["Framing", "Straw-man risk on CUDA Agent's action space; a failed RL transfer cited as success", "Stage 3"],
  ["Novelty", "K-Search, CuAsmRL, CodeAct and ReCode already occupy parts of the proposed interface", "Stage 3 + keyword re-audit"],
  ["Process", "An automated reviewer fabricated a quotation; it was excluded and our own claim weakened", "Stage 4.5 cross-check"],
  ["Self-report", "Revision notes marked R-8 done while the manuscript still lacked two disclosures", "Stage 3′"],
];

const deliverableRows = [
  ["paper/main.md", "English manuscript, 10 sections + appendix, 36 references"],
  ["paper/latex/", "main.tex, references.bib (36), figures/, BUILD.md"],
  ["paper/zh/full.md", "Chinese parallel translation, 21.5k characters"],
  ["data/", "annotations.json, exemplar_Fuse.md, informativeness_calibration.json, 55 diffs"],
  ["scripts/", "11 reproducible scripts: selection, annotation stats, calibration, gates, builds"],
  ["stage*_*.md", "Per-stage reports: research, experiment, integrity, review, revision, re-review, final integrity"],
];

export default function TransformActionSpaceRunReport() {
  return (
    <ReportShell width="wide" ariaLabel="Transform-Action-Space paper pipeline completion report">
      <Stack gap="sectionCompact">
        <header>
          <Stack gap="component">
            <H1>Program Transformation Action Space — Run Report</H1>
            <Text tone="secondary">
              Spec <Code>Transform-Action-Space_Paper_task-127.md</Code> · branch{" "}
              <Code>qwen/transform-action-space</Code> · driven by the academic-pipeline,
              academic-paper, academic-paper-reviewer and nature-academic-search skills · 2026-10-02
            </Text>
            <MetricsGrid variant="header" columns={4} items={headlineMetrics} />
          </Stack>
        </header>

        <ReportSection
          title="Headline: the paper's central result reversed itself"
          description="The first draft's prised act of self-falsification turned out to be an arithmetic bug. The manuscript reports this instead of removing it."
          divided
        >
          <Table
            headers={["Layer", "Detail", "Status"]}
            rows={reversalRows}
            rowTone={["neutral", "warning", "success", "danger"]}
          />
          <Stack gap="component">
            <img
              src={fig5}
              alt="Each descriptor's mutual information with the realised-speedup quartile against its own permutation null; only the semantic label exceeds its null."
              style={{ maxWidth: "100%", height: "auto" }}
            />
            <Text size="small" tone="secondary">
              Figure 5 — the corrected comparison. Raw mutual information still favours the 24-cell coarse
              descriptor, but its observed value sits below its own null mean, so the advantage is granularity
              over-fitting. This is why calibration is symmetric in v2.
            </Text>
          </Stack>
        </ReportSection>

        <ReportSection title="Stage sequence" divided>
          <Timeline events={stageEvents} density="compact" />
        </ReportSection>

        <ReportSection
          title="What the data actually supports"
          description="55 annotated successful edges over 45 task trees from public AdaExplore trajectories. The corpus is Triton-only: 22,007 Triton files, 0 CUDA C++ files."
          divided
        >
          <Stack gap="component">
            <Table
              headers={["Claim", "Verdict", "Evidence"]}
              rows={claimRows}
              rowTone={["success", "success", "info", "danger", "warning", "neutral"]}
            />
            <img
              src={fig1}
              alt="Description length per edge across label and diff-structure descriptors; the diff-fingerprint bar is hatched and marked as an identity."
              style={{ maxWidth: "100%", height: "auto" }}
            />
            <Text size="small" tone="secondary">
              Figure 1 — the hatched bar is identically log₂(n) and is labelled non-evidential in the manuscript.
            </Text>
          </Stack>
        </ReportSection>

        <ReportSection
          title="Two findings we did not plan for"
          description="Both changed how the action-space thesis is written, and both travel beyond CUDA."
          divided
        >
          <Stack gap="component">
            <Text>
              All five <Tag tone="warning">Other</Tag> edges turned out to be the same uncategorised decision — stop
              implementing the operation and call a library — including the two largest realised speedups in the
              sample. The pre-registered 25% numeric red line did not fire, because the refusals were homogeneous:
              coverage percentages are the wrong instrument for detecting a missing category.
            </Text>
            <Text>
              Autotuning configuration changed alongside program structure on <Tag tone="warning">53 of 55</Tag>{" "}
              successful edges, so in realistic kernel-optimization corpora hyperparameter search and structural
              transformation are not separable without being designed apart.
            </Text>
          </Stack>
        </ReportSection>

        <ReportSection title="Verification gates" description="All five re-runnable; the run finishes green." divided>
          <Table headers={["Gate", "What it checks", "Result"]} rows={gateRows} rowTone="success" />
        </ReportSection>

        <ReportSection
          title="What the pipeline caught, and where"
          description="Recorded because the instructive failure here is that self-checking passed while review did not."
          divided
        >
          <Table headers={["Class", "Problem", "Caught by"]} rows={caughtRows} />
        </ReportSection>

        <ReportSection title="Deliverables" divided>
          <Stack gap="component">
            <Table headers={["Path", "Contents"]} rows={deliverableRows} />
            <Callout tone="info" title="Commit history">
              Twenty commits on <Code>qwen/transform-action-space</Code>, grouped by artifact class — scaffold,
              stage-1 research, frozen vocabulary, edge selection, annotations, statistics, Stage 2.5, review,
              analysis fix, harness, revision, re-review, Stage 4′, final integrity, Stage 5, keyword audit, and
              required-reading reconciliation — so any stage can be rolled back independently.
            </Callout>
          </Stack>
        </ReportSection>

        <ReportSection title="Open items, left open deliberately" divided>
          <Stack gap="component">
            <Callout tone="warning" title="Length">
              Body text is roughly 13.7k words against the spec's 8–12 page target. Further cuts would move the
              argumentative centre — compress Section 5, or move the retraction narrative to an appendix — which is
              an authorial decision rather than a mechanical one, so it is logged as an acknowledged limitation
              instead of patched cosmetically.
            </Callout>
            <Callout tone="danger" title="Circularity not broken">
              Annotation is machine-generated by the same model family that designed the vocabulary. The manuscript
              states four falsifiable acceptance criteria for removing this and records that none were executed. This
              is the paper's largest real weakness.
            </Callout>
            <Callout tone="warning" title="No PDF">
              No TeX engine exists on this machine (latexmk, tectonic, xelatex, pdflatex, lualatex all absent), so
              only the LaTeX source is delivered. An HTML-to-PDF route was refused rather than substituted, and the
              reason is written into paper/latex/BUILD.md.
            </Callout>
            <Text tone="secondary" size="small">
              Stage 6 (paper creation process record) remains available on request.
            </Text>
          </Stack>
        </ReportSection>
      </Stack>
    </ReportShell>
  );
}
