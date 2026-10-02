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
  { label: "流水线阶段", value: "10", description: "Stage 1 → Stage 5，全程自动推进" },
  { label: "分支提交数", value: "22", description: "每一个都是可回退存档点" },
  { label: "引用已核验", value: "36/36", tone: "success" as const, description: "arXiv 33 · DOI 2 · URL 1" },
  { label: "标注边数", value: "55", description: "取自 2,510 条候选 / 145 棵树" },
];

const stageEvents = [
  { id: "s1", timestamp: "Stage 1", title: "文献检索与定位", state: "completed" as const, tone: "success" as const, description: "RQ 简报 + 逐源核查账。spec 点名的两份文献检索不到，禁止引用。七组关键词事后逐条补做并留档。" },
  { id: "s15", timestamp: "Stage 1.5", title: "自动标注 + 熵分析", state: "completed" as const, tone: "success" as const, description: "词汇表在读取任何一条 diff 之前冻结；55 条 MCTS 父子成功边由五个互相不可代理的批次标注。" },
  { id: "s2", timestamp: "Stage 2", title: "论文撰写", state: "completed" as const, tone: "success" as const, description: "英文初稿 8,825 词、10 节。" },
  { id: "s25", timestamp: "Stage 2.5", title: "完整性核验", state: "completed" as const, tone: "success" as const, description: "28/28 引用回源；拦下 3 处来自 spec 叙事而非一手论文的断言。" },
  { id: "s3", timestamp: "Stage 3", title: "五视角模拟评审", state: "completed" as const, tone: "warning" as const, description: "决定为 Major Revision（48–57 分）。抓出一个前四道自检全部放过的统计学错误。" },
  { id: "s4", timestamp: "Stage 4", title: "修订", state: "completed" as const, tone: "success" as const, description: "逐项处置 R-1…R-8；头条结论反号；补入四条已核实先验工作。" },
  { id: "s3p", timestamp: "Stage 3′/4′", title: "验证复审 + 定向小修", state: "completed" as const, tone: "success" as const, description: "R-1…R-6 判定 MET；发现一句未同步的自相矛盾表述，以及一处不实的自我评估。" },
  { id: "s45", timestamp: "Stage 4.5", title: "终验", state: "completed" as const, tone: "success" as const, description: "从零复核：引用 36/36、断言 117/117；修掉 3 处引用字段级错误。" },
  { id: "s5", timestamp: "Stage 5", title: "终稿交付", state: "completed" as const, tone: "success" as const, description: "LaTeX 工程 + 机械生成的 .bib + 中文对照稿。" },
];

const reversalRows = [
  ["v1 的说法", "语义标签对实测加速的信息量不优于 diff 粗结构（0.615 对 0.728 bits）", "当时的头条结论"],
  ["根因", "把边际熵的 Miller–Madow 偏差项 (k_X−1)/2n 用在互信息上，漏乘 (k_Y−1)；且只对语义标签做了置换校准，没对粗结构做", "由两位审稿人各自独立复算发现"],
  ["v2 结果", "语义标签显著优于自身零模型（p = 0.0005）；粗结构 p = 0.752、仅 hunk 数 p = 0.436 都不显著", "方向反号"],
  ["同时撤回", "把 H(diff 指纹) = 5.78 bits 当证据——它恒等于 log₂(55)；把跨任务复现当证据——在标签置换零模型下就是随机水平", "这两条在 v1 都是承重墙"],
];

const claimRows = [
  ["12 类冻结词汇表能覆盖真实成功边而不被撑大", "成立", "10 类、覆盖 90.9%，且 Other 被主动招引"],
  ["语义标签携带超出自身粒度的信息", "成立", "置换检验 p = 0.0005，对照 0.752 / 0.436"],
  ["描述长度更短", "有保留成立", "2.67 bits 对 4.26 bits（同一批 diff 的忠实粗描述）；5.78 是恒等式"],
  ["同一动作跨任务复现（可迁移性）", "不成立", "标签置换零模型下处于机会水平；每任务限 3 条边已耗尽该统计量区分力"],
  ["一个动作横跨多个代码位置", "不可辨识", "55 条边里没有一条是单标签"],
  ["搜索效率 / 学习效率", "未检验", "本文不含任何干预实验"],
];

const gateRows = [
  ["verify_references.py", "36/36 引用从零重新解析，标题与作者姓氏逐字段比对", "通过"],
  ["verify_paper_numbers.py", "117/117 数字与文本断言，全部从原始语料重导", "通过"],
  ["verify_zh_alignment.py", "中英定量 token 286/286 一致；符号对齐 8/8；引用标识符 35/35 同序", "通过"],
  ["integrity_scan.py", "引用双向闭环 36/36；英文稿中文字符 0 处；章节 11/11", "通过"],
  ["audit_spec_coverage.py", "摘要 245 词（spec 要求 200–250）；标注词数与实测一致；交付物齐备", "通过"],
];

const caughtRows = [
  ["方法层", "Miller–Madow 误用，导致头条结论反号", "Stage 3 —— 两位审稿人独立复算"],
  ["证据层", "spec 点名的 2 份文献不存在；3 处断言来自转述而非一手论文", "Stage 1 / 2.5"],
  ["立论层", "对 CUDA Agent 动作空间有稻草人风险；把一个迁移失败案例当成迁移成功", "Stage 3"],
  ["新颖性", "K-Search、CuAsmRL、CodeAct、ReCode 已占据本文所提接口的部分位置", "Stage 3 + 关键词补做"],
  ["流程层", "一个审稿子代理伪造了引文；未采纳并反过来削弱了自己的一句话", "Stage 4.5 交叉核对"],
  ["自评层", "修订记录把 R-8 标为已完成，而稿件实际还缺两处披露", "Stage 3′"],
];

const deliverableRows = [
  ["paper/main.md", "英文正文，10 节 + 附录，36 条参考文献"],
  ["paper/latex/", "main.tex、references.bib（36 条）、figures/、BUILD.md"],
  ["paper/zh/full.md", "中文对照稿，约 2.15 万字符"],
  ["data/", "annotations.json、exemplar_Fuse.md、informativeness_calibration.json、55 个 diff"],
  ["scripts/", "12 个可重跑脚本：筛边、统计、校准、五道验证门、构建"],
  ["stage*_*.md", "各阶段留档：检索、实验、完整性、评审、修订、复审、终验"],
];

export default function TransformActionSpaceRunReport() {
  return (
    <ReportShell width="wide" ariaLabel="程序变换动作空间论文流水线完成报告">
      <Stack gap="sectionCompact">
        <header>
          <Stack gap="component">
            <H1>程序变换动作空间 —— 本轮运行报告</H1>
            <Text tone="secondary">
              依据 spec <Code>Transform-Action-Space_Paper_task-127.md</Code> · 分支{" "}
              <Code>qwen/transform-action-space</Code> · 由 academic-pipeline、academic-paper、
              academic-paper-reviewer、nature-academic-search 四个 skill 编排 · 2026-10-02
            </Text>
            <MetricsGrid variant="header" columns={4} items={headlineMetrics} />
          </Stack>
        </header>

        <ReportSection
          title="头条：论文的核心结果把自己推翻了一次"
          description="初稿最珍视的那次「自我证伪」其实是一个算术 bug 的产物。这一点写进了稿件，没有被删掉。"
          divided
        >
          <Table
            headers={["层面", "内容", "状态"]}
            rows={reversalRows}
            rowTone={["neutral", "warning", "success", "danger"]}
          />
          <Stack gap="component">
            <img
              src={fig5}
              alt="每个描述子与实测加速四分位之间的互信息，以及各自的置换零模型；只有语义标签超出了自己的零模型。"
              style={{ maxWidth: "100%", height: "auto" }}
            />
            <Text size="small" tone="secondary">
              图 5 —— 修正后的对照。未校准的原始互信息仍然偏向 24 格的粗结构描述子，但它的观测值低于自身零均值，
              那份「优势」来自格数过拟合；这正是 v2 把校准做成对称的原因。图内文字为英文，因为图随英文正式稿生成。
            </Text>
          </Stack>
        </ReportSection>

        <ReportSection title="阶段顺序" divided>
          <Timeline events={stageEvents} density="compact" />
        </ReportSection>

        <ReportSection
          title="数据到底支持到哪一步"
          description="公开 AdaExplore 轨迹中 45 个任务树、55 条标注成功边。语料只有 Triton：22,007 个 Triton 文件、0 个 CUDA C++ 文件。"
          divided
        >
          <Stack gap="component">
            <Table
              headers={["论断", "判定", "依据"]}
              rows={claimRows}
              rowTone={["success", "success", "info", "danger", "warning", "neutral"]}
            />
            <img
              src={fig1}
              alt="每条边的描述长度，涵盖标签与 diff 结构各类描述子；最右柱为 diff 指纹，被标为恒等式且不作证据。"
              style={{ maxWidth: "100%", height: "auto" }}
            />
            <Text size="small" tone="secondary">
              图 1 —— 加了斜线的那一柱恒等于 log₂(n)，稿件中已明确标注它不构成证据。
            </Text>
          </Stack>
        </ReportSection>

        <ReportSection
          title="两个计划外的发现"
          description="两条都改变了这篇论文该怎么写，也都不只适用于 CUDA。"
          divided
        >
          <Stack gap="component">
            <Text>
              全部 5 条 <Tag tone="warning">Other</Tag> 边最后是同一件没被分类的事：别自己实现，改调库——其中包含样本里
              加速最大的两条（327× 与 69×）。预先约定的 25% 数值红线没有触发，因为拒绝对得整整齐齐：
              覆盖率百分比是检测「漏了一整类」的错误工具。
            </Text>
            <Text>
              55 条成功边里有 <Tag tone="warning">53 条</Tag> 同时改动了 <Code>@triton.autotune</Code>，
              也就是说在真实的 kernel 优化语料里，超参搜索与程序结构改动天然不分家——
              任何想做动作级信用分配的方案，都得先证明它们被人为拆开了。
            </Text>
          </Stack>
        </ReportSection>

        <ReportSection title="验证关卡" description="五道全部可重跑，本次运行全绿。" divided>
          <Table headers={["关卡", "检查内容", "结果"]} rows={gateRows} rowTone="success" />
        </ReportSection>

        <ReportSection
          title="这一轮被抓到的问题，以及是谁抓的"
          description="之所以记下来，是因为有意思的地方在于：自检一路通过，是评审环节没通过。"
          divided
        >
          <Table headers={["类别", "问题", "发现者"]} rows={caughtRows} />
        </ReportSection>

        <ReportSection title="交付物" divided>
          <Stack gap="component">
            <Table headers={["路径", "内容"]} rows={deliverableRows} />
            <Callout tone="info" title="提交历史">
              分支上 22 个提交，按产物类别切分——脚手架、Stage 1 检索、冻结词汇表、筛边、标注、统计、Stage 2.5、
              评审、统计修正、验证工具、修订、复审、Stage 4′、终验、Stage 5、关键词补做、必读核对、画布修复——
              任一阶段都能独立回退。
            </Callout>
          </Stack>
        </ReportSection>

        <ReportSection title="仍 open 的事项，是有意留着的" divided>
          <Stack gap="component">
            <Callout tone="warning" title="篇幅">
              正文约 1.37 万词，超出 spec 的 8–12 页目标。再往下砍只有两条路：把第 5 节压成半页，
              或者把撤回叙事与元发现移进附录——两者都会改变论文论证的重心，属于作者决定，不该由流水线代做，
              因此登记为已声明限制而不是做表面删减。
            </Callout>
            <Callout tone="danger" title="循环自证没有消解">
              标注由设计与词汇表同一个模型家族自动完成。稿件给出四条可判真伪的消解标准，并如实写明
              「四条一条都没做」。这是本文最大的真实弱点。
            </Callout>
            <Callout tone="warning" title="没有 PDF">
              本机无任何 TeX 引擎（latexmk、tectonic、xelatex、pdflatex、lualatex 全部缺失），
              因此只交付 LaTeX 源。没有改走 HTML 转 PDF 糊过去，原因写在 paper/latex/BUILD.md。
            </Callout>
            <Text tone="secondary" size="small">
              Stage 6（论文创作过程记录）随时可以做，等你一句话。
            </Text>
          </Stack>
        </ReportSection>
      </Stack>
    </ReportShell>
  );
}
