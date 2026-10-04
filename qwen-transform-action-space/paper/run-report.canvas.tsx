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
} from "qoder/canvas";

const fig5 = canvasImage("./fig5_permutation_calibration.png");

const headlineMetrics = [
  { label: "成稿页数", value: "38", description: "xelatex 编译，5 图全嵌入" },
  { label: "参考文献", value: "36", tone: "success" as const, description: "每条回源核验" },
  { label: "实证语料", value: "2,510", description: "条真实成功优化边（55 条精标）" },
  { label: "可重跑脚本", value: "17", description: "含 5 道验证关卡" },
];

const answerRows = [
  ["论文主张什么", "将 LLM 优化 kernel 的学习/搜索单元从 token 改为「带类型的程序变换」a = (变换类型, 位置, 参数)，因为一次优化通常是跨多处的一个语义动作"],
  ["用真数据测到了什么", "12 类冻结词汇表覆盖 90.9%（实际只用 10 类，且主动招引 Other）；标签比同等粒度的 diff 结构更能预测实测加速"],
  ["最硬的一个数字", "把优化按加速排四分位后，标签与结果的互信息 0.697 比特，置换检验 p = 0.0005；而 diff 粗结构 p = 0.752、只看 hunk 数 p = 0.436 都不显著"],
  ["最该被引用的一条负面结果", "同一批边换一个只给类名、不给定义的标注员重标，p 变成 0.106（不显著）。说明词汇表的信息量有一部分是「怎么问出来的」造成的"],
  ["没验证的部分", "论文声称的搜索效率与学习效率收益一个都没测——本稿不含任何干预实验，只给出一套五条件实验协议"],
];

const evidenceRows = [
  ["词汇表能覆盖真实成功优化", "支持", "10 类覆盖 90.9%；Other 被主动招引，5 条全部指向同一件没被分类的事"],
  ["标签携带超出自身粒度的信息", "有条件支持", "给定义时 p = 0.0005；不给定义时 p = 0.106；两轮一致的 38 条子集 p = 0.0016"],
  ["标签更短", "支持", "2.67 比特对 4.26 比特（对同一批 diff 的忠实粗描述）"],
  ["同一动作跨任务复现", "不支持", "在标签置换零模型下等于随机水平"],
  ["一个动作横跨多个代码位置", "无法辨识", "55 条边中没有一条是单标签，任何多区域统计量在此都不可区分"],
  ["后端库选择是一类动作", "支持，且此前被漏掉", "样本里回报最高的两条（327×、69×）全是「别自己写，改调 cuBLAS/cuDNN」"],
];

const deliverableRows = [
  ["paper/main.pdf", "英文成稿 38 页（已验证尾部章节齐备、页数与词数相符、5 图嵌入、无缺字）"],
  ["paper/main.md / paper/latex/", "英文源 + LaTeX 工程 + references.bib"],
  ["paper/zh/full.md", "中文对照稿，与英文定量 token、符号、引用编号逐项对齐"],
  ["data/", "标注集、词汇表证据样例、校准统计、盲标与开放编码结果、24 条留出边"],
  ["scripts/", "筛边、标注、统计、5 道关卡、LaTeX/PDF 构建、中英对齐"],
  ["stage*/ 与 STAGE*.md", "检索账、实验报告、评审 5 份、编辑决定、修订处置、终验报告"],
];

const gatesRows = [
  ["引用回源", "36/36", "arXiv 批量 Atom + CrossRef DOI + URL 三类解析器，逐字段比标题与作者"],
  ["数字断言", "117/117", "全部从 data/ 原始语料重导，不看论文自述"],
  ["中英对齐", "通过", "引用编号同序、定量 token 多重集相等、符号齐备"],
  ["规范符合", "通过", "摘要 250 词（要求 200–250）、字数标记与实测一致"],
  ["PDF 结构", "通过", "尾部章节存在 + 页数合理 + 图数一致 + 日志无硬错误无缺字"],
];

export default function TransformActionSpaceRunReport() {
  return (
    <ReportShell width="wide" ariaLabel="程序变换动作空间论文成果报告">
      <Stack gap="sectionCompact">
        <header>
          <Stack gap="component">
            <H1>论文成果报告：把 kernel 优化智能体的动作单元从 token 换成程序变换</H1>
            <Text tone="secondary">
              分支 <Code>qwen/transform-action-space</Code> · 定位 arXiv 分析/立场论文（8–12 页） ·
              Position/Analysis，不含干预实验 · 2026-10-02
            </Text>
            <MetricsGrid variant="header" columns={4} items={headlineMetrics} />
          </Stack>
        </header>

        <ReportSection
          title="这篇论文说了什么，以及测到了什么"
          description="不预设读者了解写作过程。五组问答即论文的全部实质内容。"
          divided
        >
          <Table headers={["问题", "回答"]} rows={answerRows} />
        </ReportSection>

        <ReportSection
          title="证据强度：支持到哪一步，不支持在哪一步"
          description="公开 AdaExplore 轨迹，语料只有 Triton（22,007 个 Triton 文件、0 个 CUDA C++ 文件）。"
          divided
        >
          <Stack gap="component">
            <Table
              headers={["论断", "判定", "依据"]}
              rows={evidenceRows}
              rowTone={["success", "caution", "success", "critical", "caution", "success"]}
            />
            <img
              src={fig5}
              alt="每个描述子与实测加速四分位之间的互信息，以及各自的对称置换零模型；只有给定定义标注的语义标签超出自身零模型。"
              style={{ maxWidth: "100%", height: "auto" }}
            />
            <Text size="small" tone="secondary">
              图 5 —— 三条虚线是各自标签的置换零均值；点为 95% 分位。观测值落在点右侧才算超出自身零模型：
              只有前两根柱子满足，第三根（diff 粗结构）观测值反而低于自身零均值。图内文字为英文，因图随英文成稿生成。
            </Text>
          </Stack>
        </ReportSection>

        <ReportSection
          title="两个计划外发现，都可能影响别人怎么设计这类系统"
          divided
        >
          <Stack gap="component">
            <Text>
              <Tag tone="warning">后端库选择是一类被漏掉的动作</Tag>
              样本中回报最高的两条（327×、69×）都是「删掉手写 kernel，改调 cuBLAS / F.cross_entropy」。
              值得注意的不是这个常识，而是<b>预先约定的数值红线没能发现它</b>：`Other` 只占 9.1%，远低于 25% 阈值，
              但 5 条拒绝全属同一件事。覆盖率百分比是检测「漏了一整类」的错误工具。
            </Text>
            <Text>
              <Tag tone="warning">调参与结构改写在真实轨迹里不可分</Tag>
              55 条成功边里 <Tag tone="warning">53 条</Tag>同时改动了 <Code>@triton.autotune</Code>（新增、删除、改候选列表、
              改 cache key）。独立的自底向上开放编码也重现了这一点：启动常数只作为伴随出现（19 次伴随、0 次独立成边）。
              任何想做动作级信用分配的方案都得先证明它们被人为拆开了——否则「学习动作」只是幻觉。
            </Text>
          </Stack>
        </ReportSection>

        <ReportSection title="交付物与可核验性" divided>
          <Stack gap="component">
            <Table headers={["路径", "内容"]} rows={deliverableRows} />
            <Callout tone="info" title="全部关卡当前状态">
              引用 36/36 · 数字断言 117/117 · 中英对齐通过 · 摘要 250 词达标 · PDF 结构性验证通过（38 页 / 5 图）。
              每条都是可重跑脚本，不依赖人工目测。
            </Callout>
          </Stack>
        </ReportSection>

        <ReportSection title="局限（论文正文已如实写明，不是免责套话）" divided>
          <Table
            headers={["限制", "现状"]}
            rows={[
              ["循环自证未彻底消解", "已执行 3 条控制；仍缺真正独立（人类）第二标注与第二个家族"],
              ["L2 / L3 论断未检验", "无任何干预实验；§4 给的是协议与止损分支，不是结果"],
              ["正文 1.5 万词，超 8–12 页目标", "进一步删减会改变论证重心，属作者决定；两个候选方案已写进 README"],
              ["统计功效受样本量限制", "55 条边、45 个任务，每任务至多 3 条；跨任务复现因此不可与随机区分"],
            ]}
            rowTone={["caution", "neutral", "caution", "neutral"]}
          />
        </ReportSection>
      </Stack>
    </ReportShell>
  );
}
