"""Render all completed observations from a fixed partial summary; never run actors."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
ARMS = ['original', 'formatted', 'source_core', 'task_only', 'phase4',
        'phase4+permissions', 'phase4+checker', 'phase4+permissions+checker']
LABELS = ['Original', 'Formatting only', 'Core rules', 'Task only', 'Phase 4 shortened',
          'Shortened + permissions', 'Shortened + checker', 'Shortened + both']
TASKS = ['causal_features', 'branch_features']
COLORS = ['#276A8C', '#BE6B35']
MARKERS = ['o', '^']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(summary):
    """One row per attempt, including unknown or externally interrupted outcomes."""
    rows = []
    seen = set()
    for t in summary['trials']:
        eligible = bool(t.get('model_observation_eligible'))
        valid = eligible and bool(t.get('valid_backend'))
        row = dict(id=t['id'], arm=t['arm'], task=t['task'], repeat=t['repeat'],
                   terminal=bool(t['terminal_result_available']), complete=eligible,
                   valid_backend=valid, state=t.get('state', 'not_started'),
                   result_sha256=t.get('result_sha256', ''),
                   generated_tokens=(t.get('usage') or {}).get('output_tokens'),
                   latency_ratio_200=None, highest_no_backlog_rate=None,
                   cost_included=eligible, performance_included=False,
                   exclusion_reason='external_interruption' if row_terminal(t) else 'not_started')
        if eligible:
            key = (t['arm'], t['task'], t['repeat'])
            assert key not in seen, f'Duplicate completed observation: {key}'
            seen.add(key)
            row['exclusion_reason'] = 'backend_invalid' if not valid else 'measurement_missing'
        paired = (t.get('final_benchmark') or {}).get('paired_measurements', {})
        cand = paired.get('candidate', {}).get('loads', [])
        ref = paired.get('reference', {}).get('loads', [])
        if valid and (t.get('final_benchmark') or {}).get('ok') and cand:
            c200 = next((x for x in cand if x['offered_requests_per_second'] == 200), None)
            r200 = next((x for x in ref if x['offered_requests_per_second'] == 200), None)
            assert c200 is not None and r200 is not None, t['id']
            assert r200['first_output_ms']['median'] > 0, t['id']
            row['latency_ratio_200'] = c200['first_output_ms']['median'] / r200['first_output_ms']['median']
            row['highest_no_backlog_rate'] = max(
                (x['offered_requests_per_second'] for x in cand if x['rate_gate_passed']), default=0)
            row['performance_included'] = True
            row['exclusion_reason'] = ''
        rows.append(row)
    return rows


def row_terminal(t):
    return bool(t['terminal_result_available'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path,
                        default=BASE / 'evidence/repeat-partial-20261020/summary.json')
    parser.add_argument('--output', type=Path,
                        default=BASE / 'evidence/repeat-partial-20261020/figure')
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    rows = extract(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / 'observations.csv'
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    source.write_bytes(buffer.getvalue().encode('utf-8'))
    complete = [r for r in rows if r['complete']]
    perf = [r for r in rows if r['performance_included']]
    counts = [sum(r['arm'] == arm for r in complete) for arm in ARMS]
    assert len(rows) == 57 and len(complete) == 37 and len(perf) == 36

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7,
                         'axes.labelsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'xtick.major.width': .6,
                         'ytick.major.width': .6, 'savefig.facecolor': 'white'})
    # Fixed final size, 183 mm: ample label margin and unequal cost/performance panels.
    width_mm = 183
    height_mm = 116
    fig, axes = plt.subplots(1, 3, figsize=(width_mm / 25.4, height_mm / 25.4),
                             gridspec_kw={'width_ratios': [1.25, 1, 1]}, sharey=True)
    fig.subplots_adjust(left=.285, right=.975, bottom=.19, top=.76, wspace=.27)
    for ax in axes:
        ax.set_ylim(7.65, -.65)
        ax.set_yticks(range(8))
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E6E6E6', linewidth=.5)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticklabels([f'{label}  ({count}/6)' for label, count in zip(LABELS, counts)])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].set_xlim(0, 105)
    axes[0].set_xticks([0, 30, 60, 90])
    axes[0].set_xlabel('Generated tokens (thousands)')
    # Log axis separates fast candidates while retaining a slow, valid outlier.
    # Log scale requires strictly positive ratios; missing observations never enter it.
    assert all(r['latency_ratio_200'] > 0 for r in perf)
    ratios = [r['latency_ratio_200'] for r in perf]
    axes[1].set_xscale('log')
    axes[1].set_xlim(min(.1, min(ratios) / 1.2), max(2, max(ratios) * 1.15))
    axes[1].set_xticks([.1, .3, 1, 3, 10])
    axes[1].set_xticklabels(['0.1', '0.3', '1', '3', '10'])
    axes[1].set_xlim(min(.1, min(ratios) / 1.2), max(2, max(ratios) * 1.15))
    axes[1].axvline(1, color='#777777', linestyle='--', linewidth=.7, zorder=1)
    axes[1].set_xlabel('First-output latency ratio\n(candidate / reference; log scale)')
    axes[2].set_xlim(-12, 220)
    axes[2].set_xticks([0, 50, 100, 200])
    axes[2].set_xlabel('Highest tested no-backlog rate\n(requests per second)')
    axes[2].axvline(200, color='#999999', linestyle=':', linewidth=.7)
    for i, title in enumerate(['Generation cost', 'Latency at 200 req/s', 'Tested load ceiling']):
        axes[i].set_title(title, fontsize=8, loc='left', pad=14)
        axes[i].text(-.07, 1.13, 'abc'[i], transform=axes[i].transAxes,
                     fontsize=8, fontweight='bold')
    for r in complete:
        ti = TASKS.index(r['task'])
        # Fixed offsets encode all six planned task/repeat slots; no data-selected jitter.
        y = ARMS.index(r['arm']) + (-.19 if ti == 0 else .19) + (r['repeat'] - 2) * .105
        style = dict(marker=MARKERS[ti], c=COLORS[ti], s=15, linewidths=.35,
                     edgecolors='white', zorder=3)
        axes[0].scatter(r['generated_tokens'] / 1000, y, **style)
        if r['performance_included']:
            axes[1].scatter(r['latency_ratio_200'], y, **style)
            axes[2].scatter(r['highest_no_backlog_rate'], y, **style)
        else:
            axes[0].annotate(r['id'] + '*', (r['generated_tokens'] / 1000, y),
                             xytext=(-2, -10), textcoords='offset points', fontsize=6,
                             ha='right', color='#555555')
    legend = [Line2D([], [], marker=m, color='none', markerfacecolor=c,
                     markeredgecolor='white', markersize=5, label=label)
              for m, c, label in zip(MARKERS, COLORS, ['Causal chain', 'Two branches'])]
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.28, .93),
               frameon=False, ncol=2, handletextpad=.3, columnspacing=1.8)
    fig.text(.035, .97, 'Visible-budget batch · partial observations', fontsize=10, va='top')
    fig.text(.035, .075, '37 complete observations; 9 external interruptions; 11 attempts pending.', fontsize=7)
    fig.text(.035, .035, '* t004: backend failed; cost retained. 200 req/s is a test limit, not maximum capacity.', fontsize=7)
    fig.savefig(args.output / 'partial-comparison.svg')
    fig.savefig(args.output / 'partial-comparison.pdf')
    fig.savefig(args.output / 'partial-comparison.png', dpi=600)
    plt.close(fig)
    # Matplotlib puts trailing spaces in multiline SVG path attributes. Newlines
    # retain their SVG whitespace meaning after removing those redundant spaces.
    svg = args.output / 'partial-comparison.svg'
    svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())
                     + '\n').encode('utf-8'))
    meta = {'summary_sha256': sha(args.summary), 'plot_script_sha256': sha(Path(__file__)),
            'source_csv_sha256': sha(source), 'all_attempts': len(rows),
            'generation_observations': len(complete), 'performance_observations': len(perf),
            'ceiling_200_count': sum(r['highest_no_backlog_rate'] == 200 for r in perf),
            'condition_complete_counts': dict(zip(ARMS, counts)),
            'backend': 'Python/matplotlib', 'matplotlib_version': matplotlib.__version__,
            'missingness': 'All attempts retained in CSV; panel inclusion and reasons explicit.',
            'exports': {ext: sha(args.output / f'partial-comparison.{ext}') for ext in ['svg', 'pdf', 'png']}}
    (args.output / 'provenance.json').write_bytes((json.dumps(meta, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
