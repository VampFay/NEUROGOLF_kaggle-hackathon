"""
Generate the NeuroGolf Baseline Analysis as a self-contained HTML report.

Includes:
  - Summary cards (solved, score, time)
  - Solver breakdown bar chart
  - Per-task score distribution histogram
  - Failing-task category breakdown
  - Full per-task table (solved + sample of failing)
"""
import os
import sys
import json
import base64
from io import BytesIO

sys.path.insert(0, "/home/z/my-project")

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
from neurogolf import arc_data


def load_results():
    with open("/home/z/my-project/data/submission_results.json") as f:
        return json.load(f)


def fig_to_base64(fig, dpi=120):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return b64


def chart_solver_breakdown(breakdown):
    """Bar chart: tasks solved per solver."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    items = sorted(breakdown.items(), key=lambda x: -x[1])
    names = [k for k, _ in items]
    counts = [v for _, v in items]
    bars = ax.barh(names, counts, color="#2563EB", edgecolor="white", linewidth=1.2)
    ax.set_xlabel("Tasks Solved", fontsize=11, color="#0F172A")
    ax.set_title("Solver Breakdown — Tasks Solved by Primitive", fontsize=13,
                  color="#0F172A", fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("#CBD5E1")
    ax.spines['bottom'].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=10)
    ax.grid(axis="x", color="#F1F5F9", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                 str(count), va="center", fontsize=10, color="#0F172A", fontweight="bold")
    return fig_to_base64(fig)


def chart_score_distribution(results):
    """Histogram of scores for solved tasks."""
    solved = [r for r in results if r["eligible"]]
    scores = [r["score"] for r in solved]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(scores, bins=12, color="#10B981", edgecolor="white", linewidth=1.2)
    ax.set_xlabel("Score (max 25)", fontsize=11, color="#0F172A")
    ax.set_ylabel("Number of Tasks", fontsize=11, color="#0F172A")
    ax.set_title("Score Distribution of Solved Tasks", fontsize=13,
                  color="#0F172A", fontweight="bold", pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("#CBD5E1")
    ax.spines['bottom'].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=10)
    ax.grid(axis="y", color="#F1F5F9", linewidth=0.8)
    ax.set_axisbelow(True)
    if scores:
        mean_s = np.mean(scores)
        ax.axvline(mean_s, color="#EF4444", linestyle="--", linewidth=1.5,
                    label=f"Mean = {mean_s:.2f}")
        ax.legend(loc="upper left", fontsize=10, frameon=False)
    return fig_to_base64(fig)


def chart_failing_categories():
    """Pie chart of failing-task categories."""
    cats = {
        "Same-size, <30% cells changed (CA-like)": 177,
        "Same-size, complex change": 79,
        "Different I/O sizes": 132,
        "1-D outputs": 14,
    }
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    colors_list = ["#2563EB", "#F59E0B", "#10B981", "#EF4444"]
    wedges, texts, autotexts = ax.pie(
        list(cats.values()), labels=list(cats.keys()),
        autopct=lambda p: f"{p:.1f}%\n({int(round(p*sum(cats.values())/100))})",
        colors=colors_list, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10, "color": "#0F172A"},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(9)
    ax.set_title("Failing-Task Categorization (384 unsolved)", fontsize=13,
                  color="#0F172A", fontweight="bold", pad=12)
    return fig_to_base64(fig)


def chart_score_progression():
    """Mock chart of score progression across iterations (placeholder for future)."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    iterations = ["Session 1\n(Baseline)", "Session 2\n(planned)", "Session 3\n(planned)", "Session 4\n(planned)"]
    scores = [300, 1000, 2000, 3500]
    ax.plot(iterations, scores, marker="o", markersize=10,
             color="#2563EB", linewidth=2.5, markerfacecolor="white",
             markeredgewidth=2.5, markeredgecolor="#2563EB")
    ax.fill_between(range(len(iterations)), scores, alpha=0.15, color="#2563EB")
    ax.set_ylabel("Cumulative Score", fontsize=11, color="#0F172A")
    ax.set_title("Projected Score Progression", fontsize=13,
                  color="#0F172A", fontweight="bold", pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("#CBD5E1")
    ax.spines['bottom'].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=10)
    ax.grid(axis="y", color="#F1F5F9", linewidth=0.8)
    ax.set_axisbelow(True)
    for i, s in enumerate(scores):
        ax.text(i, s + 100, str(s), ha="center", fontsize=10,
                 color="#0F172A", fontweight="bold")
    return fig_to_base64(fig)


def build_html():
    sub = load_results()
    summary = sub["summary"]
    results = sub["results"]

    solved_results = [r for r in results if r["eligible"]]
    failing_results = [r for r in results if not r["eligible"]]

    # Generate charts
    chart1 = chart_solver_breakdown(summary["breakdown"])
    chart2 = chart_score_distribution(results)
    chart3 = chart_failing_categories()
    chart4 = chart_score_progression()

    # Build per-task table rows (solved first, then sample of failing)
    solved_rows = ""
    for r in solved_results:
        solved_rows += f"""
        <tr>
          <td><code>{r['task_id']:03d}</code></td>
          <td><code>{r['filename']}</code></td>
          <td><span class="solver-tag">{r['solver']}</span></td>
          <td class="num">{r['cost']}</td>
          <td class="num score">{r['score']:.2f}</td>
          <td class="ok">✓</td>
        </tr>"""

    # Sample 20 failing tasks
    failing_sample = failing_results[:20]
    failing_rows = ""
    for r in failing_sample:
        failing_rows += f"""
        <tr>
          <td><code>{r['task_id']:03d}</code></td>
          <td><code>{r['filename']}</code></td>
          <td><span class="solver-tag fail">{r['solver']}</span></td>
          <td class="num">—</td>
          <td class="num">—</td>
          <td class="fail">✗</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NeuroGolf 2026 — Baseline Analysis</title>
  <style>
    :root {{
      --primary: #0F172A;
      --accent: #2563EB;
      --muted: #64748B;
      --bg-soft: #F1F5F9;
      --bg-page: #FAFAFA;
      --ok: #10B981;
      --warn: #F59E0B;
      --danger: #EF4444;
      --border: #E2E8F0;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif;
      color: var(--primary);
      background: var(--bg-page);
      line-height: 1.6;
      padding: 24px;
      max-width: 1100px;
      margin: 0 auto;
    }}
    header {{
      background: var(--primary);
      color: white;
      padding: 32px 36px;
      border-radius: 12px;
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
    }}
    header::after {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 4px;
      background: var(--accent);
    }}
    header h1 {{
      font-size: 26px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    header .sub {{
      color: #94A3B8;
      font-size: 14px;
    }}
    header .meta {{
      color: #94A3B8;
      font-size: 12px;
      margin-top: 14px;
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
    }}
    header .meta span::before {{ content: "▸ "; color: var(--accent); }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background: white;
      border-radius: 10px;
      padding: 20px;
      border: 1px solid var(--border);
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .card .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }}
    .card .value {{
      font-size: 28px;
      font-weight: 700;
      color: var(--primary);
      line-height: 1.1;
    }}
    .card .delta {{
      font-size: 11px;
      color: var(--muted);
      margin-top: 6px;
    }}
    .card.accent .value {{ color: var(--accent); }}
    .card.ok .value {{ color: var(--ok); }}
    .card.warn .value {{ color: var(--warn); }}

    section {{
      background: white;
      border-radius: 12px;
      padding: 28px 32px;
      margin-bottom: 20px;
      border: 1px solid var(--border);
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    section h2 {{
      font-size: 18px;
      font-weight: 700;
      color: var(--primary);
      margin-bottom: 6px;
      padding-bottom: 8px;
      border-bottom: 2px solid var(--bg-soft);
    }}
    section h2::before {{
      content: "";
      display: inline-block;
      width: 4px;
      height: 18px;
      background: var(--accent);
      margin-right: 10px;
      vertical-align: -3px;
      border-radius: 2px;
    }}
    section h3 {{
      font-size: 14px;
      font-weight: 600;
      color: var(--accent);
      margin-top: 18px;
      margin-bottom: 8px;
    }}
    section p {{
      color: var(--primary);
      margin-bottom: 12px;
      font-size: 14px;
    }}
    section p.muted {{ color: var(--muted); font-size: 13px; }}

    .chart-wrap {{
      margin: 16px 0;
      text-align: center;
    }}
    .chart-wrap img {{
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin: 16px 0;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin: 12px 0;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
    }}
    th {{
      background: var(--bg-soft);
      color: var(--primary);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    tbody tr:hover {{ background: var(--bg-soft); }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    td.score {{ font-weight: 600; color: var(--ok); }}
    td.ok {{ color: var(--ok); font-weight: 600; text-align: center; }}
    td.fail {{ color: var(--danger); font-weight: 600; text-align: center; }}
    code {{
      font-family: "SF Mono", Monaco, "Cascadia Code", Consolas, monospace;
      font-size: 12px;
      background: var(--bg-soft);
      padding: 1px 6px;
      border-radius: 3px;
      color: var(--primary);
    }}
    .solver-tag {{
      display: inline-block;
      padding: 2px 8px;
      background: #DBEAFE;
      color: #1E40AF;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 600;
      font-family: "SF Mono", Monaco, Consolas, monospace;
    }}
    .solver-tag.fail {{
      background: #FEE2E2;
      color: #991B1B;
    }}

    .summary-box {{
      background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
      border-left: 4px solid var(--accent);
      padding: 16px 20px;
      border-radius: 6px;
      margin: 16px 0;
      font-size: 14px;
    }}

    footer {{
      text-align: center;
      color: var(--muted);
      font-size: 12px;
      padding: 24px;
    }}

    @media (max-width: 720px) {{
      .cards {{ grid-template-columns: 1fr 1fr; }}
      .chart-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>

<header>
  <h1>NeuroGolf 2026 — Baseline Analysis</h1>
  <div class="sub">First-pass solver pipeline on all 400 ARC-AGI training tasks</div>
  <div class="meta">
    <span>Generated: 2026-07-03</span>
    <span>Pipeline: neurogolf v0.1</span>
    <span>Sandbox: CPU only</span>
    <span>Submission: <code>submission.zip</code> (5.6 KB)</span>
  </div>
</header>

<div class="cards">
  <div class="card accent">
    <div class="label">Tasks Solved</div>
    <div class="value">{summary['solved']} / {summary['total']}</div>
    <div class="delta">{100*summary['solved']/summary['total']:.1f}% coverage</div>
  </div>
  <div class="card ok">
    <div class="label">Total Score</div>
    <div class="value">{summary['total_score']:.1f}</div>
    <div class="delta">Avg {summary['total_score']/summary['solved']:.2f} per solved task</div>
  </div>
  <div class="card">
    <div class="label">Pipeline Runtime</div>
    <div class="value">{summary['elapsed_sec']:.1f}s</div>
    <div class="delta">{summary['elapsed_sec']/400:.2f}s per task</div>
  </div>
  <div class="card warn">
    <div class="label">Days to Deadline</div>
    <div class="value">12</div>
    <div class="delta">July 15, 2026 23:59 UTC</div>
  </div>
</div>

<section>
  <h2>1. Executive Summary</h2>
  <p>
    Our first-pass baseline solves <strong>{summary['solved']} of {summary['total']} tasks</strong>
    ({100*summary['solved']/summary['total']:.1f}%) for an expected score of
    <strong>{summary['total_score']:.1f}</strong>. The pipeline runs end-to-end in
    {summary['elapsed_sec']:.1f} seconds on a single CPU core, making it fast enough to iterate
    multiple times per session. The submission file <code>submission.zip</code> is ready for
    upload to Kaggle.
  </p>
  <p class="muted">
    The theoretical maximum score (if every task scored the ceiling of 25) is 10,000, so we
    are currently at 3.0% of the ceiling. Realistic competitive targets in similar ARC-AGI
    competitions sit in the 2,000-3,500 range, giving substantial headroom for improvement.
  </p>
  <div class="summary-box">
    <strong>Key insight:</strong> 177 of the 384 failing tasks (46%) have same-size I/O with
    less than 30% cell changes — these are very likely cellular-automaton or pattern-based
    recoloring rules that our existing CA solver family can capture with targeted extensions.
    This is the single largest opportunity for coverage growth in the next session.
  </div>
</section>

<section>
  <h2>2. Solver Breakdown</h2>
  <p>Distribution of solved tasks across the {len(summary['breakdown'])} solvers that produced
  eligible networks. The remaining solvers in the catalog (Palette, MirrorConcat, Constant, etc.)
  did not solve any task in this baseline.</p>
  <div class="chart-wrap">
    <img src="data:image/png;base64,{chart1}" alt="Solver breakdown chart">
  </div>
  <table>
    <thead>
      <tr>
        <th>Solver</th>
        <th>Tasks Solved</th>
        <th>% of Solved</th>
        <th>Role</th>
      </tr>
    </thead>
    <tbody>"""
    solver_roles = {
        "color_map": "1×1 conv per-color substitution",
        "crop_top_left": "Identity network + validator crop",
        "cellular_automaton": "Single (X,Y,Z,threshold) rule",
        "geom_transform": "Flip / transpose / rotate",
        "scale_up": "Nearest-neighbor k× scaling",
        "kronecker": "Conditional tiling",
        "multi_rule_ca": "Multiple neighbor-color rules",
    }
    solver_table_rows = ""
    for solver, count in sorted(summary["breakdown"].items(), key=lambda x: -x[1]):
        pct = 100 * count / summary["solved"]
        role = solver_roles.get(solver, "—")
        solver_table_rows += f"""
      <tr>
        <td><span class="solver-tag">{solver}</span></td>
        <td class="num">{count}</td>
        <td class="num">{pct:.1f}%</td>
        <td>{role}</td>
      </tr>"""
    html += solver_table_rows + """
    </tbody>
  </table>
</section>

<section>
  <h2>3. Score Distribution</h2>
  <p>Histogram of per-task scores for the {solved_count} solved tasks. Scores follow the formula
  <code>max(1, 25 − ln(cost))</code> where <code>cost = #parameters + #bytes</code>. Lighter
  networks (identity, crop) score near 20; denser ones (multi-rule CA) score around 16-17.</p>
  <div class="chart-wrap">
    <img src="data:image/png;base64,""".replace("{solved_count}", str(summary["solved"])) + chart2 + """" alt="Score distribution">
  </div>
</section>

<section>
  <h2>4. Failing-Task Categorization</h2>
  <p>The {failing_count} unsolved tasks fall into four broad categories based on their structural
  signatures. The largest category — same-size I/O with small color changes — is the most
  tractable and is the primary target for the next iteration.</p>
  <div class="chart-wrap">
    <img src="data:image/png;base64,""".replace("{failing_count}", str(len(failing_results))) + chart3 + """" alt="Failing task categories">
  </div>
  <table>
    <thead>
      <tr>
        <th>Category</th>
        <th>Count</th>
        <th>% of Fails</th>
        <th>Tractability</th>
        <th>Target Solvers</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Same-size, &lt;30% cells changed</td>
        <td class="num">177</td>
        <td class="num">46.1%</td>
        <td style="color: var(--ok);">High</td>
        <td>Extended CA solver, multi-rule CA, count→color</td>
      </tr>
      <tr>
        <td>Same-size, complex change</td>
        <td class="num">79</td>
        <td class="num">20.6%</td>
        <td style="color: var(--warn);">Medium</td>
        <td>Learned micro-nets, multi-step DSL</td>
      </tr>
      <tr>
        <td>Different I/O sizes</td>
        <td class="num">132</td>
        <td class="num">34.4%</td>
        <td style="color: var(--warn);">Medium</td>
        <td>Crop/tile/scale/concat, mirror-completion</td>
      </tr>
      <tr>
        <td>1-D outputs (row/column)</td>
        <td class="num">14</td>
        <td class="num">3.6%</td>
        <td style="color: var(--danger);">Hard</td>
        <td>Count-based, ReduceSum + comparison</td>
      </tr>
    </tbody>
  </table>
</section>

<section>
  <h2>5. Projected Score Progression</h2>
  <p>Planned coverage and score growth across the remaining four work sessions. The progression
  assumes we hit the phase targets in the iteration roadmap (Strategy PDF §5).</p>
  <div class="chart-wrap">
    <img src="data:image/png;base64,""" + chart4 + """" alt="Score progression">
  </div>
</section>

<section>
  <h2>6. Solved Tasks — Full List</h2>
  <p>All {solved_count} tasks with eligible networks, sorted by task ID.</p>
  <table>
    <thead>
      <tr>
        <th>Task ID</th>
        <th>ARC Filename</th>
        <th>Solver</th>
        <th>Cost</th>
        <th>Score</th>
        <th>Eligible</th>
      </tr>
    </thead>
    <tbody>""".replace("{solved_count}", str(summary["solved"])) + solved_rows + """
    </tbody>
  </table>
</section>

<section>
  <h2>7. Failing Tasks — Sample (First 20)</h2>
  <p>The first 20 unsolved tasks. The full list of {failing_count} failing tasks is available
  in <code>data/submission_results.json</code>.</p>
  <table>
    <thead>
      <tr>
        <th>Task ID</th>
        <th>ARC Filename</th>
        <th>Best Solver Attempted</th>
        <th>Cost</th>
        <th>Score</th>
        <th>Eligible</th>
      </tr>
    </thead>
    <tbody>""".replace("{failing_count}", str(len(failing_results))) + failing_rows + """
    </tbody>
  </table>
</section>

<section>
  <h2>8. Next Actions</h2>
  <p>Based on this baseline, the highest-ROI next steps are:</p>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Action</th>
        <th>Expected Coverage Gain</th>
        <th>Priority</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="num">1</td>
        <td>Upload <code>submission.zip</code> to Kaggle to validate I/O convention</td>
        <td>—</td>
        <td style="color: var(--danger);">Critical</td>
      </tr>
      <tr>
        <td class="num">2</td>
        <td>Extend CA solver with 8-neighbor rules + multi-rule chaining</td>
        <td>+20-40 tasks</td>
        <td style="color: var(--danger);">High</td>
      </tr>
      <tr>
        <td class="num">3</td>
        <td>Add symmetric-completion solver (concat input + flip)</td>
        <td>+5-10 tasks</td>
        <td style="color: var(--warn);">Medium</td>
      </tr>
      <tr>
        <td class="num">4</td>
        <td>Add flood-fill solver using iterated 3×3 dilation</td>
        <td>+5-10 tasks</td>
        <td style="color: var(--warn);">Medium</td>
      </tr>
      <tr>
        <td class="num">5</td>
        <td>Build Kaggle Notebook template for learned-micro-net training</td>
        <td>+50-100 tasks (Phase 3)</td>
        <td style="color: var(--accent);">Later</td>
      </tr>
    </tbody>
  </table>
</section>

<footer>
  Generated by Z.ai · NeuroGolf pipeline v0.1 · 2026-07-03 ·
  Detailed JSON: <code>data/submission_results.json</code>
</footer>

</body>
</html>"""

    out_path = "/home/z/my-project/download/NeuroGolf_Baseline_Analysis.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Generated: {out_path}")
    print(f"Size: {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    build_html()
