"""
Assemble the GitHub Pages site from the committed run artifacts.

WHAT THIS PUBLISHES
-------------------
The two graded projects' own generated reports - not a hand-written marketing
page. The point of publishing is that a stranger can read the actual output of
the pipeline, including the parts that are unflattering.

The index page's numbers are read from final_model_selection.json at build
time, so the site cannot drift out of step with the artifacts beside it. An
artifact that cannot be read is rendered as "not available" rather than being
quietly omitted - the same rule the model card follows.

Run by: .github/workflows/pages.yml
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# The published numbers come from the same module the deck and the written
# report read, so the site cannot quote a different figure for the same run.
from src.reporting.facts import PROJECTS, warn_if_stale  # noqa: E402

SITE = Path("_site")


def copy_if(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def build() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)

    cards: list[str] = []
    for facts in PROJECTS:
        slug, name, task, root = (facts.slug, facts.display_name,
                                  facts.task, facts.root)
        champion = facts.champion_name
        version = facts.champion_version
        metric = facts.selection_metric
        # The same canonical strings the deck and the report print, so a reader
        # comparing the three documents sees identical figures.
        test = facts.test_line
        validation = facts.validation_line

        links: list[str] = []
        if copy_if(root / "artifacts" / "reports" / "project_report.html",
                   SITE / slug / "report.html"):
            links.append(f'<a href="{slug}/report.html">Full report</a>')
        if copy_if(root / "artifacts" / "trust" / "model_card.md",
                   SITE / slug / "model_card.md"):
            links.append(f'<a href="{slug}/model_card.md">Model card</a>')
        if copy_if(root / "artifacts" / "reports" / "executive_briefing.md",
                   SITE / slug / "executive_briefing.md"):
            links.append(f'<a href="{slug}/executive_briefing.md">Briefing</a>')
        if copy_if(root / "artifacts" / "trust" / "subgroup_fairness.json",
                   SITE / slug / "subgroup_fairness.json"):
            links.append(
                f'<a href="{slug}/subgroup_fairness.json">Fairness audit</a>')

        links_html = " &middot; ".join(links) if links else \
            "<span class='muted'>no artifacts published for this project</span>"

        cards.append(f"""
      <article class="card">
        <h2>{name}</h2>
        <p class="task">{task} &middot; selected on <code>{metric}</code></p>
        <dl>
          <dt>Champion</dt><dd>{champion} {version}</dd>
          <dt>Validation</dt><dd>{validation}</dd>
          <dt>Test <span class="muted">(opened once)</span></dt><dd>{test}</dd>
        </dl>
        <p class="links">{links_html}</p>
      </article>""")

    (SITE / "index.html").write_text(INDEX.format(cards="\n".join(cards)),
                                     encoding="utf-8")
    # Without this, GitHub Pages runs the content through Jekyll, which skips
    # files and directories beginning with an underscore.
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built {SITE} with {len(PROJECTS)} project(s).")
    print(warn_if_stale())


INDEX = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Data Science Workbench</title>
<style>
  :root {{
    --ink: #1a1f2b; --muted: #5b6474; --accent: #1f4e79;
    --soft: #e8eff6; --rule: #d5dbe2; --bg: #ffffff;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ink: #e8ecf2; --muted: #9aa5b4; --accent: #7fb2e0;
      --soft: #1b2430; --rule: #2c3644; --bg: #11161d;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; padding: 48px 16px 64px; }}
  header {{ border-bottom: 2px solid var(--accent); padding-bottom: 20px;
            margin-bottom: 28px; }}
  h1 {{ margin: 0 0 6px; font-size: 1.9rem; letter-spacing: -0.02em; }}
  .sub {{ color: var(--muted); margin: 0; }}
  .claim {{
    background: var(--soft); border-left: 4px solid var(--accent);
    padding: 14px 18px; margin: 24px 0 32px; border-radius: 0 6px 6px 0;
  }}
  .card {{
    border: 1px solid var(--rule); border-radius: 8px;
    padding: 20px 22px; margin-bottom: 20px;
  }}
  .card h2 {{ margin: 0 0 4px; font-size: 1.2rem; color: var(--accent); }}
  .task {{ color: var(--muted); margin: 0 0 14px; font-size: 0.9rem; }}
  dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 18px;
        margin: 0 0 16px; }}
  dt {{ color: var(--muted); font-size: 0.87rem; }}
  dd {{ margin: 0; font-variant-numeric: tabular-nums; }}
  .links a {{ color: var(--accent); text-decoration: none;
              border-bottom: 1px solid var(--rule); }}
  .links a:hover {{ border-bottom-color: var(--accent); }}
  .muted {{ color: var(--muted); }}
  footer {{ margin-top: 36px; padding-top: 20px; border-top: 1px solid var(--rule);
            color: var(--muted); font-size: 0.87rem; }}
  code {{ font-size: 0.9em; }}
  @media (max-width: 520px) {{
    dl {{ grid-template-columns: 1fr; gap: 2px 0; }}
    dd {{ margin-bottom: 8px; }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>AI Data Science Workbench</h1>
      <p class="sub">A governed, task-agnostic machine learning pipeline</p>
    </header>

    <div class="claim">
      The claim is not that these models are accurate. It is that each result is
      reproducible, each model choice is documented, leakage is structurally
      prevented, and each model's own bias is measured and published beside its
      accuracy. Every page below is generated by the pipeline itself.
    </div>
{cards}

    <footer>
      Reports, model cards and fairness audits on this page are produced by the
      pipeline and committed to the repository. Continuous integration re-runs
      both pipelines end to end from a clean checkout on every push.
    </footer>
  </div>
</body>
</html>
"""


if __name__ == "__main__":
    build()
