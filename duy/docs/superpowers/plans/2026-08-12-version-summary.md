# Version-Scoped Kaggriculture Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure root-level `SUMMARY.md` so every agent version owns one self-contained strategy and performance section.

**Architecture:** Keep a compact champion declaration and cross-version comparison table at the top. Store version-specific facts below under parallel `### <version>` sections with `#### Strategy` and `#### Performance` subsections, allowing future versions to append without mixing their details.

**Tech Stack:** GitHub-flavored Markdown and read-only Python verification against the existing benchmark JSON.

## Global Constraints

- Modify only root-level `SUMMARY.md`.
- Keep `00_baseline` as the current best and only qualified local version.
- Keep the existing verified metrics and evidence links unchanged.
- Move all strategy and performance details into the `### 00_baseline` section.
- Future versions must use parallel `### <version>`, `#### Strategy`, and `#### Performance` subsections.
- Preserve the compact version comparison table near the top.
- Do not stage `.DS_Store` or unrelated `another_work/` changes.

---

### Task 1: Scope Strategy and Performance by Version

**Files:**
- Modify: `SUMMARY.md`

**Interfaces:**
- Consumes: the existing summary, `another_work/00_baseline/STRATEGY.md`, and `benchmarks/results/20260812T012722Z_00_baseline_vs_demo_agent/summary.json`.
- Produces: `SUMMARY.md` with top-level champion/comparison information and one complete section per version.

- [ ] **Step 1: Verify the current summary contains generic detail sections**

Run:

```bash
.venv/bin/python -c "from pathlib import Path; text=Path('SUMMARY.md').read_text(); assert '## Strategy Summary' in text; assert '## Current Performance' in text; assert '### `00_baseline`' not in text; print('RED: details are not version-scoped')"
```

Expected: `RED: details are not version-scoped`.

- [ ] **Step 2: Restructure the document**

Use this heading order:

```markdown
# Kaggriculture Agent Summary

## Current Best

Champion declaration for `00_baseline`.

## Version Comparison

The existing append-friendly comparison table and promotion sentence.

## Versions

### `00_baseline`

#### Strategy

The existing concise strategy bullets, strength/limitation note, and full
strategy link.

#### Performance

The existing benchmark protocol, performance table, and three evidence links.
```

Keep the existing strategy wording and numeric values. Remove the generic
`## Strategy Summary` and `## Current Performance` headings. Do not add testing
instructions or other details.

- [ ] **Step 3: Verify structure, links, and metrics**

Run:

```bash
.venv/bin/python -c "import json,pathlib,re; root=pathlib.Path('.'); text=(root/'SUMMARY.md').read_text(); assert '## Strategy Summary' not in text; assert '## Current Performance' not in text; required=['## Current Best','## Version Comparison','## Versions','### `00_baseline`','#### Strategy','#### Performance']; assert all(value in text for value in required); assert text.index('## Current Best') < text.index('## Version Comparison') < text.index('## Versions') < text.index('### `00_baseline`') < text.index('#### Strategy') < text.index('#### Performance'); links=re.findall(r'\[[^]]+\]\(([^)]+)\)', text); assert all((root/link).exists() for link in links); summary=json.loads((root/'benchmarks/results/20260812T012722Z_00_baseline_vs_demo_agent/summary.json').read_text())['summary']; values=['100 wins / 0 losses / 0 ties','\$151,986.04','\$154,871.00','\$79,225.00','\$192,508.00','\$3,368.94','+\$148,617.10','+\$148,422.00','+\$148,812.20']; assert all(value in text for value in values); assert summary['games']==100; print(f'PASS: version-scoped summary, {len(links)} valid links')"
```

Expected: `PASS: version-scoped summary, 5 valid links`.

- [ ] **Step 4: Check formatting and repository isolation**

Run:

```bash
git diff --check -- SUMMARY.md
git status --short
```

Expected: formatting passes and only `SUMMARY.md` is staged for this change;
existing unrelated working-tree changes remain untouched.

- [ ] **Step 5: Commit only the summary restructure**

```bash
git add SUMMARY.md
git commit -m "docs: scope agent summary by version"
```
