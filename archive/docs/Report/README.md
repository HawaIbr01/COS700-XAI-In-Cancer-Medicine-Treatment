# COS700 Final Report -- LaTeX Scaffolding

Structure mirrors the proposal: `main.tex` pulls in one file per
section/subsection via the `subfiles` package, so each person can open
and compile *just their own file* while waiting on others, without
needing the whole document to build.

## Compiling

- Whole report: compile `main.tex` (needs `biber` for the bibliography).
- A single section on its own: compile that section's `.tex` file
  directly, e.g. `sections/04a_sam_regularisation.tex`. The subfiles
  package pulls in the same preamble as `main.tex` automatically.

## ⚠️ Provisional until the official template arrives

The COS700 lecture slides ("Research Methods, Lecture 8") state a LaTeX
template will be provided and **must** be used -- no submission /
automatic zero otherwise. That template isn't available yet. Treat this
scaffold as a working structure to draft *content* into, not the final
submission shell. Once the real template lands, migrate section content
across rather than submitting this file as-is.

## Known conflict: chapter order

The COS700 slides' "typical structure" lists Methodology (3) before
Background (4) and Related Work (5). The actual example report we're
following (`Final_Research_Report_no_turnitin.pdf`, NICOG-approved)
puts Background/Related Work before Methodology instead. Per the
standing NICOG-takes-priority rule, this scaffold keeps the example
report's order (Introduction, Problem Statement, Literature Review,
Related Work, Methodology, Results, Discussion, Conclusion). Flag this
with a supervisor if it matters for marking.

## Merged from the proposal's main.tex

- All packages the proposal used (`amsmath`, `amssymb`, `color`,
  `listings`, `lmodern`, `mdframed`, `url`, `caption`, `subcaption`,
  `float`, `cite`, `xcolor`, `pgfgantt`, `rotating`, `tikz`,
  `pdflscape`), plus `geometry{margin=1in}`, the colour definitions,
  `lstset` code-listing style, `mdfsetup`, and `\BlackBox`.
- **Bibliography switched to plain bibtex** (`\bibliographystyle{plain}`
  + `\bibliography{references}`), matching the proposal's setup after
  it moved off biblatex/biber for labelling reasons. `references.bib`
  stays a plain `.bib` file, no `biber` run needed -- just `bibtex`.
- **Title page** now matches the proposal's cover page: UP logo,
  department/university lines, student numbers, supervisors.
  `up_logo.png` must sit alongside `main.tex` (i.e. in `docs/report/`)
  for `\includegraphics{up_logo.png}` to resolve.
- `\lstset` defaults to `language=Python` -- the proposal used C++;
  confirm which fits the report's actual code listings and change if
  needed.
- Line spacing is 1.5 (`\onehalfspacing`) per the COS700 report slides,
  not the proposal's `\linespread{1.25}` -- flag with a supervisor if
  that's wrong.

## Format (per COS700 slides)

- Single column, A4, 1.5 line spacing, 12pt font -- all set in
  `main.tex` (`\onehalfspacing`, `12pt` class option).
- Length: at least 30 pages, excluding prefix/ToC/references.
- **Reported format**: write in past tense, describing what was done
  (not what is intended, as in the proposal).
- Prefix should include abstract + keywords (both added) and a title
  page; the slides also mention a repeated "cover page" ahead of the
  title page, which we can't finalise without the official template.

## File map

| File | Chapter | Suggested owner (branch) |
|---|---|---|
| `sections/00_abstract.tex` | Abstract + Keywords | whole team, write last |
| `sections/01_introduction.tex` | Introduction | TBD |
| `sections/01a_problem_statement.tex` | Problem Statement | TBD |
| `sections/02_literature_review.tex` | Literature Review (~ "Background" in COS700 slides) | TBD |
| `sections/03_related_work.tex` | Related Work / Gap | TBD |
| `sections/04_methodology.tex` | Methodology (overview + dataset + fidelity fix) | whole team |
| `sections/04a_sam_regularisation.tex` | -- SAM | SAM branch |
| `sections/04b_rule_extraction_ga_nsga2.tex` | -- Rule extraction (GA/NSGA-II) | rule extraction branch |
| `sections/04c_promethee_ranking.tex` | -- PROMETHEE II | PROMETHEE branch |
| `sections/04d_deepccds_architecture.tex` | -- DeepCCDS architecture | DeepCCDS/data branch |
| `sections/04e_experimental_setup.tex` | -- Experimental setup | whole team, once arms are stable |
| `sections/05_results.tex` | Results | whole team, once experiments are done |
| `sections/06_discussion.tex` | Discussion | TBD |
| `sections/07_conclusion.tex` | Conclusion and Future Work | TBD, write last |
| `references.bib` | Bibliography | whole team |

Update the "Suggested owner" column against `CONTRACTS.md` once
branches are confirmed -- these are placeholders, not assignments.

## Standing rules (same as the proposal)

- `\cite{}` for all citations, never prose author-date.
- `\label{}` / `\ref{}` (or `\cref{}`) for all cross-references.
- British/South African English spelling.
- No em-dashes.
- No first person.
- Real paragraph breaks, not standalone `\\`.
- Gloss all domain-specific terms -- assume the reader knows nothing
  about the topic (Prof. Pillay's instruction).
- Technical specifications (library versions, hardware) stay out of
  the report; they belong in the dissertation only.
- Where NICOG and COS700 requirements conflict, NICOG takes priority.

Every section file currently contains headings and `% TODO` comments
only -- no drafted prose yet, so nobody's section gets overwritten by
someone else's placeholder text.
