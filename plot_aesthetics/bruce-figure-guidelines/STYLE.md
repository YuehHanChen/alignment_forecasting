# Bruce's Figure Cheatsheet

> One-page rules. If a rule isn't here, check `style.py`. Don't invent new
> conventions; ask first.

## Naming framework

Two axes:

1. **Width** — `half_page` (3.25") or `full_page` (6.75"). Set by paper
   format, not by your data.
2. **Subplot grid** — number of subplot columns (and rows, if multiple).

| Template | Width | Grid | When |
|---|---|---|---|
| `half_page_1col` | 3.25" | 1×1 | Smallest panel — fits in one paper column |
| `full_page_1col` | 6.75" | 1×1 | Single panel that needs horizontal real estate |
| `full_page_2col_shared_y` | 6.75" | 1×2 | Two panels with the same metric (shared y-axis) |
| `full_page_2col_indiv_y` | 6.75" | 1×2 | Two panels with different metrics |
| `full_page_3col_shared_y` | 6.75" | 1×3 | All panels show the same metric (shared y-axis) |
| `full_page_3col_indiv_y` | 6.75" | 1×3 | Each panel has its own metric (own y-axis) |
| `full_page_3col_2row` (Type A) | 6.75" | 2×3 | Uniform 2-row grid |
| `full_page_1col_3col` (Type B) | 6.75" | 1+3 mixed | Wide summary on top, breakdown below |

For every multi-panel layout, **provide both `_shared_y` and `_indiv_y`
variants** — never mix the two in one template. Pick the one that
matches your data.

For a new shape, create `full_page_<col-spec>.py` following the same
structure. Don't change widths casually — they match LaTeX `\textwidth`
and `\columnwidth` for NeurIPS / ICML / ACL.

## Figures DO NOT have suptitles

Figure titles belong in the LaTeX caption (`\caption{...}`), not on the
figure itself. Don't pass `suptitle=...` to `apply_layout()`. The legend
is the topmost element of the figure.

(`apply_layout` still accepts `suptitle=...` if you ever truly need one,
but the convention is no suptitle.)

## Spacing — one source of truth

All vertical and horizontal gaps come from `style.py` constants. Every
template uses the same numbers, so figures form a visual family:

| Constant | Inches | What it controls |
|---|---|---|
| `LEGEND_FROM_TOP` | 0.20 | Top of legend below figure top |
| `ROW1_FROM_TOP` | 0.70 | First row top — leaves ~0.25" after legend |
| `ROW_TO_ROW` | 0.70 | Vertical gap between panel rows |
| `COL_TO_COL_SHARED` | 0.32 | Horizontal gap when y-axis is shared |
| `COL_TO_COL_INDIV` | 0.75 | Horizontal gap when each panel has own y-label + ticks |
| `BOTTOM_PAD` | 0.60 | Below bottom row for x-label |

(The `*_SUP` and `SUPTITLE_FROM_TOP` constants exist for the rare case a
suptitle is needed; they're not used by any standard template.)

**Don't override these per-template.** If they're wrong, fix them once
in `style.py` and re-render every template.

## How a template uses the framework

Every template ends with:

```python
FIG_W, FIG_H = figsize_for(WIDTH, n_rows=N_ROWS, panel_h=PANEL_H)

# ... build axes, plot, etc ...

# share_y=True (default) when all panels share the same metric
# share_y=False when each panel has its own y-label + tick numbers
apply_layout(fig, FIG_H, n_rows=N_ROWS, n_cols=N_COLS, panel_h=PANEL_H,
             share_y=True,
             legend_handles=handles)
save_figure(fig, "my_figure")
```

## Shared vs individual y-axis

| Case | What to do | Template |
|---|---|---|
| All panels show the same metric | `share_y=True` (default). Set `ylabel` only on leftmost panel; suppress tick labels on others with `tick_params(labelleft=False)`. | `full_page_3col_shared_y.py` |
| Each panel has a different metric | `share_y=False`. Set `ylabel` on every panel; the wider `COL_TO_COL_INDIV` gap leaves room for tick numbers. | `full_page_3col_indiv_y.py` |

`figsize_for()` and `apply_layout()` are the two entry points. Don't call
`fig.subplots_adjust(top=...)` or hardcode legend positions yourself.

## Typography (set by `setup_rcparams()`)

| Role | Size | Weight |
|---|---|---|
| Panel title `(a) ...` | 9 | bold, **centered** (matplotlib default), `pad=5` |
| Axis label | 8 | normal |
| Tick label | 8 | normal |
| Legend | 8 | normal |
| Body / default | 9 | normal |

Don't pass `loc="left"` to `set_title()` — titles are centered by default.

Font: Times New Roman (serif). Top + right spines off.

## Colors (`palette()` from `style.py`)

Sampled from `cmcrameri.batlow` (colorblind-safe). Use **role names**, not raw hex:

| Role | Position | Use for |
|---|---|---|
| `baseline` | 0.10 | Baseline / control |
| `control` | 0.30 | Secondary control |
| `primary` | 0.45 | First intervention |
| `secondary` | 0.55 | Second intervention |
| `tertiary` | 0.65 | Third intervention |
| `highlight` | 0.78 | Thing being highlighted |
| `extreme` | 0.88 | Strongest intervention |

Plus utility colors: `GRID`, `REFERENCE`, `POSITIVE`, `NEGATIVE`, `NEUTRAL`.

## Markers + lines

- Each condition gets a different **shape** (not just a different color).
- Every marker has a white outline (`markeredgecolor="white"`,
  `markeredgewidth ≈ 1.0`).
- Default `ms=4.5–5.5`, `lw=1.4`.
- Shape menu: `o` (default), `D`, `s`, `^`, `v`, `p`, `h`, `P`, `X`.

## Bars

```python
ax.bar(xs, values, width=BAR_WIDTH/n_cond, color=COLORS[k],
       edgecolor="white", linewidth=0.8,
       yerr=err, error_kw=dict(elinewidth=0.7, ecolor="black", capsize=1.5))
```

`BAR_WIDTH = 0.72`. Bars use `Patch` handles in the legend, lines use
`Line2D` handles. If a figure mixes both, use `Patch` for the legend
(colors carry the meaning).

## "Better" arrow

```python
better_arrow(ax, direction="up", corner="lower right")
```

**Arrow size is FIXED.** Don't shrink it to avoid overlap — solve overlap
by **moving** the arrow to a different corner. Panel titles are centered,
so upper-left and upper-right corners are both free.

First-guess heuristics by plot shape:

| Plot shape | Recommended corner | Direction |
|---|---|---|
| Rising lines that reach the top | lower right | up |
| Rising lines that stay below ~0.7 | upper right | up |
| Falling lines that end near the bottom | upper right | down |
| Falling lines that stay above ~0.3 | lower right | down |
| Bars taller on the right, max < ~0.7 | upper right | up or down per metric |
| Bars taller on the right, max > ~0.7 | lower left | up or down per metric |

Render the PNG, look at it, and switch the corner if it touches data.
Required before the figure is done.

## Layout rules (built into `apply_layout`)

- Inch-anchored top spacing: legend, suptitle, and first row sit at
  fixed inch offsets, NOT figure fractions. Spacing stays consistent
  when `FIG_H` changes.
- Panel titles: `"(a) Description"`, `loc="left"`, `fontweight="bold"`,
  `pad=4`.
- Suptitle: centered, bold, font 11 (one above axis labels).
- Shared y-axis: tick labels on leftmost panel only.
- Figure-level legend (one row above panels), `frameon=True,
  fancybox=True, framealpha=0.95`.

## Saving

Always 600 DPI PNG + vector PDF (handled by `save_figure`). Don't call
`plt.show()` in scripts.

## Anti-patterns

- Per-template legend / suptitle / row-gap inches — use the constants in
  `style.py`.
- Default seaborn / matplotlib `tab10` colors — use `palette()`.
- `tight_layout()` with figure-level legends — use `apply_layout()`.
- Figure fractions for vertical anchors — use inch offsets.
- `argparse` / `sys.argv` for plot configs — constants block only.
- `set_title(..., loc="left")` — titles are centered.
- `better_arrow` placement that overlaps any data line/bar/marker.

## File map

```
bruce-figure-guidelines/
├── STYLE.md                       <- this file
├── style.py                       <- spacing constants + figsize_for + apply_layout
├── templates/
│   ├── half_page_1col.py             <- 3.25", 1 panel
│   ├── full_page_1col.py             <- 6.75", 1 wide panel
│   ├── full_page_2col_shared_y.py    <- 6.75", 1×2, one y-axis
│   ├── full_page_2col_indiv_y.py     <- 6.75", 1×2, per-panel y-axes
│   ├── full_page_3col_shared_y.py    <- 6.75", 1×3, one y-axis
│   ├── full_page_3col_indiv_y.py     <- 6.75", 1×3, per-panel y-axes
│   ├── full_page_3col_2row.py        <- 6.75", 2×3 (Type A — uniform)
│   └── full_page_1col_3col.py        <- 6.75", 1+3 (Type B — mixed)
└── figures/                          <- rendered PNG + PDF
```
