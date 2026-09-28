"""Shared look for experiment 4's figures and tables: the Aptos Display
font, the theme, display names, and the condition colors -- so every exp4
figure (scripts/plot_exp4_results.py, scripts/plot_exp4_ladder.py) reads as
one set.

Font. Aptos Display is a Microsoft font. Colab's Linux image doesn't ship
it, and this repo is public, so the font files are never committed: put the
Aptos Display .ttf files in a folder (e.g. on Google Drive) and pass it as
--font-dir. `use_aptos_display` refuses to fall back to another font
silently -- a figure in the wrong font is an error, not a warning. The files
name themselves typographic family "Aptos", subfamily "Display", so
matplotlib registers them as family "Aptos". We check each file really is
Aptos Display (not the text cut) before registering it.

Colors (validated with the dataviz skill's palette validator, light mode,
all pairs -- the curve panels show every adaptation condition at once):
gradient ascent orange, LoRA decoder blue, LoRA encoder aqua, LoRA
encoder + decoder violet, and the no-adaptation baseline neutral gray.
Aqua is below 3:1 contrast on the surface, so it always ships with a
legend and a table (the relief rule).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager

FONT_FAMILY_NAME = "Aptos Display"

# Chart surface and ink (the dataviz reference palette's light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

CONDITION_ORDER = [
    "mean",
    "gradient_ascent",
    "lora_ascent_decoder",
    "lora_ascent_encoder",
    "lora_ascent_encoder_decoder",
]
ADAPTATION_CONDITIONS = CONDITION_ORDER[1:]
CONDITION_LABELS = {
    "mean": "Mean (no adaptation)",
    "gradient_ascent": "Gradient ascent (latent)",
    "lora_ascent_decoder": "LoRA (decoder)",
    "lora_ascent_encoder": "LoRA (encoder)",
    "lora_ascent_encoder_decoder": "LoRA (encoder + decoder)",
}
CONDITION_COLORS = {
    "mean": MUTED,
    "gradient_ascent": "#eb6834",
    "lora_ascent_decoder": "#2a78d6",
    "lora_ascent_encoder": "#1baf7a",
    "lora_ascent_encoder_decoder": "#4a3aa7",
}

# The benchmark ladder, easiest first: (code, output subdir, display name).
LEVELS = [
    ("L0", "exp4_lpn_pattern2d", "2×2 patterns"),
    ("L1", "exp4_ladder_L1_sparse2x2", "Sparse 2×2 patterns"),
    ("L2", "exp4_ladder_L2_pattern3x3", "3×3 patterns"),
    ("L3", "exp4_ladder_L3_pattern3x3_corner", "3×3 patterns, corner anchor"),
    ("L4", "exp4_ladder_L4_color_permutation", "Color permutation"),
]
LEVEL_NAMES = {code: name for code, _, name in LEVELS}


def _is_aptos_display(path: Path) -> bool:
    from fontTools.ttLib import TTFont

    names = TTFont(str(path), lazy=True)["name"]
    return (names.getDebugName(1) or "").startswith(FONT_FAMILY_NAME)


def use_aptos_display(font_dir: str | Path | None, required: bool = True) -> str | None:
    """Registers the Aptos Display .ttf files in `font_dir` and makes them
    matplotlib's font. Returns the family name matplotlib uses. With
    `required` (the default), raises if they're missing rather than falling
    back to another font; `required=False` is for tests only.
    """
    help_text = (
        "Aptos Display font files are needed for exp4 figures. Put the Aptos Display .ttf files "
        "(Regular and Bold at least) in a folder -- e.g. on Google Drive -- and pass it as --font-dir. "
        "On Windows with Microsoft 365 they're under "
        "%LOCALAPPDATA%\\Microsoft\\FontCache\\4\\CloudFonts\\Aptos Display\\; they can also be "
        "downloaded from Microsoft."
    )
    folder = Path(font_dir) if font_dir else None
    if folder is None or not folder.is_dir():
        if not required:
            return None
        where = f"{str(font_dir)!r} (resolved to {folder.resolve()})" if folder else "(no --font-dir given)"
        hint = ""
        if folder and not folder.is_absolute() and str(font_dir).startswith("MyDrive"):
            hint = " On Colab, Drive paths start with /content/drive/ -- e.g. /content/drive/MyDrive/..."
        raise FileNotFoundError(f"Font folder not found: {where}.{hint} {help_text}")
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".ttf")
    display = [p for p in paths if _is_aptos_display(p)]
    if not display:
        if required:
            found = ", ".join(p.name for p in paths) or "no .ttf files"
            raise FileNotFoundError(f"No Aptos Display fonts in {folder} (found: {found}). {help_text}")
        return None
    for path in display:
        font_manager.fontManager.addfont(str(path))
    family = font_manager.FontProperties(fname=str(display[0])).get_name()
    matplotlib.rcParams["font.family"] = family
    return family


def apply_theme() -> None:
    """Recessive chrome, hairline grid, and embedded (TrueType) fonts in PDFs."""
    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def setup(font_dir: str | Path | None, required: bool = True) -> None:
    """The one call every exp4 plotting script makes first."""
    use_aptos_display(font_dir, required=required)
    apply_theme()


def percent_axis(ax, axis: str = "y") -> None:
    formatter = matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def save_figure(fig, out_dir: Path, stem: str) -> list[Path]:
    """Writes `stem`.png (300 dpi, for slides and docs) and `stem`.pdf
    (vector, fonts embedded, for papers).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def render_table_figure(rows: list[list[str]], header: list[str], title: str, subtitle: str | None, numeric: list[bool]):
    """A booktabs-style table as a figure (rules above and below the header
    and at the bottom, no vertical lines): text columns left-aligned, numbers
    right-aligned. For slides and documents; the CSV/Markdown/LaTeX files
    carry the same values for anything that needs to be edited or re-typeset.
    """
    char_width, row_height = 0.074, 0.3  # inches per character / per row
    widths = [max(len(str(v)) for v in [h, *(r[i] for r in rows)]) for i, h in enumerate(header)]
    col_widths = [w * char_width + 0.35 for w in widths]
    total_width = sum(col_widths)
    title_height = 0.75 if subtitle else 0.5
    height = title_height + row_height * (len(rows) + 1) + 0.25
    fig = plt.figure(figsize=(total_width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total_width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.text(0, 0.28, title, fontsize=12, fontweight="bold", color=INK, va="center")
    if subtitle:
        ax.text(0, 0.55, subtitle, fontsize=9.5, color=INK_SECONDARY, va="center")
    top = title_height
    lefts = [sum(col_widths[:i]) for i in range(len(col_widths))]

    def draw_row(values, y, weight="normal", color=INK):
        for i, value in enumerate(values):
            if numeric[i]:
                ax.text(lefts[i] + col_widths[i] - 0.15, y, value, ha="right", va="center", fontsize=9.5,
                        fontweight=weight, color=color)
            else:
                ax.text(lefts[i] + 0.05, y, value, ha="left", va="center", fontsize=9.5, fontweight=weight,
                        color=color)

    ax.plot([0, total_width], [top, top], color=INK, linewidth=1.0)
    draw_row(header, top + row_height / 2, weight="bold", color=INK_SECONDARY)
    ax.plot([0, total_width], [top + row_height, top + row_height], color=INK, linewidth=0.6)
    for r, values in enumerate(rows):
        draw_row(values, top + row_height * (r + 1.5))
    bottom = top + row_height * (len(rows) + 1)
    ax.plot([0, total_width], [bottom, bottom], color=INK, linewidth=1.0)
    return fig
