"""Method overview diagram with baked-in labels (thesis fig:technical_overview).

The thesis overlays label text on ``figures/technical_overview.pdf`` with a
TikZ picture (``latex/9_Chapter3/paper_sections/01_Introduction.tex``). This
script bakes those labels into the asset itself, emitting
``outputs/figures/technical_overview_labeled.{png,pdf}`` by default so the
LaTeX side can become a plain ``\\includegraphics``. Pass
``--output-dir publication/figures`` when promoting a reviewed version.
Deterministic, CPU-only.

The hand-drawn base lives in ``scripts/figures/assets/`` (see its README).
The base PDF is rasterised with pymupdf or ``pdftoppm`` when available,
falling back to the pre-rasterised ``technical_overview_base.png``.

Label geometry replicates the compiled thesis exactly (verified against
``latex/build/thesis.pdf`` p.73): the TikZ node coordinates are cm from the
image centre (y up, 1 cm = 28.3465 bp) and the image is shown at
``\\textwidth`` = 435.84 bp; ``\\tiny`` in the 12pt class is 5.98 bp and
``\\scalebox`` factors multiply the 11.96 bp normal size.

    PYTHONPATH=. python scripts/figures/fig_technical_overview.py
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Nimbus Roman",
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "STIXGeneral",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

import matplotlib.pyplot as plt  # noqa: E402

from _common import FIGURES_DIR  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
BASE_PDF = ASSETS_DIR / "technical_overview.pdf"
BASE_PNG = ASSETS_DIR / "technical_overview_base.png"

CM_TO_BP = 28.3465          # TikZ default unit -> PDF big points
TEXTWIDTH_BP = 435.84       # thesis \textwidth, measured in latex/build/thesis.pdf
DISPLAY_WIDTH_BP = 1.0 * TEXTWIDTH_BP  # \includegraphics[width=1.0\textwidth]

TINY = 5.98                 # \tiny at the 12pt thesis class, in bp
SCALE_06 = 7.17             # \scalebox{0.6} of the 11.96 bp normal size
SCALE_045 = 5.38            # \scalebox{0.45}

DDF_POS_COLOUR = (1.0, 0.0, 1.0)      # \definecolor{ddf_pos_colour}
DDF_DIR_COLOUR = (0.5019, 0.0, 0.0)   # \definecolor{ddf_dir_colour}
BLUE = (0.0, 0.0, 1.0)
BLACK = (0.0, 0.0, 0.0)

# (x_cm, y_cm, text, fontsize_bp, colour, rotation_deg) from the thesis TikZ.
LABELS = (
    (1.1, -2.8, "Position", TINY, DDF_POS_COLOUR, 0.0),
    (-0.1, -2.05, "Direction", SCALE_06, DDF_DIR_COLOUR, 0.0),
    (4.3, -2.05, "Direction", SCALE_06, BLUE, 0.0),
    (-4.75, 2.5, "Sky Pixel Illumination Constraint", TINY, BLACK, 0.0),
    (-4.75, 1.9, "Appearance Loss", TINY, BLACK, 0.0),
    (5.55, -1.3, "HDR Neural Illumination", TINY, BLACK, 0.0),
    (5.55, -1.5, "Prior", TINY, BLACK, 0.0),
    (5.55, -2.8, "Per-Image Illumination Latent", TINY, BLACK, 0.0),
    (1.1, -1.3, "Spherical Directional", TINY, BLACK, 0.0),
    (1.1, -1.5, "Distance Field", TINY, BLACK, 0.0),
    (2.05, -2.05, "DDF", SCALE_045, BLACK, 0.0),
    (2.9, -2.05, "V(DDF)", SCALE_045, BLACK, 0.0),
    (-0.3, 1.45, "Sky-pixel Ray", TINY, BLACK, 29.5),
)


def rasterize_base(pdf_path: Path, fallback_png: Path, zoom: float = 4.0):
    """Rasterise the base PDF, or fall back to the pre-rasterised PNG.

    Tries pymupdf, then pdftoppm; the research container has neither, so the
    committed ``technical_overview_base.png`` (same zoom) is the usual path.
    """
    try:
        import fitz  # pymupdf

        page = fitz.open(str(pdf_path))[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            pix.save(tmp.name)
            print(f"[base] pymupdf raster of {pdf_path.name} at zoom {zoom}")
            return plt.imread(tmp.name)
    except ImportError:
        pass

    if shutil.which("pdftoppm"):
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "base"
            subprocess.run(
                ["pdftoppm", "-png", "-r", str(int(72 * zoom)),
                 str(pdf_path), str(stem)],
                check=True,
            )
            out = sorted(Path(tmp).glob("base*.png"))[0]
            print(f"[base] pdftoppm raster of {pdf_path.name} at {int(72 * zoom)} dpi")
            return plt.imread(out)

    print(f"[base] no PDF rasteriser found; using {fallback_png.name}")
    return plt.imread(fallback_png)


def build_figure(base_image, add_labels: bool):
    """Base image at its thesis display size with the TikZ labels baked in."""
    h_px, w_px = base_image.shape[:2]
    width_bp = DISPLAY_WIDTH_BP
    height_bp = width_bp * h_px / w_px

    # Figure canvas == exactly the image at thesis display size, so a later
    # \includegraphics[width=1.0\textwidth] reproduces the compiled TikZ
    # geometry (labels at their absolute bp sizes) 1:1.
    fig = plt.figure(figsize=(width_bp / 72.0, height_bp / 72.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(base_image, aspect="auto", interpolation="lanczos")
    ax.set_axis_off()

    if add_labels:
        for x_cm, y_cm, text, size, colour, rotation in LABELS:
            fig.text(
                0.5 + x_cm * CM_TO_BP / width_bp,
                0.5 + y_cm * CM_TO_BP / height_bp,
                text,
                ha="center",
                va="center",
                fontsize=size,
                fontfamily="serif",
                color=colour,
                rotation=rotation,
                zorder=20,
            )
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Bake the thesis TikZ labels into the figure "
                             "(default on; --no-labels exports the bare base)")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR,
                        help="Output directory (default outputs/figures/)")
    parser.add_argument("--base", type=Path, default=BASE_PDF,
                        help="Base diagram PDF (default scripts/figures/assets/"
                             "technical_overview.pdf)")
    parser.add_argument("--dpi", type=float, default=600.0,
                        help="PNG output resolution (default 600, ~ the "
                             "pre-rasterised base at thesis display width)")
    args = parser.parse_args()

    base_image = rasterize_base(args.base, BASE_PNG)
    fig = build_figure(base_image, add_labels=args.labels)

    stem = args.output_dir / (
        "technical_overview_labeled" if args.labels else "technical_overview")
    stem.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": the canvas must stay exactly the thesis display
    # size for the cm -> fraction label mapping to hold.
    fig.savefig(f"{stem}.png", dpi=args.dpi)
    fig.savefig(f"{stem}.pdf")
    print(f"[saved] {stem}.png / .pdf")


if __name__ == "__main__":
    main()
