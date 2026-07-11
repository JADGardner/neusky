"""Outside-in visibility explainer with baked-in labels (thesis fig:implicit_visibility).

The hand-drawn diagram is stored as SVG so its circle, arrows, brackets and
markers remain vector graphics. This script appends the former TikZ labels as
SVG text, then emits
``publication/figures/implicit_visibility_explainer_labeled.{svg,png,pdf}``.
Only the rabbit rendering embedded in the source SVG is rasterised.
Deterministic, CPU-only.

The hand-drawn base lives in ``scripts/figures/assets/`` (see its README).

Label geometry replicates the compiled thesis exactly (verified against
``latex/build/thesis.pdf`` p.81): the TikZ node coordinates are cm from the
image centre (y up, 1 cm = 28.3465 bp) and the image is shown at
``0.5\\textwidth`` = 217.92 bp; ``\\tiny`` in the 12pt class is 5.98 bp. The
three multi-line groups (DDF intersection, ground-truth distance, difference)
stay grouped because consecutive lines sit 0.2 cm apart, as in the TikZ.

    PYTHONPATH=. python scripts/figures/fig_implicit_visibility.py
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg

from _common import FIGURES_DIR

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
BASE_SVG = ASSETS_DIR / "implicit_visibility_explainer.svg"

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
SODIPODI_NS = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)
ET.register_namespace("sodipodi", SODIPODI_NS)

CM_TO_BP = 28.3465          # TikZ default unit -> PDF big points
TEXTWIDTH_BP = 435.84       # thesis \textwidth, measured in latex/build/thesis.pdf
DISPLAY_WIDTH_BP = 0.5 * TEXTWIDTH_BP  # \includegraphics[width=0.5\textwidth]

TINY = 5.98                 # \tiny at the 12pt thesis class, in bp

# (x_cm, y_cm, text) from the thesis TikZ; all \tiny, black, unrotated.
LABELS = (
    # DDF intersection group
    (-2.3, 3.6, "DDF Intersection"),
    (-2.3, 3.4, "/ Query Point"),
    (-2.3, 3.2, "and Direction"),
    (0.9, 3.2, "DDF Predicted Depth"),
    (1.3, 0.6, "Query Point"),
    (1.85, 0.4, "and Direction"),
    # Ground-truth distance group
    (-2.6, -0.2, "Ground"),
    (-2.6, -0.4, "Truth"),
    (-2.6, -0.6, "Distance"),
    # Difference group
    (2.0, 2.1, "Difference"),
    (2.0, 2.3, "="),
    (2.0, 2.5, "GT - Predicted"),
    (0.0, -2.6, "Visibility = Difference < Threshold"),
)


def compose_svg(base_svg: Path, add_labels: bool) -> tuple[bytes, float, float]:
    """Return a physically sized SVG with the thesis labels on its top layer."""
    tree = ET.parse(base_svg)
    root = tree.getroot()
    view_box = root.get("viewBox")
    if view_box is None:
        raise ValueError(f"SVG has no viewBox: {base_svg}")

    x0, y0, width, height = [float(v) for v in view_box.replace(",", " ").split()]
    width_bp = DISPLAY_WIDTH_BP
    height_bp = width_bp * height / width

    # CairoSVG otherwise interprets unitless width/height as CSS pixels. Give
    # the composed asset its exact thesis display size so the former TikZ font
    # size and centimetre offsets remain unchanged.
    root.set("width", f"{width_bp / 72.0:.8f}in")
    root.set("height", f"{height_bp / 72.0:.8f}in")

    if add_labels:
        label_group = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id": "thesis-labels"})
        svg_units_per_bp = width / width_bp
        font_size = TINY * svg_units_per_bp
        for x_cm, y_cm, label in LABELS:
            x = x0 + width / 2.0 + x_cm * CM_TO_BP * svg_units_per_bp
            y = y0 + height / 2.0 - y_cm * CM_TO_BP * svg_units_per_bp
            element = ET.SubElement(
                label_group,
                f"{{{SVG_NS}}}text",
                {
                    "x": f"{x:.6f}",
                    "y": f"{y:.6f}",
                    "text-anchor": "middle",
                    "dominant-baseline": "middle",
                    "font-family": "Nimbus Roman, Times, Liberation Serif, serif",
                    "font-size": f"{font_size:.6f}",
                    "fill": "#000000",
                },
            )
            element.text = label

    return ET.tostring(root, encoding="utf-8", xml_declaration=True), width_bp, height_bp


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Bake the thesis TikZ labels into the figure "
                             "(default on; --no-labels exports the bare base)")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR,
                        help="Output directory (default publication/figures/)")
    parser.add_argument("--base", type=Path, default=BASE_SVG,
                        help="Base diagram SVG (default scripts/figures/assets/"
                             "implicit_visibility_explainer.svg)")
    parser.add_argument("--dpi", type=float, default=200.0,
                        help="PNG output resolution (default 200, ~ the base "
                             "PNG's native 599 px at thesis display width)")
    args = parser.parse_args()

    stem = args.output_dir / (
        "implicit_visibility_explainer_labeled" if args.labels
        else "implicit_visibility_explainer")
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg_bytes, width_bp, height_bp = compose_svg(args.base, add_labels=args.labels)
    Path(f"{stem}.svg").write_bytes(svg_bytes)

    cairosvg.svg2pdf(bytestring=svg_bytes, write_to=f"{stem}.pdf")
    cairosvg.svg2png(
        bytestring=svg_bytes,
        write_to=f"{stem}.png",
        output_width=round(width_bp * args.dpi / 72.0),
        output_height=round(height_bp * args.dpi / 72.0),
    )
    print(f"[saved] {stem}.svg / .png / .pdf")


if __name__ == "__main__":
    main()
