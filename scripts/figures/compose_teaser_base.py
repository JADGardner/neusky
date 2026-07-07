"""Compose the final NeuSky teaser: James's teaser_base.svg + the RENI
envmap/heat panels (rendered by fig_teaser.py's panels stage) injected
under the arrows, exported to PDF/PNG.

    PYTHONPATH=. python scripts/figures/compose_teaser_base.py
"""
import argparse
import base64
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIGS = REPO / "publication" / "figures"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels-dir", type=Path, default=FIGS / "teaser_panels")
    ap.add_argument("--output", type=Path, default=FIGS / "teaser_thesis")
    # centred under the DDF grid (centre x 235.9), with the same vertical
    # gap below albedo/normal (bottom 84.7) as DDF-to-albedo (10.3 units)
    ap.add_argument("--x", type=float, default=194.4)
    ap.add_argument("--y", type=float, default=95.0)
    ap.add_argument("--w", type=float, default=40.0)
    ap.add_argument("--gap", type=float, default=3.0)
    args = ap.parse_args()

    svg = (HERE / "assets" / "teaser_base.svg").read_text()

    def uri(name):
        return ("data:image/png;base64," + base64.b64encode(
            (args.panels_dir / name).read_bytes()).decode("ascii"))

    from PIL import Image
    iw, ih = Image.open(args.panels_dir / "envmap.png").size
    h = args.w * ih / iw
    imgs = ""
    for i, name in enumerate(("envmap.png", "envmap_heat.png")):
        x = args.x + i * (args.w + args.gap)
        imgs += (f'<image x="{x:.2f}" y="{args.y:.2f}" width="{args.w:.2f}" '
                 f'height="{h:.2f}" href="{uri(name)}" '
                 'preserveAspectRatio="none"/>')
    # insert before the first top-level <g so the base's arrows draw on top
    k = svg.index("<g")
    svg = svg[:k] + imgs + svg[k:]

    # baked text labels (original teaser wording; thesis serif)
    LABELS = [
        (30.5, 116.5, 5.0, ["Unconstrained Outdoor", "Multi-View Images"]),
        (130.5, 24.5, 3.2, ["Sky at", "Infinity"]),
        (130.5, 39.5, 3.2, ["Scene Geometry", "Bounds"]),
        (122.0, 122.0, 5.0, ["NeuS-Facto Volume"]),
        (141.5, 106.5, 2.9, ["Sky Pixel Illumination Constraint"]),
        (235.9, 52.5, 5.0, ["Differentiable Sky Visibility"]),
        (214.3, 90.5, 5.0, ["Albedo"]),
        (257.7, 90.5, 5.0, ["Normal"]),
        (235.9, 121.5, 5.0, ["HDR Neural Illumination Prior"]),
        (334.0, 66.5, 5.0, ["Relighting"]),
        (334.0, 126.5, 5.0, ["Novel Views"]),
    ]
    texts = ""
    for x, y, size, lines in LABELS:
        for i, line in enumerate(lines):
            texts += (f'<text x="{x}" y="{y + i * size * 1.15:.2f}" '
                      f'font-family="Nimbus Roman, Times New Roman, serif" '
                      f'font-size="{size}" text-anchor="middle" '
                      f'fill="#000">{line}</text>')
    svg = svg[: svg.rindex("</svg>")] + texts + "</svg>"

    out_svg = args.output.with_suffix(".svg")
    out_svg.write_text(svg)
    for fmt in ("pdf", "png"):
        subprocess.run(["rsvg-convert", "-f", fmt, "-b", "white",
                        "--zoom", "3.0" if fmt == "png" else "1.0",
                        "-o", str(args.output.with_suffix("." + fmt)), str(out_svg)],
                       check=True)
    print(f"[saved] {args.output}.svg / .pdf / .png")


if __name__ == "__main__":
    main()
