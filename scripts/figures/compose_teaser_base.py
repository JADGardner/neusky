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
