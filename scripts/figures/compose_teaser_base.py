"""Compose the final NeuSky teaser: James's teaser_base.svg + the RENI
envmap/heat panels (rendered by fig_teaser.py's panels stage) injected
under the arrows, exported to PDF/PNG.

    PYTHONPATH=. python scripts/figures/compose_teaser_base.py

This stage is CPU-only and never runs the model: the panels are read from
the cache that fig_teaser.py's GPU stage wrote (publication/figures/
teaser_panels/). Tweak the LAYOUT block below and re-run to move things.
"""
import argparse
import base64
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIGS = REPO / "publication" / "figures"

# ---------------------------------------------------------------------------
# LAYOUT -- edit and re-run (seconds, no GPU). All coordinates are in
# teaser_base.svg user units (its viewBox), origin top-left, y down.
# ---------------------------------------------------------------------------

# Injected RENI envmap + HDR-luminance heatmap, side by side, centred on
# the middle column (DDF grid centre x = 235.5, measured from the base) with
# the same vertical gap below albedo/normal (bottom 84.7) as DDF-to-albedo.
ENVMAP_X = 194.0    # left edge of the envmap panel
ENVMAP_Y = 95.0     # top edge of both panels
ENVMAP_W = 40.0     # width of each panel (height follows the image aspect)
ENVMAP_GAP = 3.0    # horizontal gap between the two panels

# Its caption is derived from the constants above so it stays centred under
# the pair; only the gap below the panels is set here.
HDR_LABEL = "HDR Neural Illumination Prior"
HDR_LABEL_SIZE = 5.0
HDR_LABEL_GAP = 6.5   # baseline distance below the panels

# Baked text labels: (centre_x, first_line_y, font_size, lines).
# Multi-line labels advance by LABEL_LINE_SPACING * font_size per line.
# Positions are centred on component bounding boxes measured from
# teaser_base.svg (2026-07-14): collage 0..74.5; arcs centred x 136 with
# outer apex y 27.5 and inner apex y 42; latent squares 118..169.8; DDF grid
# 215.8..255.3 (bottom 44.5); albedo 194..233; normal 238.3..277.3 (bottoms
# 83.3); relit renders 303..371.3 (top bottom 54.3, lower top 69.0,
# lower bottom 118.5).
LABELS = [
    (37.2, 116.5, 5.0, ["Unconstrained Outdoor", "Multi-View Images"]),
    # Both arch labels hug the centre-top of their arch (apexes y=27.5 and
    # y=42.0, clearance 1.5); NeuS-Facto shares the same centre axis.
    (120.0, 22.3, 3.2, ["Sky at", "Infinity"]),
    (120.0, 36.8, 3.2, ["Scene Geometry", "Bounds"]),
    (120.0, 122.0, 5.0, ["NeuS-Facto Volume"]),
    (143.9, 106.5, 2.9, ["Sky Pixel Illumination Constraint"]),
    (235.5, 52.5, 5.0, ["Differentiable Sky Visibility"]),
    (213.5, 90.5, 5.0, ["Albedo"]),
    (257.8, 90.5, 5.0, ["Normal"]),
    (337.1, 63.5, 5.0, ["Relighting"]),
    (337.1, 126.5, 5.0, ["Novel Views"]),
]
LABEL_FONT = "Nimbus Roman, Times New Roman, serif"
LABEL_LINE_SPACING = 1.15
LABEL_COLOR = "#000"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels-dir", type=Path, default=FIGS / "teaser_panels")
    ap.add_argument("--output", type=Path, default=FIGS / "teaser_thesis")
    # CLI overrides for the LAYOUT constants above (optional).
    ap.add_argument("--x", type=float, default=ENVMAP_X)
    ap.add_argument("--y", type=float, default=ENVMAP_Y)
    ap.add_argument("--w", type=float, default=ENVMAP_W)
    ap.add_argument("--gap", type=float, default=ENVMAP_GAP)
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

    # baked text labels (positions/wording in the LAYOUT block up top); the
    # HDR-prior caption follows the injected panels wherever they are placed
    hdr_cx = args.x + (2 * args.w + args.gap) / 2
    hdr_y = args.y + h + HDR_LABEL_GAP
    all_labels = LABELS + [(hdr_cx, hdr_y, HDR_LABEL_SIZE, [HDR_LABEL])]
    texts = ""
    for x, y, size, lines in all_labels:
        for i, line in enumerate(lines):
            texts += (f'<text x="{x}" y="{y + i * size * LABEL_LINE_SPACING:.2f}" '
                      f'font-family="{LABEL_FONT}" '
                      f'font-size="{size}" text-anchor="middle" '
                      f'fill="{LABEL_COLOR}">{line}</text>')
    svg = svg[: svg.rindex("</svg>")] + texts + "</svg>"

    out_svg = args.output.with_suffix(".svg")
    out_svg.write_text(svg)
    import shutil
    if shutil.which("rsvg-convert"):
        for fmt in ("pdf", "png"):
            subprocess.run(["rsvg-convert", "-f", fmt, "-b", "white",
                            "--zoom", "3.0" if fmt == "png" else "1.0",
                            "-o", str(args.output.with_suffix("." + fmt)), str(out_svg)],
                           check=True)
    else:
        import cairosvg
        cairosvg.svg2pdf(url=str(out_svg), background_color="white",
                         write_to=str(args.output.with_suffix(".pdf")))
        cairosvg.svg2png(url=str(out_svg), background_color="white", scale=3.0,
                         write_to=str(args.output.with_suffix(".png")))
    print(f"[saved] {args.output}.svg / .pdf / .png")


if __name__ == "__main__":
    main()
