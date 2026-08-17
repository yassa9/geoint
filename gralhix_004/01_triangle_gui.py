#!/usr/bin/env python3
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
TOLERANCE = 0.20
LABELS = ["P0", "P1", "P2"]

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default=str(HERE), help="directory to save triangle json/png into")
args = parser.parse_args()
OUT_DIR = Path(args.out_dir)

clicked = []

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_title("click P0, then P1, then P2")


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_at(v, a, b):
    va = (a[0] - v[0], a[1] - v[1])
    vb = (b[0] - v[0], b[1] - v[1])
    dot = va[0] * vb[0] + va[1] * vb[1]
    n = math.hypot(*va) * math.hypot(*vb)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / n))))


def finish():
    p0, p1, p2 = clicked
    d01, d02, d12 = dist(p0, p1), dist(p0, p2), dist(p1, p2)
    ang0 = angle_at(p0, p1, p2)
    ang1 = angle_at(p1, p0, p2)
    ang2 = angle_at(p2, p0, p1)
    ratio = d01 / d02

    result = {
        "P0": list(p0), "P1": list(p1), "P2": list(p2),
        "side_lengths": {"P0P1": round(d01, 4), "P0P2": round(d02, 4), "P1P2": round(d12, 4)},
        "side_ratios_from_P0": {"P0P1": 1.0, "P0P2": round(d02 / d01, 4), "P1P2": round(d12 / d01, 4)},
        "angles_deg": {"P0": round(ang0, 2), "P1": round(ang1, 2), "P2": round(ang2, 2)},
        "dist_ratio_P0P1_over_P0P2": round(ratio, 4),
        "tolerance_pct": TOLERANCE,
        "angle_at_P0_bounds_deg": [round(ang0 * (1 - TOLERANCE), 2), round(ang0 * (1 + TOLERANCE), 2)],
        "dist_ratio_bounds": [round(ratio * (1 - TOLERANCE), 4), round(ratio * (1 + TOLERANCE), 4)],
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = OUT_DIR / f"triangle_{stamp}.json"
    img_path = OUT_DIR / f"triangle_{stamp}.png"
    out_json.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print(f"side lengths: P0P1={d01:.3f}  P0P2={d02:.3f}  P1P2={d12:.3f}")
    print(f"angles: P0={ang0:.2f}deg  P1={ang1:.2f}deg  P2={ang2:.2f}deg")
    print(f"dist ratio P0->P1 : P0->P2 = {ratio:.3f} : 1")
    print(f"+/-{TOLERANCE*100:.0f}% bounds -> angle_at_P0: {result['angle_at_P0_bounds_deg']}  "
          f"dist_ratio: {result['dist_ratio_bounds']}")
    print(f"saved -> {out_json}")
    print(f"saved -> {img_path}")
    print("=" * 60)

    ax.set_title(f"done. angle@P0={ang0:.1f} deg, ratio={ratio:.2f}")
    fig.canvas.draw()
    fig.savefig(img_path, dpi=150)


def onclick(event):
    if event.xdata is None or event.ydata is None or len(clicked) >= 3:
        return
    clicked.append((event.xdata, event.ydata))
    ax.plot(event.xdata, event.ydata, "o", color=["red", "cyan", "magenta"][len(clicked) - 1], markersize=10)
    ax.text(event.xdata + 0.15, event.ydata + 0.15, LABELS[len(clicked) - 1], fontweight="bold")
    if len(clicked) > 1:
        (x0, y0), (x1, y1) = clicked[-2], clicked[-1]
        ax.plot([x0, x1], [y0, y1], "w-", alpha=0.6)
    fig.canvas.draw()

    if len(clicked) == 3:
        x0, y0 = clicked[0]
        x2, y2 = clicked[2]
        ax.plot([x2, x0], [y2, y0], "w-", alpha=0.6)
        fig.canvas.draw()
        finish()


fig.canvas.mpl_connect("button_press_event", onclick)
print("click order: P0, P1, P2")
plt.show()
