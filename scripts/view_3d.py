#!/usr/bin/env python
"""3D reconstruction view of the triangulated ball over the padel court.

Renders the court (10x20m floor + net) with the triangulated ball trajectory
floating in 3D space, coloured by time, from several camera angles. This is the
actual 3D reconstruction — not a 2D overlay or a flat chart.

Example:
    python scripts/view_3d.py --points /tmp/tri_gated.json --out /tmp/court_3d
"""
import sys, os, argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

COURT_W, COURT_L, NET_H = 10.0, 20.0, 0.92  # net ~0.92m at center


def draw_court(ax):
    # floor outline
    fx = [0, COURT_W, COURT_W, 0, 0]
    fy = [0, 0, COURT_L, COURT_L, 0]
    ax.plot(fx, fy, [0]*5, color="black", lw=2)
    # service lines + center
    for y in (3.0, COURT_L-3.0):
        ax.plot([0, COURT_W], [y, y], [0, 0], color="gray", lw=1)
    ax.plot([COURT_W/2, COURT_W/2], [3.0, COURT_L-3.0], [0, 0], color="gray", lw=1)
    # net at mid-court
    ax.plot([0, COURT_W], [COURT_L/2, COURT_L/2], [0, 0], color="navy", lw=1)
    for x in (0, COURT_W):
        ax.plot([x, x], [COURT_L/2, COURT_L/2], [0, NET_H], color="navy", lw=2)
    ax.plot([0, COURT_W], [COURT_L/2, COURT_L/2], [NET_H, NET_H], color="navy", lw=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="/tmp/tri_gated.json")
    ap.add_argument("--out", default="/tmp/court_3d")
    args = ap.parse_args()

    pts = [p for p in json.load(open(args.points))["points"] if p["on_court"]]
    xs = np.array([p["x"] for p in pts]); ys = np.array([p["y"] for p in pts])
    zs = np.array([p["z"] for p in pts]); ts = np.array([p["t"] for p in pts])
    print(f"{len(pts)} on-court 3D points; z range {zs.min():.2f}..{zs.max():.2f}m")

    views = [("angle", 22, -60), ("side", 5, 0), ("top", 80, -90)]
    for name, elev, azim in views:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        draw_court(ax)
        sc = ax.scatter(xs, ys, zs, c=ts, cmap="plasma", s=30)
        # connect consecutive samples close in time (a flight segment)
        for i in range(1, len(pts)):
            if ts[i] - ts[i-1] < 0.25:
                ax.plot(xs[i-1:i+1], ys[i-1:i+1], zs[i-1:i+1], color="orange", lw=1, alpha=0.6)
        ax.set_xlim(0, COURT_W); ax.set_ylim(0, COURT_L); ax.set_zlim(0, 6)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("height (m)")
        ax.set_box_aspect((COURT_W, COURT_L, 6))
        ax.view_init(elev=elev, azim=azim)
        fig.colorbar(sc, ax=ax, label="time (s)", shrink=0.5)
        ax.set_title(f"triangulated ball in 3D — {name} view")
        out = f"{args.out}_{name}.png"
        fig.savefig(out, dpi=95, bbox_inches="tight")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
