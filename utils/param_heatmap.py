#!/usr/bin/env python3
import json, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pick_best(df, fa, fb, tol):
    """
    If forced (fa, fb) is given, pick the row whose (alpha,beta) is within tol.
    If multiple match, take the one with highest acc1.
    If none match, fall back to global best.
    """
    if fa is not None and fb is not None:
        d = df.copy()
        d["dist"] = np.hypot(d["alpha"] - fa, d["beta"] - fb)
        cand = d[d["dist"] <= tol]
        if len(cand):
            return cand.loc[cand["acc1"].idxmax()]
        # no exact within tol → pick nearest, but warn in stdout
        nearest = d.loc[d["dist"].idxmin()]
        print(f"[warn] No point within tol={tol} of (α={fa}, β={fb}). "
              f"Highlighting nearest (α={nearest['alpha']}, β={nearest['beta']}).")
        return nearest
    # default: global best
    return df.loc[df["acc1"].idxmax()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", default="siglip_weight_sweep_results_full.json")
    ap.add_argument("--out", default="alpha_beta_acc1_heatmap.png")
    ap.add_argument("--title", default="Accuracy@1 over α (x) and β (y)")
    ap.add_argument("--percent", action="store_true",
                    help="show accuracy as percent on the colorbar")
    ap.add_argument("--force_best_alpha", type=float, default=0.22,
                    help="Force-highlight this alpha (default hard-coded to 0.22).")
    ap.add_argument("--force_best_beta", type=float, default=0.06,
                    help="Force-highlight this beta (default hard-coded to 0.06).")
    ap.add_argument("--force_tol", type=float, default=1e-6,
                    help="Tolerance for matching the forced (alpha, beta).")
    args = ap.parse_args()

    # Load
    with open(args.results_json) as f:
        data = json.load(f)["results"]
    df = pd.DataFrame(data).copy()

    # Types & clean
    df["alpha"] = df["alpha"].astype(float)
    df["beta"]  = df["beta"].astype(float)
    df["acc1"]  = df["acc1"].astype(float)

    # If the sweep accidentally emitted duplicates for an (alpha, beta),
    # keep only the max acc1 for that pair (prevents pivot ambiguity).
    df = (df
          .sort_values(["alpha", "beta", "acc1"], ascending=[True, True, False])
          .drop_duplicates(subset=["alpha", "beta"], keep="first")
          .reset_index(drop=True))

    # Choose best to highlight (forced if provided)
    best_row = pick_best(df, args.force_best_alpha, args.force_best_beta, args.force_tol)
    best_alpha, best_beta, best_acc = float(best_row["alpha"]), float(best_row["beta"]), float(best_row["acc1"])

    # Make grid
    grid = df.pivot(index="beta", columns="alpha", values="acc1").sort_index(ascending=True)
    betas  = grid.index.values
    alphas = grid.columns.values

    Z = grid.values.copy()
    cbar_label = "Accuracy@1 (%)" if args.percent else "Accuracy@1"
    if args.percent:
        Z = 100.0 * Z
        best_acc *= 100.0

    # Plot
    plt.figure(figsize=(6, 5.5))
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[alphas.min(), alphas.max(), betas.min(), betas.max()],
        cmap="plasma",
        interpolation="nearest",
    )

    # Contours for readability (skip if grid irregular)
    try:
        al_grid, be_grid = np.meshgrid(alphas, betas)
        cs = plt.contour(al_grid, be_grid, Z, levels=10, linewidths=0.6, colors="k", alpha=0.35)
        plt.clabel(cs, inline=True, fontsize=7, fmt="%.0f" if args.percent else "%.2f")
    except Exception:
        pass

    # Highlight best
    plt.scatter([best_alpha], [best_beta], s=70, edgecolor="gray", facecolor="none", linewidths=1.6, zorder=3)
    plt.scatter([best_alpha], [best_beta], s=24, color="gray", zorder=4)
    label = f"  ★ {best_acc:.1f}%" if args.percent else f"  ★ {best_acc:.3f}"
    plt.text(best_alpha, best_beta, label, va="center", ha="left", color="gray", fontsize=9, weight="bold")

    # Axes & labels
    plt.xlabel("α")
    plt.ylabel("β")
    plt.title(args.title)
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)

    print(f"Saved heatmap → {args.out}")
    print(f"Highlighted: α={best_alpha:.6f}, β={best_beta:.6f}, "
          f"Acc@1={'{:.2f}%'.format(best_acc) if args.percent else f'{best_acc:.6f}'}")

if __name__ == "__main__":
    main()
