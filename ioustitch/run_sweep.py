"""
Stitch saved sweeps (one folder per sweep with frame_*.png, optionally in frames/, and coordinates.json holding one
dict per frame: 'ts' and, if available, the network GM position 'gm_x_mm', 'gm_y_mm', 'quality_score' and the online
trajectory 'col0_px').

  # one sweep folder
  python -m ioustitch.run_sweep --sweep "<root>/Patient 1/10L" --out stitched

  # every sweep under a root
  python -m ioustitch.run_sweep --root "<root>" --out stitched

Per sweep writes to <out>/<Patient_xx_TOOTH>/:
  panorama.png              panorama of the proposed trajectory (probe-fixed layer removed)
  panorama_uncertainty.png  panorama + colour strip of the lateral position sd (um)
  compare.png               the same compositor with the proposed trajectory, GM tracking, the online
                            estimate and pairwise chaining
  trajectory.csv            t, p, q, sd_p, sd_q (px and mm), blind flag
  report.json               VCE scales, inlier rates, photometric consistency of every trajectory, run time
"""
import argparse, glob, json, os, re, sys, time
import numpy as np
import cv2

from .config import StitchConfig
from .pipeline import stitch_sweep, register_all
from .composite import composite, to_uint8, uncertainty_strip
from .evaluate import photometric_consistency
from . import baselines as bl


def find_sweeps(root):
    out = []
    for cp in sorted(glob.glob(os.path.join(root, "**", "coordinates.json"), recursive=True)):
        d = os.path.dirname(cp)
        if re.search(r"(^|[\\/])test([\\/]|$)", d, re.I):
            continue
        fr = os.path.join(d, "frames")
        files = sorted(glob.glob(os.path.join(fr, "frame_*.png"))) or \
            sorted(glob.glob(os.path.join(d, "frame_*.png")))
        if files:
            out.append((d, files, cp))
    return out


def sweep_tag(d):
    parts = re.split(r"[\\/]", os.path.normpath(d))
    return re.sub(r"[^0-9A-Za-z]+", "_", "_".join(parts[-2:])).strip("_")


def load_sweep(files, cp):
    coords = json.load(open(cp))
    if isinstance(coords, dict):
        coords = [coords]
    frames = [cv2.imread(f) for f in files]
    n = min(len(frames), len(coords))
    return [f for f in frames[:n]], coords[:n]


def label(img, text):
    out = cv2.cvtColor(to_uint8(img), cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1, cv2.LINE_AA)
    return out


def stack_rows(imgs, gap=4):
    W = max(i.shape[1] for i in imgs)
    rows = []
    for im in imgs:
        pad = np.zeros((im.shape[0], W, 3), np.uint8); pad[:, :im.shape[1]] = im
        rows += [pad, np.full((gap, W, 3), 90, np.uint8)]
    return np.vstack(rows[:-1])


def process(d, files, cp, out_root, cfg):
    tag = sweep_tag(d)
    od = os.path.join(out_root, tag); os.makedirs(od, exist_ok=True)
    frames, coords = load_sweep(files, cp)
    if len(frames) < 8:
        print(f"[skip] {tag}: {len(frames)} frames"); return None
    print(f"--- {tag} ({len(frames)} frames) ---")
    r = stitch_sweep(frames, coords, cfg)
    comp = r["composite"]
    cv2.imwrite(os.path.join(od, "panorama.png"), to_uint8(comp["pano"]))
    strip, um = uncertainty_strip(comp["sd_map"], cfg.px_mm_x)
    pano_bgr = cv2.cvtColor(to_uint8(comp["pano"]), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(od, "panorama_uncertainty.png"), np.vstack([pano_bgr, strip]))

    # ---- same-compositor comparison of trajectories -------------------------
    g, ws = r["frames_gray"], r["tissue"]
    q = np.where(r["blind"], 0.0, np.maximum(r["quality"], 1e-3))
    trajs = {"proposed": (r["p"], r["q"])}
    try:
        trajs["GM tracking"] = bl.gm_sg(coords, cfg.px_mm_x, cfg.px_mm_y)
    except KeyError:
        pass
    if all("col0_px" in c for c in coords):
        trajs["online estimate"] = bl.live_col0(coords, cfg.px_mm_y)
    seq = register_all(r["frames_clean"], ws, cfg, [(i, i + 1) for i in range(len(g) - 1)], r["speckle_area"])
    trajs["pairwise chaining"] = bl.seq_reg(seq, len(g))
    panels, pcs = [], {}
    fs_s = [cv2.GaussianBlur(x, (0, 0), cfg.smooth_sigma) for x in r["frames_clean"]]
    for name, (p, qq) in trajs.items():
        c = composite(r["frames_clean"], p, qq, q, None, ws, cfg)
        pcs[name] = photometric_consistency(fs_s, ws, p, qq)
        panels.append(label(c["pano"], f"{name}   PC={pcs[name]:.3f}   extent={np.ptp(p) * cfg.px_mm_x:.2f} mm"))
    cv2.imwrite(os.path.join(od, "compare.png"), stack_rows(panels))

    # ---- trajectory + report --------------------------------------------------
    with open(os.path.join(od, "trajectory.csv"), "w") as f:
        f.write("frame,t_s,p_px,q_px,sd_p_px,sd_q_px,p_mm,q_mm,sd_p_um,blind\n")
        for i in range(len(g)):
            f.write(f"{i},{r['t'][i]:.4f},{r['p'][i]:.3f},{r['q'][i]:.3f},{r['sd_p'][i]:.3f},"
                    f"{r['sd_q'][i]:.3f},{r['p'][i] * cfg.px_mm_x:.4f},{r['q'][i] * cfg.px_mm_y:.4f},"
                    f"{r['sd_p'][i] * cfg.px_mm_x * 1e3:.1f},{int(r['blind'][i])}\n")
    rep = dict(sweep=d, n_frames=len(g), n_blind=int(r["blind"].sum()),
               extent_mm=float(np.ptp(r["p"]) * cfg.px_mm_x),
               cue_sd={k: float(np.sqrt(v)) for k, v in r["scales"].items()
                       if not k.startswith(("gauge", "motion"))},
               inlier_rate=r["inlier_rate"], n_factors=r["n_factors"],
               median_sd_p_um=float(np.median(r["sd_p"]) * cfg.px_mm_x * 1e3),
               photometric_consistency=pcs, runtime_s=r["runtime_s"],
               px_mm=[cfg.px_mm_x, cfg.px_mm_y])
    json.dump(rep, open(os.path.join(od, "report.json"), "w"), indent=2)
    print(f"   extent {rep['extent_mm']:.2f} mm | PC " +
          "  ".join(f"{k.split(' ')[0]}={v:.3f}" for k, v in pcs.items()))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep"); ap.add_argument("--root")
    ap.add_argument("--only", default="", help="comma list of substrings to match sweep paths")
    ap.add_argument("--exclude", default="", help="comma list of substrings to skip")
    ap.add_argument("--out", default="stitched")
    ap.add_argument("--px-mm-x", type=float, default=None, help="lateral mm/px (CONFIRM)")
    ap.add_argument("--px-mm-y", type=float, default=None, help="axial mm/px (CONFIRM)")
    ap.add_argument("--fast", action="store_true", help="fewer lags, no loop closure")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = StitchConfig()
    if a.px_mm_x: cfg.px_mm_x = a.px_mm_x
    if a.px_mm_y: cfg.px_mm_y = a.px_mm_y
    cfg.verbose = not a.quiet
    if a.fast:
        cfg.lags = (1, 2, 3, 6); cfg.use_loop_closure = False
    if a.sweep:
        d = a.sweep
        files = sorted(glob.glob(os.path.join(d, "frames", "frame_*.png"))) or \
            sorted(glob.glob(os.path.join(d, "frame_*.png")))
        jobs = [(d, files, os.path.join(d, "coordinates.json"))]
    elif a.root:
        jobs = find_sweeps(a.root)
        only = [o.strip().replace("\\", "/") for o in a.only.split(",") if o.strip()]
        if only:
            jobs = [j for j in jobs if any(o in j[0].replace("\\", "/") for o in only)]
        excl = [o.strip() for o in a.exclude.split(",") if o.strip()]
        jobs = [j for j in jobs if not any(re.search(re.escape(o) + r"(\D|$)", j[0]) for o in excl)]
    else:
        sys.exit("give --sweep or --root")
    print(f"[jobs] {len(jobs)} sweep(s)")
    os.makedirs(a.out, exist_ok=True)
    reps = []
    for d, files, cp in jobs:
        try:
            rep = process(d, files, cp, a.out, cfg)
            if rep: reps.append(rep)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[error] {d}: {e}")
    if reps:
        json.dump(reps, open(os.path.join(a.out, "all_reports.json"), "w"), indent=2)
        print(f"\n[save] {len(reps)} sweeps -> {a.out}")


if __name__ == "__main__":
    main()
