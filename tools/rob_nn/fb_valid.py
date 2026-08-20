"""Baseline-flow trustworthiness, and the rotation labels, per record pixel.

Reads what fb_check.cpp measured -- an INDEPENDENT forward-backward consistency
test of the baseline flow, plus how much displacement a fitted global rotation
accounts for -- and projects both onto the crops rob_real.cpp wrote.

Why this exists
---------------
The harm label is |comp(p + flow_corrupt) - comp(p + flow_base)|, which is
exact as a measure of what the corruption did, and is a measure of MERGE HARM
only if flow_base is the true correspondence. The generator certifies flow_base
photometrically, with d^2/sigma^2 -- the analytic mask's own statistic -- so it
passes exactly the failure this model exists for: a tile matched onto lookalike
content, where the residual really is small and the correspondence really is
wrong. Certifying the baseline with that alone is circular in the one place it
matters, and worse than the known moving-object gap because it is silent.

Forward-backward consistency fails differently: a tile that locked onto
lookalike content generally does not reproduce the inverse offset when the
search runs the other way, because the lookalike patch has its own
neighbourhood and its own best match.

THE THRESHOLD IS RELATIVE. A fixed pixel budget is the wrong test when the
flow is 100 px and the field is genuinely rough with parallax, so a tile passes
when

    |f_fwd + f_bwd|  <  FB_ABS + FB_REL * |f_fwd|

Coordinates: fb_check writes one entry per RAW tile of `ts` raw pixels; a
record pixel (iy, ix) is guide pixel (cy + iy, cx + ix), which is raw pixel
2*(cy + iy), so the tile index is 2*(cy + iy) // ts. Same convention as
build_robustness_nn_features; see tools/rob_nn/README.md for the table.
"""
import os
import numpy as np

FB_ABS = float(os.environ.get("ROB_FB_ABS", 1.0))    # raw px
FB_REL = float(os.environ.get("ROB_FB_REL", 0.05))   # fraction of |flow|
# "This tile carries strong legitimate rotation": the fitted global rotation
# alone displaces it by more than this, in raw px. Measured, never inferred
# from flow magnitude or flow spread, both of which parallax also produces.
ROT_STRONG = float(os.environ.get("ROB_ROT_STRONG", 1.0))
ROT_NONE = float(os.environ.get("ROB_ROT_NONE", 0.25))


def load_fb(fbdir):
    """{(burst, ref, comp): dict(fb, rot, mag, tny, tnx, ts, theta)}"""
    out = {}
    if not fbdir or not os.path.isdir(fbdir):
        return out
    for fn in os.listdir(fbdir):
        if not fn.endswith(".dims"):
            continue
        stem = fn[:-5]
        try:
            b, r, c = stem.split("_")[1:4]
            key = (int(b[1:]), int(r[1:]), int(c[1:]))
        except (ValueError, IndexError):
            continue
        d = {}
        with open(os.path.join(fbdir, fn)) as f:
            for line in f:
                k, v = line.split()
                d[k] = float(v)
        tny, tnx, ts = int(d["tny"]), int(d["tnx"]), int(d["ts"])
        blob = os.path.join(fbdir, stem + ".fb")
        if not os.path.exists(blob):
            continue
        a = np.fromfile(blob, dtype=np.float32)
        nch = a.size // (tny * tnx)
        a = a.reshape(tny, tnx, nch)
        out[key] = dict(fb=a[..., 0], rot=a[..., 1],
                        mag=a[..., 2] if nch > 2 else None,
                        tny=tny, tnx=tnx, ts=ts, theta=d.get("theta", 0.0))
    return out


def record_tile_maps(fb, rec, oh, ow, flow_mag=None):
    """Per-pixel (fb_ok, rot_px) for one record, or (None, None) if unmeasured.

    flow_mag, when given, is the record's own per-pixel |flow| in raw px (from
    feature channels 9-10) and makes the FB threshold relative to it. Without
    it the absolute part alone is used, which is stricter.
    """
    key = (rec["burst"], rec["ref"], rec["comp"])
    e = fb.get(key)
    if e is None:
        return None, None
    ts, tny, tnx = e["ts"], e["tny"], e["tnx"]
    iy = np.arange(oh)[:, None] + rec["cy"]
    ix = np.arange(ow)[None, :] + rec["cx"]
    ty = np.clip((2 * iy) // ts, 0, tny - 1)
    tx = np.clip((2 * ix) // ts, 0, tnx - 1)
    fbe = e["fb"][ty, tx]
    rot = e["rot"][ty, tx]
    lim = FB_ABS + FB_REL * (flow_mag if flow_mag is not None else 0.0)
    return (fbe < lim), rot


def summarise(fb):
    """What the test says about the bursts as a whole, printed once.

    Read the headline with the picture in mind. On these bursts the tiles it
    rejects are the SKY, in one contiguous region (tools/rob_nn saved the map;
    bad tiles have 3.8 of 4 bad neighbours against 2.05 for a random spread),
    and they are rejected because a textureless region has no determined
    correspondence in either direction -- the test being right, not broken.

    Measured on the training split, the gate removes 44.0% of trainable pixels
    but only 3.8% of the quadrant this model exists for (R_normal high,
    R_ideal low) and 6.0% of rotational-AND-textured pixels. So the
    supervision is thin where nothing can be known and nothing is at stake, and
    essentially intact where the model has to be strong. That is the answer to
    "is the baseline-flow assumption hiding a hole in the training set" -- no,
    and here is the number.
    """
    if not fb:
        return "no forward-backward measurements found"
    tot = rej = rot_tot = rot_rej = 0
    for e in fb.values():
        ok = e["fb"] < FB_ABS
        tot += ok.size; rej += int((~ok).sum())
        r = e["rot"] > ROT_STRONG
        rot_tot += int(r.sum()); rot_rej += int((r & ~ok).sum())
    return (f"forward-backward validation over {len(fb)} frame pairs:\n"
            f"  tiles failing at |fb| >= {FB_ABS} px: {100.0*rej/max(tot,1):.1f}%\n"
            f"  strongly rotational tiles: {100.0*rot_tot/max(tot,1):.1f}% of all, "
            f"of which {100.0*rot_rej/max(rot_tot,1):.1f}% fail")
