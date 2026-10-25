"""The app's own stages on a synthetic burst: build/conf_harness.exe (cpp/conf_harness.cpp, built by build.sh against
the current core/ sources). One persistent server process per Python process."""
import os
import subprocess
import tempfile

import numpy as np

from .config import tool_path

EXE = os.environ.get("MFSR_HARNESS", tool_path("build", "conf_harness.exe"))
FLAG_HOMOG, FLAG_ORACLE, FLAG_PARITY, FLAG_RMAP = 1, 2, 4, 8
_SERVER = None


def _job(pi, po):
    global _SERVER
    if _SERVER is None or _SERVER.poll() is not None:
        _SERVER = subprocess.Popen([EXE, "--server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                   bufsize=1)
    _SERVER.stdin.write(f"{pi} {po}\n")
    _SERVER.stdin.flush()
    line = _SERVER.stdout.readline()
    if not line.startswith("OK 0"):
        try:
            _SERVER.kill()
        finally:
            _SERVER = None
        raise RuntimeError(f"conf_harness failed: {line!r}")


def run(raw, gt, pert, hpert, cfa, scale, alpha, beta, prewhite, flags=0, seed=0, tile=0, tune=None, rmap=None):
    """raw [N,H,W] loader-domain frames; gt / pert [N,H,W,2] raw px (x, y); hpert [N,3,3].
    -> dict(tile, flow [N-1,ny,nx,2], flow_al, H [N-1,9], has_h [N-1], r0 [N-1,H,W], cov_ref [ch,cw,4],
            cov [N-1,ch,cw,4], grey_ref [H,W], grey [N-1,H,W], tune [4], (parity) Aref, Bref, A, B [.,Ho,Wo,3])."""
    N, H, W = raw.shape
    tile_force = int(tile)
    tn = (list(tune) + [0.0] * 6)[:6] if tune is not None else [0.0] * 6     # k_detail k_denoise D_th D_tr k_stretch k_shrink
    with tempfile.TemporaryDirectory() as d:
        pi, po = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
        with open(pi, "wb") as fh:
            np.array([N, H, W, int(seed) & 0x7fffffff, flags, cfa[0][0], cfa[0][1], cfa[1][0], cfa[1][1], tile_force, 0, 0, 0, 0, 0, 0],
                     np.int32).tofile(fh)
            np.array([scale, alpha, beta, prewhite[0], prewhite[1], prewhite[2], 1.0] + tn + [0.0] * 3,
                     np.float32).tofile(fh)
            np.ascontiguousarray(raw, np.float32).tofile(fh)
            np.ascontiguousarray(gt, np.float32).tofile(fh)
            np.ascontiguousarray(pert, np.float32).tofile(fh)
            np.ascontiguousarray(np.asarray(hpert, np.float32).reshape(N, 9)).tofile(fh)
            if flags & FLAG_RMAP:
                np.ascontiguousarray(rmap, np.float32).reshape(N - 1, H, W).tofile(fh)
        _job(pi, po)
        buf = np.fromfile(po, np.uint8)
    pos = 0

    def take(n, dt=np.float32):
        nonlocal pos
        a = buf[pos:pos + 4 * n].view(dt)
        pos += 4 * n
        return a

    tile_, ny, nx, ch, cw = (int(v) for v in take(5, np.int32))
    tune_ = take(4).copy()
    grey_ref = take(H * W).reshape(H, W).copy()
    cov_ref = take(ch * cw * 4).reshape(ch, cw, 4).copy()
    fl, fla, Hs, hh, r0, cov, grey = [], [], [], [], [], [], []
    for _ in range(N - 1):
        fl.append(take(ny * nx * 2).reshape(ny, nx, 2).copy())
        fla.append(take(ny * nx * 2).reshape(ny, nx, 2).copy())
        Hs.append(take(9).copy())
        hh.append(float(take(1)[0]))
        r0.append(take(H * W).reshape(H, W).copy())
        cov.append(take(ch * cw * 4).reshape(ch, cw, 4).copy())
        grey.append(take(H * W).reshape(H, W).copy())
    out = dict(tile=tile_, flow=np.stack(fl), flow_al=np.stack(fla), H=np.stack(Hs), has_h=np.array(hh, np.float32),
               r0=np.stack(r0), cov_ref=cov_ref, cov=np.stack(cov), grey_ref=grey_ref, grey=np.stack(grey), tune=tune_)
    if flags & FLAG_PARITY:
        Ho, Wo = int(round(scale * H)), int(round(scale * W))
        P = Ho * Wo * 3
        out["Aref"] = take(P).reshape(Ho, Wo, 3).copy()
        out["Bref"] = take(P).reshape(Ho, Wo, 3).copy()
        A, B = [], []
        for _ in range(N - 1):
            A.append(take(P).reshape(Ho, Wo, 3).copy())
            B.append(take(P).reshape(Ho, Wo, 3).copy())
        out["A"], out["B"] = np.stack(A), np.stack(B)
    if pos != buf.size:
        raise RuntimeError(f"conf_harness output size mismatch: read {pos} of {buf.size} bytes")
    return out
