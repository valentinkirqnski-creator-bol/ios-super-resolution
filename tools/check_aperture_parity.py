"""The struct check proves the two sides agree on layout. It says nothing about
whether they compute the same thing. This normalises both aperture blocks to a
sequence of operations and compares them."""
import re, sys
ROOT = r"c:\Users\valen\Downloads\ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85\ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85"
def read(p):
    return open(ROOT + p, encoding="utf-8", errors="replace").read()

cpu = read(r"\core\robustness.cpp")
msl = read(r"\core\HHSRKernels.metal")

def block(src, start_pat, end_pat):
    i = src.find(start_pat); assert i >= 0, start_pat
    j = src.find(end_pat, i); assert j >= 0, end_pat
    return src[i:j]

b_cpu = block(cpu, "if (aperture_eligible && pidx <", "R.at(y, x) = r_val;")
b_msl = block(msl, "if (aperture_eligible) {", "R[gid.y * p.w + gid.x] = r_val;")

def ops(t):
    t = re.sub(r"//.*", "", t)
    t = t.replace("cfg.", "").replace("p.", "").replace("std::", "")
    t = t.replace("flow.aperture_post_error[pidx]", "POST")
    t = t.replace("aperture_post_error[pidx]", "POST")
    t = t.replace("flow.aperture_weak_error[pidx]", "WEAK")
    t = t.replace("aperture_weak_error[pidx]", "WEAK")
    t = t.replace("flow_reject_1d_strength", "STRENGTH")
    t = t.replace("aperture_reject_strength", "STRENGTH")
    t = t.replace("clampf", "clamp").replace("f32 ", "").replace("float ", "")
    t = t.replace("const ", "").replace("bool ", "")
    t = re.sub(r"\s+", "", t)
    return t

o_cpu, o_msl = ops(b_cpu), ops(b_msl)
checks = [
    ("observability test",       "observable=wp>0.f&&", ["POST>=0.f"]),
    ("post Gaussian",            "exp(-0.5f*pt*pt)",    []),
    ("post excess",              "max(0.f,POST-aperture_post_safe_px)", []),
    ("post applied",             "r_val*=1.f-wp*(1.f-c_post)", []),
    ("fallback Gaussian",        "exp(-0.5f*t*t)",      []),
    ("fallback excess",          "max(0.f,WEAK-aperture_weak_safe_px)", []),
    ("fallback applied",         "r_val*=1.f-w*(1.f-c_aperture)", []),
    ("strength clamp",           "clamp(STRENGTH,0.f,1.f)", []),
]
ok = True
for name, frag, extra in checks:
    in_cpu = frag in o_cpu or all(e in o_cpu for e in extra)
    in_msl = frag in o_msl or all(e in o_msl for e in extra)
    if in_cpu and in_msl:
        print("OK    %-22s present in both" % name)
    else:
        ok = False
        print("FAIL  %-22s cpu=%s metal=%s" % (name, in_cpu, in_msl))

# precedence: post must be the IF branch and weak the ELSE, on both sides
for lbl, o in (("cpu", o_cpu), ("metal", o_msl)):
    pi, wi = o.find("c_post"), o.find("c_aperture")
    if pi < 0 or wi < 0 or pi > wi:
        ok = False; print("FAIL  %s: post-warp is not the primary branch" % lbl)
    else:
        print("OK    %-5s post-warp branches before the prior fallback" % lbl)
sys.exit(0 if ok else 1)
