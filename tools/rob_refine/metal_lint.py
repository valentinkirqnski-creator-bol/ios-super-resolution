"""Structural checks on the Metal source, standing in for a compile.

There is no Metal toolchain on this machine, so this is what can be verified
without one: that the file is balanced, that the kernel's signature matches
what the host binds, that the parameter struct the host writes has the size the
host asserts, and that every identifier the new kernel uses is actually defined
somewhere it can see. None of that proves it compiles; all of it catches the
mistakes that a compile would catch first.
"""
import re
import sys

metal = open("core/HHSRKernels.metal", encoding="utf-8").read()
shared = open("core/robustness_refine_shared.h", encoding="utf-8").read()
host = open("core/metal_gpu.mm", encoding="utf-8").read()

fail = []


def strip_comments_strings(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    s = re.sub(r'"(?:\\.|[^"\\])*"', '""', s)
    s = re.sub(r"'(?:\\.|[^'\\])*'", "''", s)
    return s


for name, src in (("HHSRKernels.metal", metal), ("robustness_refine_shared.h", shared)):
    c = strip_comments_strings(src)
    for open_ch, close_ch in (("{", "}"), ("(", ")"), ("[", "]")):
        d = c.count(open_ch) - c.count(close_ch)
        if d:
            fail.append(f"{name}: {open_ch}{close_ch} unbalanced by {d}")

# The kernel's buffer indices must be exactly what the host sets, in order.
kern = metal[metal.index("kernel void rob_refine_mask"):]
kern = kern[:kern.index("\n}") + 2]
kbufs = re.findall(r"\[\[buffer\((\d+)\)\]\]", kern)
if kbufs != [str(i) for i in range(11)]:
    fail.append(f"kernel buffer indices {kbufs}, expected 0..10 with no gaps")

hb = host[host.index("id<MTLComputePipelineState> refine = c.pipe"):]
hb = hb[:hb.index("dispatch2(enc, refine")]
hidx = re.findall(r"atIndex:(\d+)\]", hb)
if sorted(int(i) for i in hidx) != list(range(11)):
    fail.append(f"host binds indices {sorted(int(i) for i in hidx)}, expected 0..10")

# Argument order: buffer(n) in the kernel must be the n-th thing the host binds.
korder = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)\]\]", kern)
korder = {int(n): nm for nm, n in korder}
horder = {}
for m in re.finditer(r"\[enc set(?:Buffer|Bytes):&?(\w+)[^\]]*atIndex:(\d+)\]", hb):
    horder[int(m.group(2))] = m.group(1)
PAIRS = {
    0: ("R", "b_out"), 1: ("comp_means", "b_gmeans"), 2: ("ref_means", "b_ref_m"),
    3: ("ref_vars", "b_ref_v"), 4: ("std_curve", "b_std"), 5: ("diff_curve", "b_diff"),
    6: ("S", "b_S"), 7: ("flow", "b_flow"), 8: ("match_ambiguous", "b_match_amb"),
    9: ("weights", "b_w"), 10: ("p", "rp"),
}
for i, (kname, hname) in PAIRS.items():
    if korder.get(i) != kname:
        fail.append(f"buffer {i}: kernel has {korder.get(i)!r}, expected {kname!r}")
    if horder.get(i) != hname:
        fail.append(f"buffer {i}: host binds {horder.get(i)!r}, expected {hname!r}")

# Everything the kernel body calls must be defined in the shared header or
# earlier in the metal file.
body = kern[kern.index("{"):]
called = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", strip_comments_strings(body)))
BUILTIN = {"if", "for", "while", "return", "int", "uint", "float", "sizeof"}
for fn in sorted(called - BUILTIN):
    if f"{fn}(" in shared or re.search(rf"\b\w+\s+{fn}\s*\(", metal[:metal.index('kernel void rob_refine_mask')]):
        continue
    fail.append(f"kernel calls {fn}() which is not defined before it")

# RefineParams must be 64 bytes as both sides assume.
fields = shared[shared.index("struct RefineParams {"):]
fields = fields[:fields.index("};")]
n_scalars = 0
for line in fields.splitlines():
    line = re.sub(r"//.*", "", line).strip().rstrip(";")
    if not line or line.startswith("struct"):
        continue
    if line.split()[0] in ("int", "float", "unsigned"):
        n_scalars += len([p for p in line.split(None, 1)[1].split(",") if p.strip()])
if n_scalars * 4 != 64:
    fail.append(f"RefineParams has {n_scalars} 4-byte fields = {n_scalars*4} bytes, "
                "but both sides assert 64")
if 'static_assert(sizeof(RefineParams) == 64' not in host:
    fail.append("metal_gpu.mm no longer asserts sizeof(RefineParams) == 64")

# Forbidden in the shared header's CODE, per its own portability rules. Its
# comments discuss these deliberately, so strip them first.
shared_code = strip_comments_strings(shared)
for bad in ("std::", "template<", "template "):
    if bad in shared_code:
        fail.append(f"shared header uses {bad!r}, which is not portable to Metal")
# rr_log1p is the wrapper; a bare call to log1p is what must not appear.
if re.search(r"(?<!rr_)\blog1p\s*\(", shared_code):
    fail.append("shared header calls log1p directly; use rr_log1p")

print(f"{len(fail)} problem(s)")
for f in fail:
    print("  FAIL", f)
sys.exit(1 if fail else 0)
