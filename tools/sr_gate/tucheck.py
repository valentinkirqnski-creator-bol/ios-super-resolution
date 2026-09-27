"""Two compiler-free checks on one translation unit, aimed at exactly the class
of mistake that broke the iOS build twice:

  A) a FILE-SCOPE symbol of this .mm used before the line that declares it.
     C++ needs the declaration first, and inserting a helper block too early in
     a 5000-line file is easy to get wrong. (This is how g_rob_curve_n broke.)

  B) a project symbol used that is declared neither in this file nor in any
     header this file includes, transitively. (This is how kSrGateWeightCount
     broke -- it lives in sr_gate_weights.h, which metal_gpu.mm does not
     include.)

Names it cannot resolve either way are printed once as "external", which should
be system / Metal / Foundation API only.

Both checks were written after the iOS build failed on exactly these two, and
both reproduce them from the committed tree:

    python tools/sr_gate/tucheck.py core/metal_gpu.mm core 1681 1880

Known false positives, so read the output rather than trusting the exit code:
struct members (c.fft_c0 reports as `fft_c0`, since the checker only sees
file-scope declarations), `inline`-prefixed function definitions, and a range's
own declarations. Three pre-existing hits in the whole-file run
(metal_merge_flush_online, metal_merge_wait_inflight,
metal_normalize_band_rgb16_ptr) are forward-declared in ways the patterns miss --
that file compiled before any of this, so they are noise.

Usage: tucheck.py <file.mm> <core_dir> [range_lo range_hi]
"""
import os
import re
import sys

path = sys.argv[1]
coredir = sys.argv[2]
rng = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else None

src = open(path, encoding='utf-8').read().split('\n')
line_comment = re.compile('//.*$')
dq = re.compile('"[^"]*"')
ident = re.compile('[A-Za-z_][A-Za-z0-9_]*')


def clean(line):
    return dq.sub('""', line_comment.sub('', line))


cleaned = [clean(l) for l in src]

# ---- A: file-scope declarations of this file, by line ---------------------
# Anything introduced at column 0. That is where every file-scope declaration in
# this codebase sits, and indented lines are function bodies.
decl_line = {}
patterns = [
    re.compile(r'^static\s+(?:__strong\s+)?(?:const\s+)?[A-Za-z_][\w:<>,\s\*&]*?'
               r'([A-Za-z_]\w*)\s*(?:\(|=|;|\[)'),
    re.compile(r'^(?:struct|class|enum)\s+([A-Za-z_]\w*)'),
    re.compile(r'^#\s*define\s+([A-Za-z_]\w*)'),
    re.compile(r'^(?:constexpr|const)\s+[A-Za-z_][\w:<>,\s\*&]*?([A-Za-z_]\w*)\s*(?:=|;|\[)'),
    re.compile(r'^(?:id<[^>]+>|[A-Za-z_][\w:<>,]*)\s+([A-Za-z_]\w*)\s*\('),
    re.compile(r'^typedef\s+.*?([A-Za-z_]\w*)\s*;'),
]
for i, l in enumerate(cleaned, 1):
    for p in patterns:
        m = p.match(l)
        if m:
            decl_line.setdefault(m.group(1), i)
            break

# ---- B: identifiers available from the headers this file includes ---------
inc_re = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
seen_hdr = set()
header_ids = set()


def scan_header(name):
    if name in seen_hdr:
        return
    seen_hdr.add(name)
    p = os.path.join(coredir, name)
    if not os.path.exists(p):
        return
    txt = open(p, encoding='utf-8').read().split('\n')
    for l in txt:
        c = clean(l)
        m = inc_re.match(l)
        if m:
            scan_header(os.path.basename(m.group(1)))
        for mm in ident.finditer(c):
            header_ids.add(mm.group(0))


for l in src:
    m = inc_re.match(l)
    if m:
        scan_header(os.path.basename(m.group(1)))

lo, hi = rng if rng else (1, len(src))
KEYWORDS = set("""auto bool break case char class const constexpr continue default delete do
double else enum explicit extern false float for goto if inline int long namespace new nullptr
operator private protected public register return short signed sizeof static struct switch
template this throw true try typedef typename union unsigned using virtual void volatile while
size_t uint32_t int32_t uint16_t uint64_t int64_t uint8_t uint uint2 id nil YES NO self
nonnull nullable restrict and or not static_assert sizeof alignof""".split())

# locals declared inside the range, so they are not reported as external
local = set()
loc_re = re.compile(r'^\s*(?:const\s+)?(?:__strong\s+)?[A-Za-z_][\w:<>,\s\*&]*?'
                    r'([A-Za-z_]\w*)\s*(?:=|\{|\[|;)')
for i in range(lo, hi + 1):
    m = loc_re.match(cleaned[i - 1])
    if m:
        local.add(m.group(1))
    # function parameters
    for mm in re.finditer(r'(?:const\s+)?[A-Za-z_][\w:<>,\*&]*\s+([a-z_]\w*)\s*[,)]',
                          cleaned[i - 1]):
        local.add(mm.group(1))

bugs_a, bugs_b, external = [], [], set()
for i in range(lo, hi + 1):
    for m in ident.finditer(cleaned[i - 1]):
        n = m.group(0)
        if n in KEYWORDS or len(n) <= 1:
            continue
        if n in decl_line:
            if decl_line[n] > i and n not in local:
                bugs_a.append((n, i, decl_line[n]))
            continue
        if n in header_ids or n in local:
            continue
        external.add(n)

print('%s  lines %d-%d' % (os.path.basename(path), lo, hi))
print('  headers scanned: %d   file-scope decls found: %d'
      % (len(seen_hdr), len(decl_line)))
if bugs_a:
    print('  A) USED BEFORE DECLARED IN THIS FILE:')
    for n, u, d in sorted(set(bugs_a)):
        print('       %-28s used %d, declared %d' % (n, u, d))
else:
    print('  A) no use-before-declaration')
proj = sorted(n for n in external if n.startswith(('k', 'g_', 'SRG_', 'rob_', 'sr_',
                                                   'merge_', 'align_', 'prof_',
                                                   'fft', 'mps_', 'bf_')))
if proj:
    print('  B) project-looking symbols in neither this file nor its headers:')
    for n in proj:
        print('       %s' % n)
else:
    print('  B) no unresolved project symbols')
print('  %d other unresolved names (expect system / Metal / Foundation API)'
      % (len(external) - len(proj)))
sys.exit(1 if bugs_a or proj else 0)
