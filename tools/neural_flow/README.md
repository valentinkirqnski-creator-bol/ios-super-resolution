# PWCNet -> Core ML conversion

Produces `PWCNetFlow.mlpackage`, the model `core/neural_flow.mm` loads at
runtime (as `PWCNetFlow.mlmodelc`, which Xcode compiles it into automatically
once added to the app target).

## Status

Built and debugged on a non-Apple machine (no Xcode/Core ML runtime
available). The full PyTorch -> Core ML graph conversion was verified to
complete cleanly: every op traced, all MIL optimization passes, all backend
`mlprogram` passes. It fails only on the final step -- writing the compiled
weights to disk -- because Windows `coremltools` doesn't ship the native
`libmilstoragepython` blob-writer library (Apple only builds that piece for
macOS). Running the same script here, unmodified, should produce the
`.mlpackage` directly.

Not yet verified: the produced `.mlpackage`'s numerical output (no way to
run Core ML inference outside this environment), and the on-device
`neural_flow.mm` integration (no Xcode available to build/run it).

## Steps (on a Mac)

```bash
python3 -m venv pwcnet_env
source pwcnet_env/bin/activate
pip install torch==2.7.0 coremltools==9.0 numpy==1.26.4 pillow gdown

# Pretrained PWCNet weights (sniklaus/pytorch-pwc, matches deep-rep-master's
# PWCNetAlignment).
mkdir -p pretrained_networks
gdown "https://drive.google.com/uc?id=1s11Ud1UMipk2AbZZAypLPRpnXOS9Y1KO" \
    -O pretrained_networks/pwcnet-network-default.pth

python3 convert_coreml.py
# -> PWCNetFlow.mlpackage
```

Then in Xcode: drag `PWCNetFlow.mlpackage` into the app target (Copy items
if needed, add to target). Xcode compiles it to `PWCNetFlow.mlmodelc` and
bundles it automatically -- `neural_flow.mm`'s `[[NSBundle mainBundle]
URLForResource:@"PWCNetFlow" withExtension:@"mlmodelc"]` finds it by that
name, so no code changes are needed for the load path.

## Why torch==2.7.0 / coremltools==9.0 / numpy<2 specifically

- `coremltools` 9.0 self-reports torch 2.7.0 as the most recently tested
  version; later torch releases traced fine here but weren't risked further.
- `numpy>=2.0` makes coremltools' MIL int-cast conversion hard-fail
  (`TypeError: only 0-dimensional arrays can be converted to Python
  scalars`) on a pattern (`int(1-element-array)`) that numpy<2 handled
  leniently. Pin numpy below 2.0 or this conversion breaks partway through,
  in a way that has nothing to do with the model itself.

## Why `pwcnet_pure.py` isn't just the upstream PWCNet

`deep-rep-master/models/alignment/pwcnet.py` requires a CUDA/CuPy custom
kernel (`external/pwcnet/correlation`) for its cost-volume layer -- no CPU
path, and no relationship to Core ML/Metal execution either way.
`pwcnet_pure.py` reimplements that layer with plain PyTorch ops (verified
against the CUDA kernel's exact math: 9x9 displacement window, per-channel
mean not sum, zero-padded border -- see the comment at the top of the file),
and the pretrained `state_dict` loads into it unmodified since every layer
name matches the original exactly.

It also threads every internal pyramid-level size through as a plain Python
int rather than reading it from tensor `.shape` mid-network. That's not
stylistic -- tracing an intermediate tensor's `.shape` into an arithmetic op
(a division, a `linspace` step count) captures a symbolic graph node even
when the size is fixed for one deployment resolution, and coremltools'
conversion hard-fails on several of the resulting patterns (int-cast of a
non-scalar array, `reciprocal` of an int32 tensor). Since this model is
converted for one fixed resolution (`H, W` at the top of `convert_coreml.py`,
currently 1512x2016 -- must match the app's actual guide resolution, i.e.
half the raw sensor capture size), every level's size is known in closed
form ahead of time (each halves the previous via a stride-2 conv), so
nothing is lost by making that explicit instead of shape-derived.

If you change `H`/`W` in `convert_coreml.py` (different sensor resolution),
also update `kModelH`/`kModelW` in `core/neural_flow.mm` to match -- they're
intentionally not derived from each other, so a mismatch fails loudly
(`neural_flow_estimate` refuses rather than silently resizing) instead of
quietly feeding the model a shape it wasn't converted for.
