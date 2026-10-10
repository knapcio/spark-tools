# CPU and packaging checks

CI runs the core suite on Linux with Python 3.10 and 3.13 and on macOS with
Python 3.13. The core job installs `.[test]` and explicitly excludes
`tests/test_rankharness.py` and `tests/test_rankharness_fix1.py`, which require
the optional CPU PyTorch/Gloo and NumPy dependencies. Its example runner
identifies optional rank examples when those dependencies are unavailable.
The full Linux rank job installs `.[test,ranks]` and runs every test and example.

The distribution job builds a wheel and source archive, checks their public
assets and licenses, runs the dependency-light suite from the extracted source
archive, and audits the source export. It installs the wheel into a fresh
environment and runs the quickstart commands outside the checkout. Wheel
configuration, documentation and example copies live at
`share/spark-tools` under that environment; package imports come from the
installed wheel. The wheel also retains the occupancy C++ source and MoE prior.

The checks use local CPU fixtures. CUDA compilation, GPUs, external endpoints
and publishing are outside this workflow. CI produces downloadable distribution
artifacts and has read-only repository permissions; no publishing credentials
are required.

Export audits use standalone repositories without a remote, staging each
exported file so the audit includes its Git blob and index. They check the
exported snapshot; auditing all original repository history remains a separate
release requirement described in [the export policy](export-policy.md).
