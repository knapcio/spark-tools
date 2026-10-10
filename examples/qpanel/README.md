# Null-calibrated quality fixture

From the checkout root:

```sh
python3 -m qpanel decide --prereg examples/qpanel/prereg.json \
  --a examples/qpanel/A.json --a2 examples/qpanel/A2.json --b examples/qpanel/B.json
```

Expected: PASS. Runtime: under one second on macOS/Linux. No GPU or endpoint.
`expected-decision.json` contains the complete output.

This is the existing synthetic 116-item decision fixture, explained in order:

1. A passes 100 items. A2, the same-model shape-perturbed null, passes 98.
2. Null noise is the absolute **net** loss: |100-98| = 2 items. Both directions
   of item changes are reported separately.
3. A's binomial standard error is sqrt(116 * (100/116) * (16/116)) = 3.714 items.
4. The preregistered tolerance is max(2, 3.714) = 3.714 items.
5. B passes 97: its loss of 3 items is within tolerance, so the rule returns PASS.

These are saved test scores, not a model evaluation or a live load experiment.
The small HTTP adapter accepts caller-provided exact-answer tasks. The owner
of the endpoint must arrange A2's load/shape perturbation on the same baseline
model while preserving the foreground requests. No background traffic or
server lifecycle is created by this package. See [endpoint usage](../../docs/endpoints.md).

Install the package and the dependencies used by the test commands:

```sh
pip install -e '.[test]'
```

The console command `qpanel` accepts the same arguments after installation.
For caller-supplied exact-answer tasks, freeze
`configs/qpanel-endpoint.json` with `python3 -m qpanel preregister`, then use
`python3 -m qpanel run --prereg PLAN --arm A --out OUTPUT` for each arm.
`PLAN` and `OUTPUT` are paths chosen by the caller. Plans and their `.sha256`
companions are create-only; use an external output directory for evaluations.
The existing null-pass rule is fixed. Unsupported rule names, inconsistent
item counts and duplicate task keys are rejected before freezing.

The original mocked-transport fixture and CLI regressions run with:

```sh
python3 -m pytest tests/test_qpanel.py tests/test_qpanel_cli.py
```

Expected: 24 passed, under two seconds on macOS/Linux, no GPU or endpoint.
Only the mocked transport is used. A live run needs an endpoint reachable
from the client; GPU requirements depend on that endpoint's model.
