# Offline experiment fixture

Run from the repository root:

```sh
lab experiment check examples/experiment/manifest.json
lab experiment report examples/experiment/manifest.json --out report.json --markdown report.md
```

Both output destinations must be new files. The example combines a synthetic
static kernel PASS and a cyclesim estimate. Neither is a hardware qualification.
All identity and configuration hashes are illustrative operator declarations.
The report retains `fixture` and `simulation` origins and the original estimated
and assumed status strings. Its integrity PASS only verifies the explicit bytes
and supported adapter contracts. It has no combined promotion decision.

See [the contract](../../docs/experiment.md) for preregistered A/B/A-prime and
quality-panel adapters, evidence classification and limits.
