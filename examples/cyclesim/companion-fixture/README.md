# Synthetic companion fixture

From the existing offline regression fixture: four ranks and a bracketed
capture with hashed companions. These files contain synthetic events and
generic configuration metadata.

Run from the checkout root:

```sh
python3 -m cyclesim.profile examples/cyclesim/companion-fixture \
  --ledger examples/cyclesim/ledger.json --check
```

Expected: usable-companions-only, 0 ranks and 0 kernel rows.
Runtime: under one second on macOS or Linux. No GPU.
