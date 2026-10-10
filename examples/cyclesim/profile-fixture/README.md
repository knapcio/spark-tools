# Synthetic profile fixture

From the existing offline regression fixture: four ranks and a bracketed
capture with hashed companions. These files contain synthetic events and
generic configuration metadata.

Run from the checkout root:

```sh
python3 -m cyclesim.profile examples/cyclesim/profile-fixture \
  --ledger examples/cyclesim/ledger.json --check
```

Expected: measured-time/estimated-traffic, 4 ranks and 16 kernel rows.
Runtime: under one second on macOS or Linux. No GPU.
