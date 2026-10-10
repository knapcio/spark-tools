# File and counter utilities

From the checkout root:

```sh
lab acceptance --before examples/lab/before.prom \
  --after examples/lab/after.prom --k 2
lab sha256 examples/lab/before.prom
```

Expected: survival [0.8, 0.4], 2.2 committed tokens per cycle and ten draft
opportunities; the hash command returns the file's SHA256. Runtime: under one
second on macOS/Linux. No GPU or endpoint. These are the existing counter
arithmetic test fixtures. Counter resets, missing positions and inconsistent
accepted totals fail. Nominal drafted-token counts do not identify executed
pass counts. The utility library also retains bounded Chrome trace streaming,
interval union and descriptive statistics from the source mac-tools bundle.
