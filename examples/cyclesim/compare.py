# SPDX-License-Identifier: Apache-2.0
"""Compare CPU what-if estimates against published lever observations."""
from pathlib import Path
from common.config import dump, load
from cyclesim.model import Model


def main():
    directory = Path(__file__).resolve().parent
    model = Model(load(directory / "calibration.json"))
    measured = load(directory / "measured.json")
    baseline = model.predict()['cycle_bench_hard_panel']['tok_s_per_stream']
    projection = model.predict(**{'kernel_ms': load(directory / "fp8-projection.json")['kernel_ms']})
    glue = model.predict(**load(directory / "glue-lite.json"))
    return dump({"model": "GLM-5.3", "tp": 4,
                 "fp8_projection": {"predicted_prose_gain_pct": 100 * (projection['cycle_bench_hard_panel']['tok_s_per_stream'] / baseline - 1),
                                    "measured_prose_gain_pct": measured["fp8_projection"]["prose_gain_pct"]},
                 "glue_lite": {"predicted_prose_gain_pct": 100 * (glue['cycle_bench_hard_panel']['tok_s_per_stream'] / baseline - 1),
                               "measured_prose_gain_pct": measured["glue_lite"]["prose_gain_pct"]},
                 "interpretation": "hypotheses compared individually; no combined performance claim"})


if __name__ == "__main__":
    main()
