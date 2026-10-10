# SPDX-License-Identifier: Apache-2.0
"""Run the existing calibration validator on anonymised timing observations."""
from pathlib import Path
from common.config import dump, load
from cyclesim.calibrate import validate


def main():
    directory = Path(__file__).resolve().parent
    calibration = load(directory / 'calibration.json')
    rows = validate(calibration, load(directory / 'holdout.json'))
    return dump({'rows': rows, 'metric_errors': calibration['metric_errors'],
                 'scope': 'held-out cycle timing only; no coverage guarantee'})


if __name__ == '__main__':
    main()
