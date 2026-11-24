# KMC dislocation glide in Ni-Fe

MSE454 final project on dislocation motion in nickel and Ni-Fe alloys.
The current model uses kinetic Monte Carlo to simulate independent segments
hopping along one axis, recording every jump and waiting time.

The implementation has passed baseline checks. Material parameters still need
physical validation; connected 2D dislocation lines and Fe interactions are planned.

## View the simulation

Open [the playback](outputs/v0_baseline/playback.html) locally in a browser or
VS Code Live Preview. It works offline. The stress slider selects recorded runs
from -100 to +100 MPa and resets playback when changed. Use the event controls
to inspect individual jumps. Vertical rows are segment IDs.

## Run

From the project folder:

```bash
python -m pip install -r requirements.txt
python kmc_visualize.py --stress-slider --verify
```

Defaults: 300 K, 10 MPa, 24 segments, 2,000 events, seed 454.
Results and playback are saved in `outputs/v0_baseline/`.
Use `python kmc_visualize.py --help` for other settings.

## Check the implementation

```bash
python -m unittest discover -s tests -v
python tools/check_legacy_baseline.py
```

The [project brief](Documents/KMC%20Project%20Brief%20and%20Parameters.pdf)
records the assumptions, parameters, and verification results.
