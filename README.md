# KMC dislocation glide in Ni-Fe

Version 0.1 records and displays the existing model: independent segments hopping
forward or backward along one axis. The physics and event-selection algorithm
match the original `kmc_test.py`. Fe obstacles are a later step.

## Open the recorded simulation

Open [`outputs/v0_baseline/playback.html`](outputs/v0_baseline/playback.html) in a
web browser. It works offline. **Play** follows physical time; **Next event** and
**Previous event** inspect individual jumps. The vertical rows are segment IDs,
not a second spatial direction. Positions remain fixed between events.

The browser only replays Python-generated data. It does not simulate new motion.
At high playback speed, several recorded events may occur between display frames.
The event controls still expose every jump.

Equations and scientific notation use LaTeX rendered with KaTeX. Its library and
fonts are embedded in the exported HTML, so formulas work offline and update with
the stress slider and event controls. The pinned library, license, and file hashes
are in `visualization/vendor/katex/`; no Node installation or CDN is needed to replay.

The **stress slider** selects one of 21 separate constant-stress Python runs,
from **−100 to +100 MPa in 10 MPa steps**. Each setting uses 300 K, 24 segments,
2,000 events, and seed 454. Changing stress resets playback; it does not apply
a stress ramp to an ongoing trajectory. The embedded data work offline and in
VS Code Live Preview. Every setting is computed and verified without interpolation.

Physical-time playback defaults to **1 µs of simulation per second of playback**.
That scale stays fixed when stress changes. At high stress, select a smaller clock
scale or step event by event. Plot axes rescale for each run; runs stop at the same
event count, so their simulated durations differ. At zero stress, hops continue
in both directions with equal probability. Negative stress favors leftward motion.

The CSV and JSON download links follow the selected stress. Files named
`trace.json`, `events.csv`, and `overview.*` on disk remain the 10 MPa reference.
`stress_sweep.json` summarizes verification for all settings. Reusing the same seed
pairs the runs for comparison; it does not supply independent statistical replicates.

## Generate another run

Python needs NumPy and Matplotlib. From this directory:

```bash
python kmc_visualize.py --stress-slider --verify
python kmc_visualize.py --temperature 300 --stress 0 --segments 24 --steps 2000 --seed 454 --output outputs/v0_zero_stress
```

The first command reproduces the demonstration (300 K, 10 MPa, 24 segments,
2,000 events, seed 454) and runs analytical drift comparisons. An output folder
contains the offline playback, full-precision `trace.json`, `events.csv`, and
static verification plots in PNG/PDF. The slider embeds the additional full traces
in the HTML (about 20 MB). Omit `--stress-slider` to export just one run.
The optional `verification.json` contains
replicate statistics. Physical units in the data files are SI, except energies
in eV and the explicitly named stress field in MPa.

For direct use:

```python
from kmc_test import DislocationKMC

sim = DislocationKMC(temp_k=300, stress_mpa=10, num_segments=24, seed=454)
metrics = sim.run_simulation(2000, record_events=True)
trace = sim.recorded_trace()
```

With no seed, the original NumPy global random stream remains available.
A supplied seed uses an independent `RandomState` with the same legacy sequence.
Repeated runs on one instance continue that stream; each trace saves its starting
RNG state. Recording is optional and consumes no extra random numbers.

## Verification

```bash
python -m unittest discover -s tests -v
python tools/check_legacy_baseline.py
```

The tests check reconstruction, unchanged random draws, rate symmetry below the
barrier floor, segment-count behavior, and input/empty-run handling. The legacy
comparison checks exact agreement with commit `a7e4325` across six cases.

Browser checks are optional and need Playwright with Chromium:

```bash
python tests/browser_playback.py
```

Use `--browser /path/to/chromium` when the binary is outside Playwright's default
location. The browser test checks states during waiting periods and at event
boundaries, backward seeking, playback controls, and the mobile layout. It also
compares selected stress runs to fresh Python results, checks resets and clock
scale, and verifies that downloaded CSV/JSON contain the selected run.

## Project brief

The editable brief and PDF are in `Documents/`. Its version record tracks the
simulation, assumptions, outputs, and verification. Update
`Documents/project_brief.json` when a model version changes, then run:

```bash
python tools/build_project_brief.py
```

This writes the Word document and an HTML print version. The helper's `--pdf`
option also renders a PDF when Playwright and Chromium are available. It accepts
the same `--browser` override.

## Limits of Version 0

The **0.35 eV barrier**, **20b³ activation volume**, and **10¹² s⁻¹ prefactor**
are assumed inputs inherited from the starting code. No source or fit currently
validates them for Ni glide. The slider explores this assumed law; its range is
not a validated material range.

Stress is a uniform, constant **resolved shear stress**, entered in MPa and
converted to Pa. It changes the barriers by the work term τV*:

```text
E_forward  = max(0.0001 eV, E0 − τV*)
E_backward = E0 + τV*
r± = ν0 exp(−E± / kBT)
P(forward hop, any segment) = r+ / (r+ + r−)
```

Convert τV* from joules to eV before using these expressions. At 10 MPa,
τV* = 0.01924875 eV, so the barriers are 0.33075125 and 0.36924875 eV.
At 300 K, the forward probability is about 81.6%. The same rates set the waiting
times through R = N(r+ + r−). Stress is not applied through boundary forces or
a calculated strain field; segment length and mechanical geometry remain undefined.

The segments do not form a mechanically connected line. The rates are spatially
uniform and constant during a run. Material parameters remain provisional.
The original forward-only barrier floor near 182 MPa is preserved and flagged
in the replay. Agreement with the analytical drift verifies the stochastic
implementation; quantitative validation against nickel data remains open.
