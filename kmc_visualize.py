"""Record a seeded Version 0 run and export an offline, data-driven replay.

Run: python kmc_visualize.py --verify
Only the Python KMC engine generates motion. The HTML file replays that record.
"""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kmc_test import DislocationKMC

ROOT = Path(__file__).resolve().parent


def verify_trace(trace):
    """Reconstruct all positions and check each recorded event against its math."""
    p = trace["parameters"]
    positions = np.array(trace["initial_positions_m"], dtype=float)
    previous_time = 0.0
    for number, event in enumerate(trace["events"], 1):
        segment = event["segment_id"]
        assert event["event"] == number
        assert event["direction"] in (-1, 1)
        assert event["time_before_s"] == previous_time
        assert event["time_s"] > previous_time
        assert event["position_before_m"] == positions[segment]
        expected_position = positions[segment] + event["direction"] * p["hop_distance_m"]
        assert event["position_after_m"] == expected_position
        np.testing.assert_allclose(event["dt_s"], -np.log(event["u_time"]) / event["total_rate_s1"], rtol=1e-14)
        np.testing.assert_allclose(event["time_s"], previous_time + event["dt_s"], rtol=1e-14)
        individual_rates = np.r_[np.full(len(positions), event["rate_forward_s1"]),
                                 np.full(len(positions), event["rate_backward_s1"])]
        cdf = np.cumsum(individual_rates) / np.sum(individual_rates)
        chosen = int(np.searchsorted(cdf, event["u_event"]))
        assert chosen % len(positions) == segment
        assert (1 if chosen < len(positions) else -1) == event["direction"]
        assert event["selected_rate_s1"] == individual_rates[chosen]
        positions[segment] = event["position_after_m"]
        np.testing.assert_allclose(event["mean_position_m"], positions.mean(), rtol=1e-13, atol=1e-22)
        previous_time = event["time_s"]
    np.testing.assert_array_equal(positions, trace["final_positions_m"])
    assert previous_time == trace["results"]["total_time_s"]
    return {"passed": True, "events_reconstructed": len(trace["events"]),
            "checks": ["position updates", "event selection", "waiting times", "mean positions", "final state"]}


def baseline_checks(replicates=24, events=2000):
    """Compare independent seeded runs with the exact constant-rate drift."""
    cases = []
    for temperature, stress in [(300, 0), (300, 10), (300, -10), (300, 100), (600, 10)]:
        values = []
        for seed in range(replicates):
            sim = DislocationKMC(temp_k=temperature, stress_mpa=stress, num_segments=24, seed=seed)
            values.append(sim.run_simulation(events)["velocity_m_s"])
        forward, backward = sim.compute_rates()
        exact = sim.b * (forward - backward)
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(replicates))
        cases.append({"temperature_k": temperature, "stress_mpa": stress,
                      "analytic_velocity_m_s": float(exact), "mean_velocity_m_s": mean,
                      "standard_error_m_s": sem, "z_score": float((mean-exact)/sem),
                      "passed": bool(abs(mean-exact) <= 4*sem + 1e-12)})
    return {"replicates": replicates, "events_per_run": events, "num_segments": 24,
            "seeds": list(range(replicates)), "criterion": "absolute error <= 4 standard errors + 1e-12 m/s",
            "limitation": "Checks the stated stochastic model, not its physical accuracy for Ni. Fixed-event stopping gives a small finite-run ratio bias.",
            "cases": cases, "passed": all(case["passed"] for case in cases)}


def plot_overview(trace, destination):
    p, events = trace["parameters"], trace["events"]
    times = np.array([0.0] + [e["time_s"] for e in events])
    means = np.array([0.0] + [e["mean_position_m"] for e in events])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    fig.suptitle(f"Version 0: recorded KMC data | {p['temperature_k']:g} K, {p['stress_mpa']:g} MPa, seed {p['seed']}", fontsize=14)
    ax = axes[0, 0]
    ax.scatter(np.array(trace["final_positions_m"])*1e9, np.arange(p["num_segments"]), s=25, color="#176781")
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set(xlabel="Glide position x (nm)", ylabel="Segment ID (not a spatial coordinate)", title="Final recorded positions")
    ax = axes[0, 1]
    ax.step(times*1e6, means*1e9, where="post", label="Recorded mean", color="#176781")
    ax.plot(times*1e6, p["analytic_velocity_m_s"]*times*1e9, "--", label="Exact ensemble mean", color="#a96119")
    ax.set(xlabel="Physical time (µs)", ylabel="Mean displacement (nm)", title="Positions are constant between events")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    if events:
        waits = np.sort([e["dt_s"]*e["total_rate_s1"] for e in events])
        ax.step(waits, np.arange(1, len(waits)+1)/len(waits), where="post", label="Recorded waits")
        x = np.linspace(0, max(5, float(waits.max())), 300)
        ax.plot(x, -np.expm1(-x), "--", label="Exponential prediction")
    ax.set(xlabel="Scaled waiting time R Δt", ylabel="Cumulative probability", title="Waiting-time distribution", ylim=(0, 1.02))
    if events: ax.legend(fontsize=8)
    ax = axes[1, 1]
    rate_sum = p["rate_forward_s1"] + p["rate_backward_s1"]
    observed = np.array([sum(e["direction"] == d for e in events) for d in (1, -1)]) / max(1, len(events))
    expected = np.array([p["rate_forward_s1"], p["rate_backward_s1"]]) / rate_sum if rate_sum else np.zeros(2)
    x = np.arange(2)
    ax.bar(x-.17, observed, .34, label="Recorded", color="#176781")
    ax.bar(x+.17, expected, .34, label="Rate prediction", color="#d99448")
    ax.set(xticks=x, xticklabels=["Forward", "Backward"], ylabel="Fraction of events", title="Event selection", ylim=(0, 1))
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    fig.savefig(destination/"overview.png", dpi=180)
    fig.savefig(destination/"overview.pdf")
    plt.close(fig)


def record_stress_runs(reference, stresses=range(-100, 101, 10)):
    """Run the original Python KMC independently at each constant stress.

    Reusing a seed makes comparisons reproducible; these are paired trajectories,
    not independent statistical replicates. Include the requested reference case.
    """
    p = reference["parameters"]
    runs = []
    for stress in sorted(set(stresses) | {p["stress_mpa"]}):
        if stress == p["stress_mpa"]:
            runs.append(reference)
            continue
        sim = DislocationKMC(temp_k=p["temperature_k"], stress_mpa=stress,
                            num_segments=p["num_segments"],
                            disloc_density=p["mobile_density_m2"], seed=p["seed"])
        sim.run_simulation(p["requested_events"], record_events=True)
        runs.append(sim.recorded_trace())
    return runs


def write_playback(payload, destination):
    """Embed recorded data and KaTeX assets in one offline HTML file."""
    vendor = ROOT/"visualization"/"vendor"/"katex"
    css = (vendor/"katex.min.css").read_text()
    # Chromium/VS Code supports WOFF2; embed each font once, without network URLs.
    def embed_font(match):
        font = base64.b64encode((vendor/match[1]).read_bytes()).decode("ascii")
        return f'src:url(data:font/woff2;base64,{font}) format("woff2")'
    css = re.sub(r'src:url\((fonts/[^)]+\.woff2)\)[^}]+', embed_font, css)
    script = (vendor/"katex.min.js").read_text().replace("</script", "<\\/script")
    license_text = (vendor/"LICENSE").read_text().replace("*/", "* /")
    template = (ROOT/"visualization"/"playback.html").read_text()
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = template.replace("__KATEX_CSS__", css).replace("__KATEX_JS__", f"/*\n{license_text}\n*/\n{script}")
    Path(destination).write_text(html.replace("__TRACE_DATA__", data))


def export_run(trace, output, stress_runs=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = stress_runs if stress_runs is not None else [trace]
    code_hash = hashlib.sha256((ROOT/"kmc_test.py").read_bytes()).hexdigest()
    for run in runs:
        run["simulation_code_sha256"] = code_hash
        run["verification"] = verify_trace(run)
    if stress_runs is not None:
        report = {"simulation_code_sha256": code_hash, "passed": True,
                  "note": "Separate constant-stress Python KMC runs with the same seed; not a loading ramp or independent replicates.",
                  "cases": [{"parameters": r["parameters"], "results": r["results"],
                             "verification": r["verification"]} for r in runs]}
        (output/"stress_sweep.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    else:
        (output/"stress_sweep.json").unlink(missing_ok=True)
    text = json.dumps(trace, allow_nan=False, separators=(",", ":"))
    (output/"trace.json").write_text(text + "\n")
    fieldnames = list(trace["events"][0]) if trace["events"] else [
        "event", "segment_id", "direction", "time_before_s", "time_s", "dt_s", "position_before_m",
        "position_after_m", "mean_position_m", "rate_forward_s1", "rate_backward_s1", "total_rate_s1",
        "selected_rate_s1", "u_time", "u_event"]
    with (output/"events.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace["events"])
    write_playback({"default_stress_mpa": trace["parameters"]["stress_mpa"], "runs": runs}, output/"playback.html")
    plot_overview(trace, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=300)
    parser.add_argument("--stress", type=float, default=10, help="resolved shear stress, MPa")
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=454)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs"/"v0_baseline")
    parser.add_argument("--verify", action="store_true", help="also run the seeded analytical comparisons")
    parser.add_argument("--stress-slider", action="store_true",
                        help="embed verified runs from -100 to 100 MPa in 10 MPa steps for offline stress control")
    args = parser.parse_args()
    sim = DislocationKMC(temp_k=args.temperature, stress_mpa=args.stress, num_segments=args.segments, seed=args.seed)
    sim.run_simulation(args.steps, record_events=True)
    trace = sim.recorded_trace()
    runs = record_stress_runs(trace) if args.stress_slider else None
    export_run(trace, args.output, runs)
    if runs is not None:
        print(f"Stress slider: {len(runs)} verified runs, {sum(len(r['events']) for r in runs):,} events")
    if args.verify:
        checks = baseline_checks()
        (args.output/"verification.json").write_text(json.dumps(checks, indent=2, allow_nan=False)+"\n")
        if not checks["passed"]:
            raise SystemExit("Analytical comparison failed; inspect verification.json")
        print(f"Analytical comparisons passed: {len(checks['cases'])} cases, {checks['replicates']} seeds each")
    print(f"Verified {len(sim.events)} recorded events")
    print(f"Open in a browser: {args.output/'playback.html'}")


if __name__ == "__main__":
    main()
