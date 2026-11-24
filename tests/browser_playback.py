"""Optional browser verification. Requires Playwright and a Chromium binary."""
import argparse
import bisect
import json
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kmc_test import DislocationKMC

parser = argparse.ArgumentParser()
parser.add_argument("--html", type=Path, default=Path("outputs/v0_baseline/playback.html"))
parser.add_argument("--browser")
parser.add_argument("--screenshots", type=Path, default=Path("/tmp/kmc-playback-check"))
args = parser.parse_args()
args.screenshots.mkdir(parents=True, exist_ok=True)
trace = json.loads(args.html.with_name("trace.json").read_text())
events = trace["events"]
times = [e["time_s"] for e in events]

def check_state(page, expected_trace=None):
    reference = trace if expected_trace is None else expected_trace
    reference_events = reference["events"]
    reference_times = [e["time_s"] for e in reference_events]
    actual = page.evaluate("window.kmcReplay.getState()")
    assert actual["stress_mpa"] == reference["parameters"]["stress_mpa"]
    count = bisect.bisect_right(reference_times, actual["time_s"])
    assert actual["event_index"] == count
    expected = reference["initial_positions_m"].copy()
    for event in reference_events[:count]: expected[event["segment_id"]] = event["position_after_m"]
    assert actual["positions_m"] == expected
    assert actual["mean_position_m"] == (reference_events[count-1]["mean_position_m"] if count else 0)
    return actual

with sync_playwright() as playwright:
    launch = {"headless": True, "timeout": 15000, "args": ["--no-sandbox"]}
    if args.browser: launch["executable_path"] = args.browser
    browser = playwright.chromium.launch(**launch)
    page = browser.new_page(viewport={"width": 1360, "height": 1000}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    remote_requests = []
    page.on("request", lambda request: remote_requests.append(request.url) if request.url.startswith(("http://", "https://")) else None)
    page.context.set_offline(True)
    page.goto(args.html.resolve().as_uri(), wait_until="load")
    page.wait_for_function("Boolean(window.kmcReplay)")
    page.evaluate("document.fonts.ready")
    for field in ("stress-work", "barrier-law-forward", "barrier-law-backward", "rate-math", "forward-probability", "rate-forward", "rate-backward"):
        assert page.locator(f"#{field} .katex-html").is_visible()
    assert page.locator("#rate-math math mfrac").count() == 1
    assert check_state(page)["event_index"] == 0
    if events:
        page.locator("#next").click()
        assert check_state(page)["event_index"] == 1
        assert page.locator("#wait-substitution .katex-html").is_visible()
        assert page.locator("#wait-substitution math mfrac").count() == 1
        page.evaluate("t => kmcReplay.seekTime(t)", times[0]/2)
        assert check_state(page)["event_index"] == 0
        page.evaluate("t => kmcReplay.seekTime(t)", times[0])
        assert check_state(page)["event_index"] == 1
        for fraction in (1, .31, .77, .13, 0, .99, .5):
            page.evaluate("t => kmcReplay.seekTime(t)", times[-1]*fraction)
            check_state(page)
        # Explicitly check every state in a short prefix and both sides of jumps.
        for i in range(min(30, len(events))):
            before = 0 if i == 0 else times[i-1]
            page.evaluate("t => kmcReplay.seekTime(t)", (before+times[i])/2)
            assert check_state(page)["event_index"] == i
            page.evaluate("n => kmcReplay.seekEvent(n)", i+1)
            assert check_state(page)["event_index"] == i+1
        page.locator("#segment").select_option(str(min(5, trace["parameters"]["num_segments"]-1)))
        check_state(page)
        page.locator("#reset").click()
        page.locator("#play").click()
        page.wait_for_timeout(350)
        page.locator("#play").click()
        state = check_state(page)
        assert state["time_s"] > 0
        page.wait_for_timeout(100)
        assert page.evaluate("kmcReplay.getState().time_s") == state["time_s"]
        page.locator("#mode").select_option("event")
        page.locator("#event-rate").select_option("20")
        page.locator("#reset").click()
        page.locator("#play").click()
        page.wait_for_timeout(300)
        page.locator("#play").click()
        assert check_state(page)["event_index"] > 0
        page.locator("#seek").evaluate("el => {el.value=1;el.dispatchEvent(new Event('input'));}")
        assert check_state(page)["event_index"] == len(events)
        page.locator("#previous").click()
        assert check_state(page)["event_index"] == len(events)-1
        page.locator("#mode").select_option("time")
        page.evaluate("n => kmcReplay.seekEvent(n)", max(1, len(events)//2))
    else:
        assert page.locator("#play").is_disabled()
        assert page.locator("#next").is_disabled()
    stresses = page.evaluate("kmcReplay.availableStresses")
    if len(stresses) > 1:
        original_clock = page.locator("#duration").input_value()
        for stress in (0, -10, 100, trace["parameters"]["stress_mpa"]):
            if stress not in stresses: continue
            # Change stress while playback is active: the next run must start paused.
            if events:
                page.locator("#reset").click()
                page.locator("#play").click()
            page.locator("#stress").evaluate("(el,n)=>{el.value=n;el.dispatchEvent(new Event('input'));}", stresses.index(stress))
            p = trace["parameters"]
            sim = DislocationKMC(temp_k=p["temperature_k"], stress_mpa=stress,
                                num_segments=p["num_segments"], seed=p["seed"], disloc_density=p["mobile_density_m2"])
            sim.run_simulation(p["requested_events"], record_events=True)
            expected_trace = sim.recorded_trace()
            actual_trace = page.evaluate("kmcReplay.trace")
            assert actual_trace["events"] == expected_trace["events"]
            assert actual_trace["parameters"] == expected_trace["parameters"]
            state = check_state(page, expected_trace)
            assert state["event_index"] == 0 and not state["running"]
            assert page.locator("#duration").input_value() == original_clock
            assert page.locator("#stress-value").inner_text() == f"{stress:g} MPa"
            assert page.locator("#wait-substitution").is_hidden()
            work = stress*1e6*p["activation_volume_m3"]/1.602176634e-19
            assert f"{work:.5f}" in page.locator("#stress-work annotation").text_content()
            if stress == 0:
                assert r"50.000\%" in page.locator("#forward-probability annotation").text_content()
            if expected_trace["events"]:
                first_time = expected_trace["events"][0]["time_s"]
                for t in (first_time/2, first_time, expected_trace["results"]["total_time_s"]):
                    page.evaluate("t=>kmcReplay.seekTime(t)", t)
                    check_state(page, expected_trace)
            # Selected-run exports must not silently download the 10 MPa baseline.
            with page.expect_download() as download_info:
                page.locator("#download-trace").click()
            download = download_info.value
            downloaded = json.loads(Path(download.path()).read_text())
            assert downloaded == actual_trace
            with page.expect_download() as download_info:
                page.locator("#download-events").click()
            import csv
            with Path(download_info.value.path()).open() as file:
                rows = list(csv.DictReader(file))
            assert len(rows) == len(expected_trace["events"])
            if rows: assert float(rows[-1]["time_s"]) == expected_trace["results"]["total_time_s"]
        page.evaluate("n=>kmcReplay.seekEvent(n)", len(events)//2)
        check_state(page)
    page.screenshot(path=str(args.screenshots/"desktop.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
    page.screenshot(path=str(args.screenshots/"mobile.png"), full_page=True)
    assert not errors, errors
    assert page.locator(".katex-error").count() == 0
    assert not remote_requests, remote_requests
    browser.close()
print("Browser checks passed: exact states, controls, stress changes, downloads, offline LaTeX rendering, and mobile layout")
