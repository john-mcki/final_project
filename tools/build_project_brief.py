"""Build the project brief from its editable source and recorded model results."""
import argparse
from html import escape
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT/"Documents"
NAME = "KMC Project Brief and Parameters"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W)


def version_record(name):
    folder = ROOT/"outputs"/name
    trace = json.loads((folder/"trace.json").read_text())
    checks = json.loads((folder/"verification.json").read_text())
    regression = json.loads((folder/"legacy_regression.json").read_text())
    p = trace["parameters"]
    sweep_path = folder/"stress_sweep.json"
    sweep = json.loads(sweep_path.read_text()) if sweep_path.exists() else None
    sweep_rows = []
    if sweep:
        stresses = [c["parameters"]["stress_mpa"] for c in sweep["cases"]]
        sweep_rows = [["Stress control", f"{len(stresses)} separately computed runs, {min(stresses):g} to {max(stresses):g} MPa. Changing stress resets playback. Temperature, seed, and event count stay fixed; this is not a loading ramp."],
                      ["How stress enters", "At 10 MPa, τV* = 0.01925 eV. E₊ = 0.33075 eV and E₋ = 0.36925 eV, giving 81.6% forward hops at 300 K. These follow assumed parameters, not a Ni fit."]]
    rows = [[f"{c['temperature_k']} K, {c['stress_mpa']} MPa", f"{c['analytic_velocity_m_s']:.3e}",
             f"{c['mean_velocity_m_s']:.3e} ± {c['standard_error_m_s']:.2e}"] for c in checks["cases"]]
    return [
        ["title", "Version 0.1: stress and event playback" if sweep else "Version 0: recorded event playback"],
        ["subtitle", "Model record · November 23, 2026"],
        ["p", "Version 0 records the original 1D KMC model. Version 0.1 adds stress selection, simpler styling, and LaTeX equations that update during playback and work offline. The barrier law and event algorithm are unchanged."],
        ["table", ["Item", "Version 0"], [
            ["Demonstration", f"{p['temperature_k']:g} K; {p['stress_mpa']:g} MPa; {p['num_segments']} independent segments; {len(trace['events']):,} events; seed {p['seed']}. This smaller segment count keeps individual dots readable."],
            *sweep_rows,
            ["Export and replay", "kmc_visualize.py embeds full event logs in offline HTML. CSV and JSON downloads follow the selected stress. The browser reconstructs recorded states."],
            ["Playback timing", "Positions stay fixed between events. Physical clock speed stays fixed across stress settings; plot axes rescale. Event mode exposes every jump. Fast runs can finish within one display frame."],
            ["Views", "Segment positions, mean displacement and exact expectation, one segment's trajectory, barriers, rates, and waiting-time calculation."],
        ], [25, 75]],
        ["heading", "Verification during this update"],
        ["p", f"All {sum(c['verification']['events_reconstructed'] for c in sweep['cases']) if sweep else len(trace['events']):,} recorded events reconstruct correctly. Nine unit tests and {len(regression['cases'])} exact comparisons with the original code pass. Browser checks cover waits, jumps, seeking, stress changes, fixed clock scale, and selected-run downloads."],
        ["p", f"Analytical drift checks use {checks['replicates']} seeds per case, {checks['num_segments']} segments, and {checks['events_per_run']:,} events per run. All five cases pass the criterion of agreement within four standard errors. Values below are in m/s; uncertainty is one standard error of the mean."],
        ["table", ["Condition", "Exact drift", "KMC mean ± standard error"], rows, [28, 24, 48]],
        ["p", "The finite-event stopping rule gives a small bias in the displacement/time estimator. These comparisons check the stochastic implementation. Parameter calibration and physical validation for Ni remain open."],
        ["heading", "Files and next step"],
        ["p", "Open outputs/v0_baseline/playback.html. trace.json, events.csv, and overview plots hold the 10 MPa reference. stress_sweep.json records checks for all slider settings."],
        ["p", "Reproduce: python kmc_visualize.py --stress-slider --verify. See README.md for tests and parameter limitations."],
        ["callout", "Review the recorded jumps and the physical parameter sources before adding obstacles or interactions. Add a new model record here when the physics changes."],
    ]


def element(tag, attrs=None, text=None):
    node = ET.Element(f"{{{W}}}{tag}", {f"{{{W}}}{k}": str(v) for k, v in (attrs or {}).items()})
    if text is not None: node.text = text
    return node


def child(parent, tag, attrs=None, text=None):
    node = element(tag, attrs, text)
    parent.append(node)
    return node


def paragraph(text, style):
    p = element("p")
    child(child(p, "pPr"), "pStyle", {"val": style})
    child(child(p, "r"), "t", text=text)
    return p


def word_table(block):
    _, headers, rows, widths = block
    table = element("tbl")
    props = child(table, "tblPr")
    child(props, "tblW", {"w": 10080, "type": "dxa"})
    child(props, "tblLayout", {"type": "fixed"})
    margins = child(props, "tblCellMar")
    for edge, value in [("top", 55), ("bottom", 45), ("left", 85), ("right", 85)]:
        child(margins, edge, {"w": value, "type": "dxa"})
    borders = child(props, "tblBorders")
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        child(borders, edge, {"val": "single", "sz": 4, "color": "D8E1E7"})
    grid = child(table, "tblGrid")
    for width in widths: child(grid, "gridCol", {"w": int(10080*width/100)})
    for i, row in enumerate([headers]+rows):
        tr = child(table, "tr")
        trprops = child(tr, "trPr")
        child(trprops, "cantSplit")
        if i == 0: child(trprops, "tblHeader")
        for value, width in zip(row, widths):
            cell = child(tr, "tc")
            props = child(cell, "tcPr")
            child(props, "tcW", {"w": int(10080*width/100), "type": "dxa"})
            child(props, "shd", {"fill": "123A52" if i == 0 else ("F1F6F8" if i % 2 else "FFFFFF")})
            cell.append(paragraph(value, "TableHead" if i == 0 else "Cell"))
    return table


def write_docx(pages, title):
    # Reuse this project's existing Word style sheet while replacing its content.
    destination = DOCS/(NAME+".docx")
    with ZipFile(destination) as old:
        package = {name: old.read(name) for name in old.namelist()}
    document = element("document")
    body = child(document, "body")
    mapping = {"title": "Title", "subtitle": "Subtitle", "heading": "Heading", "p": "Normal",
               "step": "Step", "source": "Source", "equation": "Equation", "callout": "Callout"}
    for i, page in enumerate(pages):
        if i: child(child(child(body, "p"), "r"), "br", {"type": "page"})
        for block in page:
            body.append(word_table(block) if block[0] == "table" else paragraph(block[1], mapping[block[0]]))
    section = child(body, "sectPr")
    child(section, "pgSz", {"w": 12240, "h": 15840})
    child(section, "pgMar", {"top": 720, "right": 1080, "bottom": 720, "left": 1080, "header": 360, "footer": 360, "gutter": 0})
    package["word/document.xml"] = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    temporary = destination.with_suffix(".docx.tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as out:
        for name, contents in package.items(): out.writestr(name, contents)
    with ZipFile(temporary) as check:
        assert check.testzip() is None
        for name in check.namelist(): ET.fromstring(check.read(name))
    temporary.replace(destination)


CSS = """
@page{size:Letter;margin:0}*{box-sizing:border-box}body{margin:0;color:#24354b;font-family:Arial,'DejaVu Sans',sans-serif;font-size:10pt;line-height:1.22}
.page{width:8.5in;height:11in;padding:.45in .65in .38in;break-after:page;position:relative}.page:last-child{break-after:auto}
h1{margin:0 0 5pt;font-size:21pt;line-height:1.1;color:#123a52}h2{font-size:11.8pt;margin:10pt 0 5pt;color:#123a52}p{margin:0 0 6pt}
.subtitle{color:#586f7f;font-size:9pt;margin-bottom:11pt}.callout{background:#eaf3f6;border-left:3pt solid #50879c;padding:7pt 9pt;margin:7pt 0;color:#123a52;font-weight:600}
.step{margin-bottom:5pt}.equation{color:#123a52;padding:1pt 10pt 4pt;font-size:11pt}table{width:100%;border-collapse:collapse;margin:5pt 0 8pt;table-layout:fixed;font-size:9pt;line-height:1.18}
th{background:#123a52;color:white;text-align:left}th,td{padding:5pt 6pt;border:.5pt solid #d8e1e7;vertical-align:top}tr:nth-child(even) td{background:#f1f6f8}
.source{font-size:7.7pt;color:#526272;margin-bottom:4pt}.footer{position:absolute;bottom:.2in;left:.65in;right:.65in;border-top:.5pt solid #d8e1e7;padding-top:4pt;color:#6c7a86;font-size:7.5pt;display:flex;justify-content:space-between}
.page:nth-child(2) table{font-size:8.7pt}.page:nth-child(2) td{padding:4.6pt 6pt}.page:nth-child(2) p{font-size:9.2pt}.page:nth-child(2) p.source{font-size:7.7pt}
.page:nth-child(n+3) p{font-size:9.3pt}.page:nth-child(n+3) table{font-size:8.8pt}
"""


def write_html(pages, title):
    html = [f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{escape(title)}</title><style>{CSS}</style><body>']
    for i, page in enumerate(pages, 1):
        html.append('<section class="page">')
        for block in page:
            if block[0] == "table":
                _, headers, rows, widths = block
                html.append('<table><colgroup>'+''.join(f'<col style="width:{w}%">' for w in widths)+'</colgroup><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>')
                for row in rows: html.append('<tr>'+''.join('<td>'+escape(x)+'</td>' for x in row)+'</tr>')
                html.append('</tbody></table>')
            else:
                kind, text = block
                tag = {"title": "h1", "heading": "h2"}.get(kind, "p")
                html.append(f'<{tag} class="{kind}">{escape(text)}</{tag}>')
        html.append(f'<div class="footer"><span>MSE454 · Working plan; physical parameters under review</span><span>{i} / {len(pages)}</span></div></section>')
    html.append('</body></html>')
    path = DOCS/(NAME+".html")
    path.write_text('\n'.join(html))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--browser")
    args = parser.parse_args()
    source = json.loads((DOCS/"project_brief.json").read_text())
    pages = []
    for page in source["pages"]:
        expanded = []
        for block in page:
            expanded.extend(version_record(block[1]) if block[0] == "version_record" else [block])
        pages.append(expanded)
    write_docx(pages, source["title"])
    html = write_html(pages, source["title"])
    if args.pdf:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            launch = {"headless": True, "timeout": 15000, "args": ["--no-sandbox"]}
            if args.browser: launch["executable_path"] = args.browser
            browser = p.chromium.launch(**launch)
            page = browser.new_page(viewport={"width": 816, "height": 1056}, device_scale_factor=1.5)
            page.goto(html.as_uri(), wait_until="load")
            sizes = page.locator('.page').evaluate_all('(pages) => pages.map(p => ({bodyBottom:p.children[p.children.length-2].getBoundingClientRect().bottom-p.getBoundingClientRect().top,footerTop:p.lastElementChild.getBoundingClientRect().top-p.getBoundingClientRect().top}))')
            assert all(size["bodyBottom"] < size["footerTop"]-8 for size in sizes), sizes
            page.pdf(path=str(DOCS/(NAME+".pdf")), print_background=True, prefer_css_page_size=True)
            preview = Path('/tmp/kmc-doc-review')
            preview.mkdir(exist_ok=True)
            for i, section in enumerate(page.locator('.page').all(), 1):
                section.screenshot(path=str(preview/f'page_{i}.png'))
            browser.close()
        print(f"Rendered and checked {len(pages)} PDF pages")
    print(f"Updated {DOCS/(NAME+'.docx')}")


if __name__ == "__main__":
    main()
