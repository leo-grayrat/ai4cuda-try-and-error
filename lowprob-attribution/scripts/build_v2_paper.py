"""Build the v2 reading copy, LaTeX PDF and DOCX from stage2_write/paper.md."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "v2/stage2_write/paper.md"
FINAL = ROOT / "v2/stage5_finalize"
MIKTEX = Path(os.environ["LOCALAPPDATA"]) / "Programs/MiKTeX/miktex/bin/x64"
PANDOC = shutil.which("pandoc")


def run(args: list[str], cwd: Path = FINAL) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed ({result.returncode}):\n{result.stdout[-5000:]}\n{result.stderr[-5000:]}")
    return result.stdout


def main() -> None:
    if not PANDOC:
        raise RuntimeError("pandoc was not found")
    text = SOURCE.read_text(encoding="utf-8")
    ast = json.loads(run([PANDOC, str(SOURCE), "-t", "json"]))
    headers = [b["c"] for b in ast["blocks"] if b["t"] == "Header"]
    labels = {}
    numbering = []
    major = minor = 0
    for level, attributes, inlines in headers:
        slug = attributes[0]
        if level == 1:
            numbering.append("")
        elif level == 2:
            if slug == "abstract" or slug.startswith("appendix-") or slug == "references":
                numbering.append("")
                labels[slug] = "Appendix " + slug.split("-")[1].upper() if slug.startswith("appendix-") else slug.title()
            else:
                major += 1
                minor = 0
                labels[slug] = str(major)
                numbering.append(str(major))
        elif level == 3:
            minor += 1
            labels[slug] = f"{major}.{minor}"
            numbering.append(f"{major}.{minor}")
        else:
            raise RuntimeError(f"Unsupported heading depth: {level}")
    figure_ids = re.findall(r"\{#(fig:[^}\s]+)[^}]*\}", text)
    labels.update({name: str(index) for index, name in enumerate(figure_ids, 1)})
    unknown = sorted(set(re.findall(r"\\ref\{([^}]+)\}", text)) - labels.keys())
    if unknown:
        raise RuntimeError(f"Unresolved source cross-references: {unknown}")
    text = re.sub(r"Section~\\ref\{([^}]+)\}", lambda m: "Section " + labels[m[1]], text)
    text = re.sub(r"Figure~\\ref\{([^}]+)\}", lambda m: "Figure " + labels[m[1]], text)
    text = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels[m[1]], text)
    heading_index = 0
    output = []
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        match = re.match(r"^(#{1,6}) (.+)$", line) if not fenced else None
        if match:
            number = numbering[heading_index]
            heading_index += 1
            if number:
                line = match[1] + " " + number + ". " + match[2]
        output.append(line)
    if heading_index != len(headers):
        raise RuntimeError("Heading numbering did not match the document")
    (FINAL / "paper.md").write_text("\n".join(output) + "\n", encoding="utf-8")
    common = [PANDOC, "paper.md", "--standalone", "--bibliography=references.bib"]
    run(common + ["--natbib", "-V", "natbiboptions=numbers,sort&compress", "-V", "biblio-style=unsrtnat",
                  "-V", "geometry=margin=2.5cm", "-V", "fontsize=11pt", "-o", "main.tex"])
    # Pandoc 2.12 rounds table fractions to two decimals; their sum can exceed
    # one and push an otherwise fitting table beyond the text width.
    tex_path = FINAL / "main.tex"
    tex = tex_path.read_text(encoding="utf-8")
    # Compilation runs in build/ to isolate auxiliary files; also search paths
    # relative to the directory containing main.tex for linked figures.
    tex = tex.replace("\\begin{document}", "\\graphicspath{{../}}\n\\begin{document}")
    for name in figure_ids:
        tex = re.sub(r"\bFigure " + labels[name] + r"\b",
                     lambda _: "Figure~\\ref{" + name + "}", tex)
    # Use vector plots in the PDF while the DOCX embeds the high-resolution PNGs.
    tex = re.sub(r"(\{\.\./figures/fig[^}]+)\.png\}", r"\1.pdf}", tex)
    def normalise_table_width(match: re.Match[str]) -> str:
        header = match[0]
        fractions = re.findall(r"\\real\{([0-9.]+)\}", header)
        total = sum(float(value) for value in fractions)
        if total > 1.00001:
            header = re.sub(r"\\real\{([0-9.]+)\}",
                            lambda value: "\\real{" + f"{float(value[1]) / total:.6f}" + "}", header)
        return header
    tex = re.sub(r"\\begin\{longtable\}.*?\\toprule", normalise_table_width, tex, flags=re.S)
    # These are short comparison tables. Keep each table together so that its
    # categories or total row cannot be stranded on the following page.
    def keep_table_together(match: re.Match[str]) -> str:
        body = match[2].replace("\\endhead", "")
        return ("\\begin{center}\n\\begin{minipage}{\\linewidth}\n\\centering\n"
                "\\begin{tabular}{" + match[1] + "\n\\toprule\n" + body
                + "\\end{tabular}\n\\end{minipage}\n\\end{center}")
    tex = re.sub(r"\\begin\{longtable\}\[\]\{(.*?)\n\\toprule\n(.*?)\\end\{longtable\}",
                 keep_table_together, tex, flags=re.S)
    tex = tex.replace("\\bibliography{references.bib}",
                      "\\begingroup\n\\interlinepenalty=10000\n"
                      "\\bibliography{references.bib}\n\\endgroup")
    tex_path.write_text(tex, encoding="utf-8")
    run(common + ["--citeproc", "--csl=numeric-bracket.csl",
                  "--metadata=reference-section-title:References", "-o", "paper.docx"])
    docx_path = FINAL / "paper.docx"
    with zipfile.ZipFile(docx_path) as archive:
        contents = [(info, archive.read(info.filename)) for info in archive.infolist()]
    word_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ET.register_namespace("w", word_ns)
    tag = lambda name: "{" + word_ns + "}" + name
    for index, (info, data) in enumerate(contents):
        if info.filename != "word/document.xml":
            continue
        document = ET.fromstring(data)
        section = document.find(".//" + tag("sectPr"))
        if section is not None:
            page_size = ET.SubElement(section, tag("pgSz"))
            page_size.set(tag("w"), "11906")
            page_size.set(tag("h"), "16838")
            margins = ET.SubElement(section, tag("pgMar"))
            for edge in ("top", "bottom", "left", "right"):
                margins.set(tag(edge), "1417")
        body_width = 9072
        for table in document.iter(tag("tbl")):
            rows = table.findall(tag("tr"))
            column_headers = ["".join(cell.itertext()) for cell in rows[0].findall(tag("tc"))]
            count = len(column_headers)
            # Pandoc 2.12 may emit an empty grid and a zero-percent table width
            # for pipe tables. Give every column an explicit physical width.
            weights = [1 / count] * count
            first = column_headers[0]
            if "Question" in first:
                weights = [.12, .88]
            elif "Component" in first:
                weights = [.38, .44, .18]
            elif "Task" in first:
                weights = {2: [.54, .46], 3: [.58, .21, .21], 4: [.55, .12, .13, .20],
                           5: [.49, .12, .13, .13, .13]}[count]
            elif "category" in first:
                weights = [.30, .12, .58]
            elif "Boundary" in first:
                weights = [.13, .20, .13, .19, .16, .19]
            elif "Saved collection" in first:
                weights = [.19, .14, .20, .23, .24]
            elif "Label assigned" in first:
                weights = [.28, .18, .27, .27]
            widths = [int(body_width * weight) for weight in weights]
            widths[-1] = body_width - sum(widths[:-1])
            properties = table.find(tag("tblPr"))
            total_width = properties.find(tag("tblW"))
            total_width.set(tag("type"), "dxa")
            total_width.set(tag("w"), str(body_width))
            layout = ET.SubElement(properties, tag("tblLayout"))
            layout.set(tag("type"), "fixed")
            grid = table.find(tag("tblGrid"))
            grid.clear()
            for width in widths:
                ET.SubElement(grid, tag("gridCol"), {tag("w"): str(width)})
            for row in rows:
                for cell, width in zip(row.findall(tag("tc")), widths):
                    cell_properties = cell.find(tag("tcPr"))
                    if cell_properties is None:
                        cell_properties = ET.Element(tag("tcPr"))
                        cell.insert(0, cell_properties)
                    cell_width = cell_properties.find(tag("tcW"))
                    if cell_width is None:
                        cell_width = ET.SubElement(cell_properties, tag("tcW"))
                    cell_width.set(tag("type"), "dxa")
                    cell_width.set(tag("w"), str(width))
            for row in list(table.findall(tag("tr")))[:-1]:
                for paragraph in row.iter(tag("p")):
                    properties = paragraph.find(tag("pPr"))
                    if properties is None:
                        properties = ET.Element(tag("pPr"))
                        paragraph.insert(0, properties)
                    if properties.find(tag("keepNext")) is None:
                        ET.SubElement(properties, tag("keepNext"))
        figure_number = 0
        for paragraph in document.iter(tag("p")):
            properties = paragraph.find(tag("pPr"))
            style = properties.find(tag("pStyle")) if properties is not None else None
            if style is not None and style.get(tag("val")) == "ImageCaption":
                figure_number += 1
                prefix = ET.Element(tag("r"))
                text_node = ET.SubElement(prefix, tag("t"))
                text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                text_node.text = f"Figure {figure_number}: "
                paragraph.insert(1, prefix)
        contents[index] = (info, ET.tostring(document, encoding="utf-8", xml_declaration=True))
    with zipfile.ZipFile(docx_path, "w") as archive:
        for info, data in contents:
            archive.writestr(info, data)
    build = FINAL / "build"
    build.mkdir(exist_ok=True)
    os.environ["PATH"] = str(MIKTEX) + os.pathsep + os.environ.get("PATH", "")
    latex = [str(MIKTEX / "xelatex.exe"), "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "../main.tex"]
    # BibTeX resolves the bibliography in build/main.aux; keep a local copy of
    # the existing bibliography so the build has no external search dependency.
    shutil.copy2(FINAL / "references.bib", build / "references.bib")
    run(latex, cwd=build)
    run([str(MIKTEX / "bibtex.exe"), "main"], cwd=build)
    run(latex, cwd=build)
    run(latex, cwd=build)
    log = (build / "main.log").read_text(encoding="utf-8", errors="replace")
    failures = [line for line in log.splitlines() if "undefined" in line.lower() or line.startswith("! ") or "Missing character:" in line]
    if failures:
        raise RuntimeError("LaTeX checks failed:\n" + "\n".join(failures))
    shutil.copy2(build / "main.pdf", FINAL / "main.pdf")
    shutil.copy2(build / "main.log", FINAL / "main.log")
    summary = {"source": str(SOURCE.relative_to(ROOT)), "headings": len(headers),
               "figures": len(figure_ids),
               "local_tool_section": labels["local-tool-call-capture"],
               "local_tool_discussion": labels["complementary-coverage-of-reply-text-and-tool-parameters"],
               "overfull_boxes": log.count("Overfull "), "underfull_boxes": log.count("Underfull "),
               "pdf_bytes": (FINAL / "main.pdf").stat().st_size,
               "docx_bytes": (FINAL / "paper.docx").stat().st_size}
    (FINAL / "local-tool-build.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
