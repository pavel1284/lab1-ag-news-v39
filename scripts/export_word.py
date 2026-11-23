"""Export LAB1 Markdown report into a readable Word document."""
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/LAB1/lab1_report.md"
OUTPUT = ROOT / "reports/LAB1/lab1_report.docx"


def cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def add_formatted(paragraph, text):
    parts = re.split(r"(\*\*.*?\*\*|`[^`]+`)", text)
    for item in parts:
        if not item:
            continue
        is_bold = item.startswith("**") and item.endswith("**")
        is_code = item.startswith("`") and item.endswith("`")
        if is_bold:
            item = item[2:-2]
        elif is_code:
            item = item[1:-1]
        part = paragraph.add_run(item)
        if is_bold:
            part.bold = True
        if is_code:
            part.font.name = "DejaVu Sans Mono"
            part.font.size = Pt(8.5)
            part.font.color.rgb = RGBColor(48, 82, 111)


def add_table(doc, rows):
    cells = [[value.strip() for value in line.strip().strip("|").split("|")]
             for line in rows if not re.fullmatch(r"\s*\|[-|:\s]+\|\s*", line)]
    if not cells:
        return
    width = max(len(r) for r in cells)
    t = doc.add_table(rows=1, cols=width)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    t.autofit = True
    for j in range(width):
        c = t.rows[0].cells[j]
        c.text = cells[0][j] if j < len(cells[0]) else ""
        cell_shading(c, "214B68")
        for paragraph in c.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(8.8)
    for i, values in enumerate(cells[1:]):
        row = t.add_row()
        for j, cell in enumerate(row.cells):
            cell.text = values[j].replace("**", "") if j < len(values) else ""
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if i % 2:
                cell_shading(cell, "EDF2F6")
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    run.font.size = Pt(8.7)
    doc.add_paragraph("")


def main() -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.1)
    sec.bottom_margin = Cm(2)
    sec.left_margin = Cm(2.3)
    sec.right_margin = Cm(1.9)
    style = doc.styles["Normal"]
    style.font.name = "DejaVu Sans"
    style.font.size = Pt(10)
    style.font.color.rgb = RGBColor(38, 49, 58)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.17
    for name, size, color in [("Title", 20, RGBColor(25, 58, 84)),
                              ("Heading 1", 14, RGBColor(25, 68, 99)),
                              ("Heading 2", 11.5, RGBColor(30, 83, 120))]:
        s = doc.styles[name]
        s.font.name = "DejaVu Sans"
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = color
    header = sec.header.paragraphs[0]
    header.text = "ЛАБОРАТОРНАЯ РАБОТА №1   /   AG NEWS   /   ВАРИАНТ 39"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(100, 121, 139)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.text = "AG News · локально проверенная сборка     |     стр. "
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    lines = INPUT.read_text(encoding="utf-8").splitlines()
    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            p = doc.add_paragraph(style="Normal")
            p.paragraph_format.left_indent = Cm(0.55)
            p.paragraph_format.space_after = Pt(1)
            rr = p.add_run(lines[i])
            rr.font.name = "DejaVu Sans Mono"
            rr.font.size = Pt(8)
            i += 1
            continue
        if not line:
            i += 1
            continue
        img = re.fullmatch(r"(?:\d+\.\s*)?!\[(.*?)\]\((.*?)\)", line)
        if img:
            dest = (INPUT.parent / img.group(2)).resolve()
            if dest.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(dest), width=Cm(15.9))
            i += 1
            continue
        if line.startswith("|"):
            group = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                group.append(lines[i]); i += 1
            add_table(doc, group)
            continue
        if line.startswith("# "):
            p = doc.add_paragraph(style="Title")
            add_formatted(p, line[2:])
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=1)
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=2)
        elif re.match(r"^\d+\.\s", line):
            p = doc.add_paragraph(style="List Paragraph")
            add_formatted(p, line)
        elif line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_formatted(p, line[2:])
        else:
            p = doc.add_paragraph()
            add_formatted(p, line)
        i += 1
    doc.core_properties.title = "ЛР1 — AG News: темы, вариант 39"
    doc.core_properties.subject = "EDA AG News и контроль Git"
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
