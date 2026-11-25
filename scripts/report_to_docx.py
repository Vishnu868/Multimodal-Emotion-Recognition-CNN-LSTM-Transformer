"""Convert REPORT.md (headings, paragraphs, lists, tables, images, code) into a Word document.

    python scripts/report_to_docx.py REPORT.md Project_Report.docx
"""
import os
import re
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


def add_inline(par, text):
    """**bold**, *italic*, `code` and [links](url) -> runs."""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    for tok in re.split(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("`"):
            r = par.add_run(tok[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        elif tok.startswith("*") and len(tok) > 2:
            par.add_run(tok[1:-1]).italic = True
        else:
            par.add_run(tok)


def main(src, dst):
    base = os.path.dirname(os.path.abspath(src))
    lines = open(src, encoding="utf-8").read().splitlines()
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):                                     # code block (mermaid skipped)
            lang = ln[3:].strip()
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            if lang != "mermaid":
                p = doc.add_paragraph()
                r = p.add_run("\n".join(block))
                r.font.name = "Consolas"
                r.font.size = Pt(9)
        elif ln.startswith("#"):
            level = len(ln) - len(ln.lstrip("#"))
            h = doc.add_heading(ln.lstrip("#").strip(), level=0 if level == 1 else min(level - 1, 3))
            if level == 1:
                h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif ln.startswith("!["):                                    # image
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", ln)
            path = os.path.join(base, m.group(2))
            if os.path.exists(path):
                doc.add_picture(path, width=Inches(6.3))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap.add_run(m.group(1)).italic = True
        elif ln.startswith("|"):                                     # table
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[\s:|-]+\|$", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            i -= 1
            t = doc.add_table(rows=len(rows), cols=len(rows[0]))
            t.style = "Light Grid Accent 1"
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for r, row in enumerate(rows):
                for c, cell in enumerate(row[: len(rows[0])]):
                    par = t.cell(r, c).paragraphs[0]
                    add_inline(par, cell)
                    if r == 0:
                        for run in par.runs:
                            run.bold = True
            doc.add_paragraph()
        elif re.match(r"^\s*([-*]|\d+\.)\s", ln):                    # list item
            numbered = bool(re.match(r"^\s*\d+\.", ln))
            indent = (len(ln) - len(ln.lstrip())) // 2
            style = ("List Number" if numbered else "List Bullet") + (" 2" if indent else "")
            add_inline(doc.add_paragraph(style=style), re.sub(r"^\s*([-*]|\d+\.)\s", "", ln))
        elif ln.startswith(">"):
            p = doc.add_paragraph(style="Intense Quote")
            add_inline(p, ln.lstrip("> "))
        elif ln.strip() in ("---", ""):
            pass
        else:                                                        # paragraph (join wrapped lines)
            para = [ln]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||!\[|```|>|\s*([-*]|\d+\.)\s)",
                                                                              lines[i + 1]):
                i += 1
                para.append(lines[i])
            add_inline(doc.add_paragraph(), " ".join(para))
        i += 1
    doc.save(dst)
    print("wrote", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
