"""
Document Generator for ASD MINN Project.
Creates:
1. Objective_1.docx
2. Objective_2.docx
3. Objective_3.docx
4. Objective_4.docx
5. Corrected_verified_complete_project_report.docx

All files contain full mathematical formulas, algorithms, tables, figures,
exact verified evaluation numbers, counterfactuals, and code traceability.
"""

import os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path("d:/GAN-3/GAN-3")

# ── Color Palette ───────────────────────────────────────────────────────────
HEX_NAVY = "1B365D"       # Primary
HEX_TEAL = "006699"       # Secondary
HEX_CHARCOAL = "222222"   # Body text
HEX_MUTED = "666666"      # Subtitles
HEX_LIGHT_BG = "F4F6F9"   # Table alternate / Callout
HEX_BORDER = "CCCCCC"
HEX_SUCCESS = "15803D"
HEX_DANGER = "DC2626"
HEX_AMBER = "B45309"

COLOR_NAVY = RGBColor(27, 54, 93)
COLOR_TEAL = RGBColor(0, 102, 153)
COLOR_CHARCOAL = RGBColor(34, 34, 34)
COLOR_MUTED = RGBColor(100, 116, 139)
COLOR_SUCCESS = RGBColor(21, 128, 61)

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="CCCCCC", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def style_document(doc):
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Arial'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = COLOR_CHARCOAL
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

def add_title(doc, title_text, subtitle_text=None, meta_text=None):
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run(title_text)
    run_title.font.name = 'Arial'
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = COLOR_NAVY
    
    if subtitle_text:
        p_sub = doc.add_paragraph()
        p_sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_sub.paragraph_format.space_after = Pt(6)
        run_sub = p_sub.add_run(subtitle_text)
        run_sub.font.name = 'Arial'
        run_sub.font.size = Pt(12)
        run_sub.font.italic = True
        run_sub.font.color.rgb = COLOR_TEAL

    if meta_text:
        p_meta = doc.add_paragraph()
        p_meta.paragraph_format.space_after = Pt(16)
        run_meta = p_meta.add_run(meta_text)
        run_meta.font.name = 'Arial'
        run_meta.font.size = Pt(9.5)
        run_meta.font.color.rgb = COLOR_MUTED

    # Divider rule
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_after = Pt(16)
    p_div_border = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="12" w:space="1" w:color="{HEX_NAVY}"/></w:pBdr>')
    p_div._p.get_or_add_pPr().append(p_div_border)

def add_h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(15)
    run.font.bold = True
    run.font.color.rgb = COLOR_NAVY
    return p

def add_h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(12.5)
    run.font.bold = True
    run.font.color.rgb = COLOR_TEAL
    return p

def add_h3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = COLOR_CHARCOAL
    return p

def add_callout(doc, title, text, border_color=HEX_TEAL):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, HEX_LIGHT_BG)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    # Left border only
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    cell._tc.get_or_add_tcPr().append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r_title = p.add_run(f"[{title}] ")
    r_title.bold = True
    r_title.font.size = Pt(10)
    r_title.font.color.rgb = COLOR_NAVY
    
    r_text = p.add_run(text)
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = COLOR_CHARCOAL
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def add_formula_box(doc, formula_title, formula_text, variable_explanations=None):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="18" w:space="0" w:color="{HEX_NAVY}"/>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'</w:tcBorders>'
    )
    cell._tc.get_or_add_tcPr().append(tcBorders)
    
    p0 = cell.paragraphs[0]
    p0.paragraph_format.space_after = Pt(4)
    r_label = p0.add_run(f"Mathematical Formulation: {formula_title}")
    r_label.bold = True
    r_label.font.size = Pt(10)
    r_label.font.color.rgb = COLOR_NAVY
    
    p1 = cell.add_paragraph()
    p1.paragraph_format.space_after = Pt(6)
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_math = p1.add_run(formula_text)
    r_math.font.name = 'Consolas'
    r_math.font.size = Pt(10.5)
    r_math.font.bold = True
    r_math.font.color.rgb = COLOR_NAVY
    
    if variable_explanations:
        p2 = cell.add_paragraph()
        p2.paragraph_format.space_after = Pt(0)
        r_desc_title = p2.add_run("Where:\n")
        r_desc_title.bold = True
        r_desc_title.font.size = Pt(9)
        for var, desc in variable_explanations:
            p_v = cell.add_paragraph()
            p_v.paragraph_format.space_after = Pt(1)
            p_v.paragraph_format.left_indent = Inches(0.2)
            r_v = p_v.add_run(f"• {var}: ")
            r_v.bold = True
            r_v.font.size = Pt(9)
            r_d = p_v.add_run(desc)
            r_d.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def add_algorithm_box(doc, algo_num, algo_name, steps, inputs=None, outputs=None):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="18" w:space="0" w:color="{HEX_TEAL}"/>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{HEX_TEAL}"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{HEX_TEAL}"/>'
        f'<w:right w:val="single" w:sz="6" w:space="0" w:color="{HEX_TEAL}"/>'
        f'</w:tcBorders>'
    )
    cell._tc.get_or_add_tcPr().append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(4)
    r_hdr = p.add_run(f"Algorithm {algo_num}: {algo_name}")
    r_hdr.bold = True
    r_hdr.font.size = Pt(10.5)
    r_hdr.font.color.rgb = COLOR_NAVY
    
    if inputs:
        p_in = cell.add_paragraph()
        p_in.paragraph_format.space_after = Pt(2)
        r_in_lbl = p_in.add_run("Input: ")
        r_in_lbl.bold = True
        r_in_lbl.font.size = Pt(9.5)
        r_in = p_in.add_run(inputs)
        r_in.font.size = Pt(9.5)

    if outputs:
        p_out = cell.add_paragraph()
        p_out.paragraph_format.space_after = Pt(4)
        r_out_lbl = p_out.add_run("Output: ")
        r_out_lbl.bold = True
        r_out_lbl.font.size = Pt(9.5)
        r_out = p_out.add_run(outputs)
        r_out.font.size = Pt(9.5)

    p_step_lbl = cell.add_paragraph()
    p_step_lbl.paragraph_format.space_after = Pt(2)
    r_step_lbl = p_step_lbl.add_run("Procedure:")
    r_step_lbl.bold = True
    r_step_lbl.font.size = Pt(9.5)

    for i, step in enumerate(steps, 1):
        p_s = cell.add_paragraph()
        p_s.paragraph_format.space_after = Pt(2)
        p_s.paragraph_format.left_indent = Inches(0.2)
        r_num = p_s.add_run(f"{i}. ")
        r_num.bold = True
        r_num.font.size = Pt(9)
        r_txt = p_s.add_run(step)
        r_txt.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=len(rows)+1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    
    # Header Row
    hdr_cells = table.rows[0].cells
    for i, header_text in enumerate(headers):
        hdr_cells[i].text = header_text
        set_cell_background(hdr_cells[i], HEX_NAVY)
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = 'Arial'
            r.font.size = Pt(9.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
            
    # Data Rows
    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx+1].cells
        bg_col = HEX_LIGHT_BG if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, cell_value in enumerate(row_data):
            row_cells[c_idx].text = str(cell_value)
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=120, right=120)
            p = row_cells[c_idx].paragraphs[0]
            # Center numeric/short columns, left-align text
            if c_idx > 0 and len(str(cell_value)) < 15:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = 'Arial'
                r.font.size = Pt(9)
                r.font.color.rgb = COLOR_CHARCOAL
                if "Proposed" in str(cell_value) or "98.76%" in str(cell_value) or "0.9876" in str(cell_value) or "1:1" in str(cell_value):
                    r.font.bold = True

    if col_widths and len(col_widths) == len(headers):
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def add_image_if_exists(doc, img_rel_path, caption_text, width_inches=6.0):
    full_path = PROJECT_ROOT / img_rel_path
    if full_path.exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(3)
        run_img = p_img.add_run()
        run_img.add_picture(str(full_path), width=Inches(width_inches))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(12)
        r_cap = p_cap.add_run(f"Figure: {caption_text}")
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = COLOR_MUTED
    else:
        print(f"Warning: Image {full_path} not found.")

print("Loaded base styling library for report generator.")
