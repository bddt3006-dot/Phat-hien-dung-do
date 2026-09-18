# -*- coding: utf-8 -*-
"""
Script to generate 7 separate, comprehensive project reports following the 7-step software process
Output: 7 .docx files and 7 .md files in 'Bao_cao_7_buoc/'
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUTPUT_DIR = os.path.join(os.getcwd(), 'Bao_cao_7_buoc')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_base_doc():
    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
    
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    style.paragraph_format.line_spacing = 1.15
    style.paragraph_format.space_after = Pt(4)
    return doc

def add_national_header(doc, place_date="Hà Nội, ngày 16 tháng 09 năm 2026"):
    tbl = doc.add_table(rows=1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    c1 = tbl.cell(0, 0)
    p1 = c1.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p1.add_run("BỘ GIÁO DỤC VÀ ĐÀO TẠO\nDỰ ÁN PHÁT TRIỂN PHẦN MỀM AI")
    r1.font.name = 'Times New Roman'
    r1.font.size = Pt(10)
    r1.font.bold = True
    
    c2 = tbl.cell(0, 2 if False else 1)
    p2 = c2.paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập – Tự do – Hạnh phúc\n-------------------")
    r2.font.name = 'Times New Roman'
    r2.font.size = Pt(10)
    r2.font.bold = True
    
    p_date = doc.add_paragraph()
    p_date.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_date = p_date.add_run(f"*{place_date}*")
    r_date.font.name = 'Times New Roman'
    r_date.font.size = Pt(11)
    r_date.font.italic = True
    doc.add_paragraph()

def add_title_block(doc, step_label, doc_title, subtitle="Hệ thống Phát hiện Phương tiện Dừng Đỗ Trái Phép trên Luồng Video Giám sát"):
    p_step = doc.add_paragraph()
    p_step.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_step = p_step.add_run(step_label.upper())
    r_step.font.name = 'Times New Roman'
    r_step.font.size = Pt(13)
    r_step.font.bold = True
    r_step.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run(doc_title.upper())
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(16)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x0F, 0x29, 0x42)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run(f"Dự án: {subtitle}")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(12)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)
    doc.add_paragraph()

def add_h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

def add_h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)

def add_h3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.italic = True
    r.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

def add_p(doc, text, bold_prefix=""):
    p = doc.add_paragraph()
    if bold_prefix:
        r_b = p.add_run(bold_prefix + " ")
        r_b.font.name = 'Times New Roman'
        r_b.font.bold = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'

def add_bullet(doc, text, bold_prefix=""):
    p = doc.add_paragraph(style='List Bullet')
    if bold_prefix:
        r_b = p.add_run(bold_prefix + " ")
        r_b.font.name = 'Times New Roman'
        r_b.font.bold = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'

def add_table(doc, headers, rows):
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    
    # Header row
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        cell = hdr_cells[i]
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.bold = True
        r.font.size = Pt(11)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F7FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row_cells[c_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            p = cell.paragraphs[0]
            if c_idx == 0 or len(val) <= 15:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(val))
            r.font.name = 'Times New Roman'
            r.font.size = Pt(11)
            
    doc.add_paragraph()

def save_both(doc, md_content, basename):
    docx_path = os.path.join(OUTPUT_DIR, f"{basename}.docx")
    md_path = os.path.join(OUTPUT_DIR, f"{basename}.md")
    
    doc.save(docx_path)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"-> Generated: {basename}.docx and {basename}.md")

print("Module initialized successfully.")
