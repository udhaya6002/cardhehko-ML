import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

pdf_filename = "deliverables/report.pdf"
doc = SimpleDocTemplate(
    pdf_filename,
    pagesize=letter,
    rightMargin=40,
    leftMargin=40,
    topMargin=40,
    bottomMargin=40
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Heading1"],
    fontSize=20,
    leading=24,
    textColor=colors.HexColor("#1a237e"),
    spaceAfter=8,
    fontName="Helvetica-Bold"
)

subtitle_style = ParagraphStyle(
    "DocSubTitle",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#37474f"),
    spaceAfter=12,
    fontName="Helvetica"
)

h1_style = ParagraphStyle(
    "SectionH1",
    parent=styles["Heading2"],
    fontSize=13,
    leading=17,
    textColor=colors.HexColor("#1565c0"),
    spaceBefore=12,
    spaceAfter=6,
    fontName="Helvetica-Bold",
    keepWithNext=True
)

h2_style = ParagraphStyle(
    "SectionH2",
    parent=styles["Heading3"],
    fontSize=11,
    leading=15,
    textColor=colors.HexColor("#00695c"),
    spaceBefore=8,
    spaceAfter=4,
    fontName="Helvetica-Bold",
    keepWithNext=True
)

body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=9,
    leading=13,
    textColor=colors.HexColor("#263238"),
    spaceAfter=5
)

bullet_style = ParagraphStyle(
    "Bullet",
    parent=body_style,
    leftIndent=15,
    firstLineIndent=-10,
    spaceAfter=3
)

def md_to_rl(text):
    # Strip markdown links
    t = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Bold
    t = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', t)
    # Italic
    t = re.sub(r'\*(.*?)\*', r'<i>\1</i>', t)
    # Code
    t = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', t)
    # XML escape & and < > unless in tags
    t = t.replace("& ", "&amp; ")
    return t

story = []

story.append(Paragraph("Used Car Price Prediction: A Robust, Anti-Leakage Supervised Regression Framework", title_style))
story.append(Paragraph("<b>Course Project:</b> Machine Learning | <b>Institution:</b> Vellore Institute of Technology, Andhra Pradesh (VIT-AP)<br/><b>Dataset:</b> CarDekho Used Car Corpus | <b>Date:</b> October 2026 | <b>Status:</b> Completed Deliverable", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a237e"), spaceAfter=12))

with open("docs/report_draft.md", "r", encoding="utf-8") as f:
    text = f.read()

sections = text.split("\n## ")

for sec in sections[1:]:
    lines = sec.split("\n")
    sec_title = md_to_rl(lines[0].strip())
    sec_content = "\n".join(lines[1:]).strip()

    story.append(Paragraph(sec_title, h1_style))

    sub_lines = sec_content.split("\n")
    table_buffer = []
    in_table = False

    def flush_table():
        if not table_buffer:
            return
        t_data = []
        for t_row in table_buffer:
            cells = [Paragraph(md_to_rl(c.strip()), body_style) for c in t_row.split("|")[1:-1]]
            if cells:
                t_data.append(cells)
        if t_data:
            col_w = (letter[0] - 80) / len(t_data[0])
            t = Table(t_data, colWidths=[col_w]*len(t_data[0]))
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e8eaf6")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#1a237e")),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#b0bec5")),
            ]))
            story.append(t)
            story.append(Spacer(1, 6))

    for l in sub_lines:
        s_strip = l.strip()
        if not s_strip:
            if in_table:
                flush_table()
                table_buffer = []
                in_table = False
            continue

        if s_strip.startswith("|") and s_strip.endswith("|"):
            if "---" in s_strip:
                continue
            in_table = True
            table_buffer.append(s_strip)
            continue

        if in_table:
            flush_table()
            table_buffer = []
            in_table = False

        if s_strip.startswith("### "):
            story.append(Paragraph(md_to_rl(s_strip.replace("### ", "")), h2_style))
        elif s_strip.startswith("- ") or s_strip.startswith("* "):
            story.append(Paragraph(f"• {md_to_rl(s_strip[2:])}", bullet_style))
        elif re.match(r'^\d+\.\s', s_strip):
            story.append(Paragraph(md_to_rl(s_strip), bullet_style))
        else:
            story.append(Paragraph(md_to_rl(s_strip), body_style))

    if in_table:
        flush_table()
        table_buffer = []
        in_table = False

    story.append(Spacer(1, 6))

    # Add images to specific sections
    if "11. System and Methodology Architecture" in sec_title and os.path.exists("figures/architecture.png"):
        story.append(Image("figures/architecture.png", width=520, height=290))
        story.append(Spacer(1, 8))
    elif "13. Experimental Results" in sec_title and os.path.exists("figures/eval_model_comparison.png"):
        story.append(Image("figures/eval_model_comparison.png", width=520, height=180))
        story.append(Spacer(1, 8))

doc.build(story)
print(f"Compiled {pdf_filename} successfully!")
