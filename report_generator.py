"""Downloadable Field Diagnosis PDF Report Generator.

Generates an official, publication-quality agricultural diagnostic card:
- Side-by-side display of original field photo and AI attention attribution heatmap.
- Calibrated prediction, confidence, and top-3 candidate ranking.
- Verified organic, chemical, and preventive management advisories.
- Regulatory disclaimer: Preliminary AI-assisted advisory.
"""

from pathlib import Path
from datetime import datetime
import io
import hashlib
from typing import Dict, Any, List, Optional
from PIL import Image

# Python 3.8 OpenSSL compatibility patch for reportlab
if not hasattr(hashlib, "_orig_md5_unpatched"):
    hashlib._orig_md5_unpatched = hashlib.md5
    def _safe_md5(*args, **kwargs):
        kwargs.pop("usedforsecurity", None)
        return hashlib._orig_md5_unpatched(*args, **kwargs)
    hashlib.md5 = _safe_md5

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def generate_pdf_report(
    farmer_or_plot_id: str,
    crop: str,
    condition_display: str,
    category: str,
    severity: str,
    confidence: float,
    top_predictions: List[tuple],
    organic_remedies: List[str],
    chemical_treatments: List[str],
    prevention_tips: List[str],
    original_image: Optional[Image.Image] = None,
    heatmap_image: Optional[Image.Image] = None,
) -> bytes:
    """Builds and returns an official PDF diagnostic card in memory as bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1b4332"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#2d6a4f"),
        spaceAfter=12,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#081c15"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#222222"),
    )
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2b2b2b"),
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#555555"),
        alignment=1,  # Center
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("🌾 RiceMaize AI — Field Diagnostic & Advisory Card", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
        f"Plot / Field ID: <b>{farmer_or_plot_id or 'General Farm'}</b> &nbsp;|&nbsp; "
        f"Crop: <b>{crop}</b>",
        subtitle_style,
    ))
    story.append(Spacer(1, 4))

    # 2. Side-by-Side Images (Original + Heatmap)
    img_cells = []
    cell_w, cell_h = 2.4 * inch, 2.4 * inch

    if original_image is not None:
        orig_buf = io.BytesIO()
        orig_copy = original_image.copy()
        orig_copy.thumbnail((400, 400))
        orig_copy.save(orig_buf, format="JPEG", quality=85)
        orig_buf.seek(0)
        img_cells.append([
            RLImage(orig_buf, width=cell_w, height=cell_h),
            Paragraph("<b>Original Uploaded Leaf</b>", body_style),
        ])

    if heatmap_image is not None:
        heat_buf = io.BytesIO()
        heat_copy = heatmap_image.copy()
        heat_copy.thumbnail((400, 400))
        heat_copy.save(heat_buf, format="JPEG", quality=85)
        heat_buf.seek(0)
        img_cells.append([
            RLImage(heat_buf, width=cell_w, height=cell_h),
            Paragraph("<b>AI Attention Focus Heatmap</b>", body_style),
        ])

    if len(img_cells) == 2:
        img_table = Table(
            [[img_cells[0][0], img_cells[1][0]], [img_cells[0][1], img_cells[1][1]]],
            colWidths=[2.6 * inch, 2.6 * inch],
        )
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8))

    # 3. Diagnosis Summary Box
    sev_bg = "#d8f3dc" if severity == "Healthy" else ("#ffd6d6" if severity == "Critical" else "#ffe6a7")
    diag_data = [
        [
            Paragraph(f"<b>Diagnosed Condition:</b><br/><font size=12><b>{condition_display}</b></font>", body_style),
            Paragraph(f"<b>Nature:</b> {category}<br/><b>Severity:</b> {severity}", body_style),
            Paragraph(f"<b>Calibrated Confidence:</b><br/><font size=12><b>{confidence:.1f}%</b></font>", body_style),
        ]
    ]
    diag_table = Table(diag_data, colWidths=[2.5 * inch, 2.5 * inch, 2.0 * inch])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(sev_bg)),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#2d6a4f")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 8))

    # 4. Top-3 Candidate Predictions
    if top_predictions:
        top_str = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join([
            f"<b>{i+1}. {c.split('/')[-1].replace('_', ' ')}</b>: {p:.1f}%"
            for i, (c, p) in enumerate(top_predictions[:3])
        ])
        story.append(Paragraph(f"<b>Differential Model Candidates:</b> {top_str}", body_style))
        story.append(Spacer(1, 8))

    # 5. Actionable Advisory Sections
    story.append(Paragraph("🌿 Organic & Biological Remedies", section_heading))
    for r in organic_remedies[:3]:
        clean_r = r.replace("<b>", "").replace("</b>", "")
        story.append(Paragraph(f"• {clean_r}", bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("🧪 Recommended Chemical Treatments", section_heading))
    for c in chemical_treatments[:3]:
        clean_c = c.replace("<b>", "").replace("</b>", "")
        story.append(Paragraph(f"• {clean_c}", bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("🚜 Agronomic Prevention & Field Recovery", section_heading))
    for p in prevention_tips[:3]:
        clean_p = p.replace("<b>", "").replace("</b>", "")
        story.append(Paragraph(f"• {clean_p}", bullet_style))

    # 6. Safety & Official Disclaimer
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "<b>OFFICIAL REGULATORY DISCLAIMER:</b> This document contains an automated decision-support assessment generated via computer vision. "
        "It provides preliminary guidance and does not constitute absolute microbiological or laboratory proof of disease. "
        "Before applying large-scale chemical pesticides, farmers must confirm diagnoses with their local Krishi Vigyan Kendra (KVK) "
        "or Agricultural Extension Officer. Always follow chemical label rates and wear protective safety equipment.",
        disclaimer_style,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def test_pdf_build():
    """Unit test generating a sample PDF report in memory."""
    from PIL import Image
    test_img = Image.new("RGB", (200, 200), color=(80, 180, 100))
    pdf_bytes = generate_pdf_report(
        farmer_or_plot_id="Plot-A4",
        crop="Rice (Paddy)",
        condition_display="Brown Spot Disease",
        category="Fungal Disease (Helminthosporium oryzae)",
        severity="Medium",
        confidence=96.5,
        top_predictions=[("Rice/Disease/02_Brown_spot", 96.5), ("Rice/Disease/01_Bacterial_leaf_blight", 2.1)],
        organic_remedies=["Apply 5% Neem Seed Kernel Extract (NSKE).", "Seed treatment with Trichoderma asperellum."],
        chemical_treatments=["Spray Mancozeb 75% WP @ 2 g/L.", "Apply Tricyclazole 75% WP @ 0.6 g/L."],
        prevention_tips=["Apply balanced NPK.", "Incorporate green manure."],
        original_image=test_img,
        heatmap_image=test_img,
    )
    print(f"Sample PDF Report Generated Successfully! Size: {len(pdf_bytes)} bytes")
    assert len(pdf_bytes) > 5000, "PDF bytes should be non-empty and well-formed!"


if __name__ == "__main__":
    test_pdf_build()
