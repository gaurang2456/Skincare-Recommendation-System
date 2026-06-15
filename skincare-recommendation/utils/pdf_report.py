"""
utils/pdf_report.py

Generate an in-memory PDF report from a RecommendationResult and AssessmentInput.

Uses ReportLab's Platypus framework.

Requirements: 7.1, 7.2
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ml.predictor import AssessmentInput, RecommendationResult


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BRAND_BLUE = colors.HexColor("#2C5F8A")
_LIGHT_GREY = colors.HexColor("#F5F5F5")
_ALT_ROW = colors.HexColor("#EAF2FB")
_WHITE = colors.white
_DARK_TEXT = colors.HexColor("#222222")
_MUTED_TEXT = colors.HexColor("#888888")


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _build_styles() -> dict:
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=18,
            textColor=_BRAND_BLUE,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "timestamp": ParagraphStyle(
            "Timestamp",
            parent=base["Normal"],
            fontSize=10,
            textColor=_MUTED_TEXT,
            spaceAfter=12,
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontSize=13,
            textColor=_BRAND_BLUE,
            spaceBefore=14,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=11,
            textColor=_DARK_TEXT,
            spaceAfter=4,
        ),
        "score": ParagraphStyle(
            "Score",
            parent=base["Normal"],
            fontSize=14,
            textColor=_BRAND_BLUE,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "step": ParagraphStyle(
            "Step",
            parent=base["Normal"],
            fontSize=11,
            textColor=_DARK_TEXT,
            spaceAfter=3,
            leftIndent=12,
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_hr(styles: dict) -> list:
    return [
        Spacer(1, 0.15 * cm),
        HRFlowable(width="100%", thickness=1, color=_BRAND_BLUE, spaceAfter=6),
    ]


def _build_skin_profile(profile: AssessmentInput, styles: dict) -> list:
    elements = []
    elements.append(Paragraph("YOUR SKIN PROFILE", styles["section_heading"]))
    elements += _section_hr(styles)
    elements.append(Paragraph(f"<b>Age:</b> {profile.age}", styles["body"]))
    elements.append(Paragraph(f"<b>Skin Type:</b> {profile.skin_type}", styles["body"]))

    concerns_str = ", ".join(
        f"{c} ({profile.severity_scores.get(c, '?')})"
        for c in profile.concerns
    )
    elements.append(Paragraph(f"<b>Concerns:</b> {concerns_str}", styles["body"]))

    if profile.excluded_ingredients:
        excl_str = ", ".join(profile.excluded_ingredients)
        elements.append(Paragraph(f"<b>Excluded Ingredients:</b> {excl_str}", styles["body"]))
    else:
        elements.append(Paragraph("<b>Excluded Ingredients:</b> None", styles["body"]))

    return elements


def _build_health_score(result: RecommendationResult, styles: dict) -> list:
    elements = []
    elements.append(Paragraph("SKIN HEALTH SCORE", styles["section_heading"]))
    elements += _section_hr(styles)
    elements.append(
        Paragraph(
            f"{result.skin_health_score}/100 — {result.skin_health_label}",
            styles["score"],
        )
    )
    # One-sentence explanation based on label
    explanations = {
        "Excellent": "Your skin is in great shape — keep up the routine!",
        "Good": "Your skin shows some minor concerns that are manageable with the right routine.",
        "Fair": "Your skin has a few concerns; a targeted routine will make a noticeable difference.",
        "Needs Attention": "Your skin needs focused care — the recommended products are specially chosen to help.",
    }
    explanation = explanations.get(
        result.skin_health_label,
        "Consistent skincare will improve your skin health over time.",
    )
    elements.append(Paragraph(explanation, styles["body"]))
    return elements


def _build_products_table(result: RecommendationResult, styles: dict) -> list:
    elements = []
    elements.append(Paragraph("RECOMMENDED PRODUCTS", styles["section_heading"]))
    elements += _section_hr(styles)

    # Table header
    header = ["Product Name", "Category", "Key Ingredients", "Confidence", "Reason"]
    recs = [result.cleanser, result.moisturizer, result.serum, result.sunscreen]

    data = [header]
    for rec in recs:
        if rec is not None:
            data.append([
                Paragraph(rec.name, styles["body"]),
                Paragraph(rec.category, styles["body"]),
                Paragraph(", ".join(rec.key_ingredients) if rec.key_ingredients else "—", styles["body"]),
                Paragraph(f"{rec.confidence_score * 100:.1f}%", styles["body"]),
                Paragraph(rec.reason, styles["body"]),
            ])

    col_widths = [3.8 * cm, 2.4 * cm, 3.8 * cm, 1.8 * cm, 5.0 * cm]

    table = Table(data, colWidths=col_widths, repeatRows=1)

    # Build row colours: alternating between white and _ALT_ROW
    row_styles = [
        # Header
        ("BACKGROUND", (0, 0), (-1, 0), _BRAND_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]

    # Alternating row colours for data rows
    for row_idx in range(1, len(data)):
        bg = _ALT_ROW if row_idx % 2 == 0 else _WHITE
        row_styles.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))

    table.setStyle(TableStyle(row_styles))
    elements.append(table)
    return elements


def _build_routine(title: str, routine: list, styles: dict) -> list:
    elements = []
    elements.append(Paragraph(title, styles["section_heading"]))
    elements += _section_hr(styles)
    for step in routine:
        elements.append(
            Paragraph(
                f"<b>Step {step.step}</b> — {step.category}: {step.product}",
                styles["step"],
            )
        )
    return elements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pdf(result: RecommendationResult, profile: AssessmentInput) -> bytes:
    """Generate an in-memory PDF report and return the raw bytes.

    Args:
        result:  The RecommendationResult produced by get_recommendation().
        profile: The AssessmentInput submitted by the user.

    Returns:
        PDF file contents as bytes (starts with b'%PDF').
    """
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = _build_styles()
    story = []

    # --- Title ---
    story.append(Paragraph("AI Skincare Recommendation Report", styles["title"]))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated: {ts}", styles["timestamp"]))
    story.append(HRFlowable(width="100%", thickness=2, color=_BRAND_BLUE, spaceAfter=10))

    # --- Skin Profile ---
    story += _build_skin_profile(profile, styles)
    story.append(Spacer(1, 0.3 * cm))

    # --- Skin Health Score ---
    story += _build_health_score(result, styles)
    story.append(Spacer(1, 0.3 * cm))

    # --- Recommended Products table ---
    story += _build_products_table(result, styles)
    story.append(Spacer(1, 0.3 * cm))

    # --- Morning Routine ---
    story += _build_routine("☀️ MORNING ROUTINE", result.morning_routine, styles)
    story.append(Spacer(1, 0.3 * cm))

    # --- Night Routine ---
    story += _build_routine("🌙 NIGHT ROUTINE", result.night_routine, styles)

    doc.build(story)
    return buffer.getvalue()
