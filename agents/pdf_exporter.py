from __future__ import annotations

import html
import importlib
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

colors = None
A4 = None
ParagraphStyle = None
getSampleStyleSheet = None
inch = None
HRFlowable = None
ListFlowable = None
ListItem = None
Paragraph = None
SimpleDocTemplate = None
Spacer = None


_INLINE_CURRENCY_MAP = {
    "₹": "INR ",
    "€": "EUR ",
    "£": "GBP ",
    "¥": "JPY ",
}


def _load_reportlab_dependencies() -> None:
    global colors
    global A4
    global ParagraphStyle
    global getSampleStyleSheet
    global inch
    global HRFlowable
    global ListFlowable
    global ListItem
    global Paragraph
    global SimpleDocTemplate
    global Spacer

    if colors is not None:
        return

    def _import_modules() -> tuple:
        reportlab_colors = importlib.import_module("reportlab.lib.colors")
        reportlab_pagesizes = importlib.import_module("reportlab.lib.pagesizes")
        reportlab_styles = importlib.import_module("reportlab.lib.styles")
        reportlab_units = importlib.import_module("reportlab.lib.units")
        reportlab_platypus = importlib.import_module("reportlab.platypus")

        return (
            reportlab_colors,
            reportlab_pagesizes.A4,
            reportlab_styles.ParagraphStyle,
            reportlab_styles.getSampleStyleSheet,
            reportlab_units.inch,
            reportlab_platypus.HRFlowable,
            reportlab_platypus.ListFlowable,
            reportlab_platypus.ListItem,
            reportlab_platypus.Paragraph,
            reportlab_platypus.SimpleDocTemplate,
            reportlab_platypus.Spacer,
        )

    try:
        (
            colors,
            A4,
            ParagraphStyle,
            getSampleStyleSheet,
            inch,
            HRFlowable,
            ListFlowable,
            ListItem,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        ) = _import_modules()
        return
    except ModuleNotFoundError as import_error:
        if import_error.name != "reportlab":
            raise

    try:
        install = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "reportlab",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
    except subprocess.TimeoutExpired as timeout_error:
        raise RuntimeError(
            "Timed out while installing reportlab. "
            "Install it manually with `python -m pip install reportlab`."
        ) from timeout_error
    if install.returncode != 0:
        raise RuntimeError(
            "PDF export dependency missing: install reportlab with "
            "`python -m pip install reportlab` or `pip install -r requirements.txt`."
        )

    (
        colors,
        A4,
        ParagraphStyle,
        getSampleStyleSheet,
        inch,
        HRFlowable,
        ListFlowable,
        ListItem,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    ) = _import_modules()


def _sanitize_text(text: str) -> str:
    cleaned = text or ""
    for symbol, replacement in _INLINE_CURRENCY_MAP.items():
        cleaned = cleaned.replace(symbol, replacement)
    cleaned = cleaned.replace("•", "*")
    # Keep ASCII for safer built-in PDF fonts.
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return cleaned


def _markdown_inline_to_para_text(text: str) -> str:
    escaped = html.escape(_sanitize_text(text))
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def _build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "FinTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=10,
            textColor=colors.HexColor("#0f172a"),
        ),
        "h2": ParagraphStyle(
            "FinH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            spaceBefore=10,
            spaceAfter=4,
            textColor=colors.HexColor("#1e293b"),
        ),
        "h3": ParagraphStyle(
            "FinH3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=8,
            spaceAfter=3,
            textColor=colors.HexColor("#334155"),
        ),
        "body": ParagraphStyle(
            "FinBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            spaceAfter=4,
            textColor=colors.HexColor("#111827"),
        ),
        "bullet": ParagraphStyle(
            "FinBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            leftIndent=5,
            textColor=colors.HexColor("#111827"),
        ),
        "meta": ParagraphStyle(
            "FinMeta",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#6b7280"),
            spaceAfter=8,
        ),
    }


def _on_page(canvas, doc):
    page = canvas.getPageNumber()
    footer = f"Fin-Alpha Analysis Report | Page {page}"
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.45 * inch, footer)
    canvas.setStrokeColor(colors.HexColor("#e5e7eb"))
    canvas.line(doc.leftMargin, 0.62 * inch, A4[0] - doc.rightMargin, 0.62 * inch)


def export_analysis_to_pdf(symbol: str, response_text: str) -> str:
    _load_reportlab_dependencies()

    output_dir = Path("output/pdf")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_symbol = re.sub(r"[^A-Za-z0-9._-]+", "_", (symbol or "UNKNOWN")).strip("_") or "UNKNOWN"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{safe_symbol}_analysis_{timestamp}.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title=f"{safe_symbol} Analysis Report",
        author="Fin-Alpha",
    )
    styles = _build_styles()
    story: List = []
    story.append(Paragraph(_markdown_inline_to_para_text(f"{safe_symbol} Analysis Report"), styles["title"]))
    story.append(
        Paragraph(
            _markdown_inline_to_para_text(
                f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
            ),
            styles["meta"],
        )
    )

    lines = response_text.splitlines() if response_text else []
    bullet_buffer: List[str] = []

    def flush_bullets():
        if not bullet_buffer:
            return
        items = [
            ListItem(Paragraph(_markdown_inline_to_para_text(item), styles["bullet"]))
            for item in bullet_buffer
        ]
        story.append(
            ListFlowable(
                items,
                bulletType="bullet",
                start="circle",
                bulletFontName="Helvetica",
                bulletFontSize=8,
                leftIndent=15,
            )
        )
        story.append(Spacer(1, 4))
        bullet_buffer.clear()

    for raw in lines:
        line = _sanitize_text(raw).rstrip()
        stripped = line.strip()

        if not stripped:
            flush_bullets()
            story.append(Spacer(1, 4))
            continue

        if stripped in {"---", "***"}:
            flush_bullets()
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#d1d5db")))
            story.append(Spacer(1, 6))
            continue

        bullet_match = re.match(r"^(?:[-*]\s+|\d+\.\s+)(.+)$", stripped)
        if bullet_match:
            bullet_buffer.append(bullet_match.group(1).strip())
            continue

        flush_bullets()

        if stripped.startswith("### "):
            story.append(Paragraph(_markdown_inline_to_para_text(stripped[4:].strip()), styles["h3"]))
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(_markdown_inline_to_para_text(stripped[3:].strip()), styles["h2"]))
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(_markdown_inline_to_para_text(stripped[2:].strip()), styles["title"]))
            continue

        story.append(Paragraph(_markdown_inline_to_para_text(stripped), styles["body"]))

    flush_bullets()

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return os.fspath(output_path)
