"""
generate_sample_pdf.py
────────────────────────────
Run this once to generate a sample HR policy PDF for testing.

Usage:
    python generate_sample_pdf.py
"""

from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "data" / "sample_docs" / "hr_policy_handbook.pdf"
TEXT_PATH   = Path(__file__).parent / "data" / "sample_docs" / "hr_policy_handbook.txt"


def generate_pdf():
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.enums import TA_LEFT

        content = TEXT_PATH.read_text(encoding="utf-8")
        doc = SimpleDocTemplate(
            str(OUTPUT_PATH),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm,
        )

        styles = getSampleStyleSheet()
        story = []

        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("━"):
                story.append(Spacer(1, 12))
            elif stripped.isupper() and len(stripped) > 10:
                story.append(Paragraph(f"<b>{stripped}</b>", styles["Heading2"]))
            elif stripped:
                story.append(Paragraph(stripped, styles["Normal"]))
                story.append(Spacer(1, 4))

        doc.build(story)
        print(f"✅ Sample PDF created: {OUTPUT_PATH}")

    except ImportError:
        # Fallback: create a simple PDF using only built-in tools
        print("reportlab not found — creating minimal PDF...")
        _create_minimal_pdf()


def _create_minimal_pdf():
    """Create a minimal valid PDF without external libraries."""
    content = TEXT_PATH.read_text(encoding="utf-8")
    lines = [l for l in content.split("\n") if l.strip()][:50]  # First 50 lines

    pdf_content = "%PDF-1.4\n"
    text_stream = "\n".join(
        f"BT /F1 10 Tf 50 {750 - i*14} Td ({line[:80].replace('(','').replace(')','')}) Tj ET"
        for i, line in enumerate(lines)
    )

    stream = f"stream\n{text_stream}\nendstream"
    resources = "<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>"

    obj1 = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = f"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources {resources} >>\nendobj\n"
    obj4 = f"4 0 obj\n<< /Length {len(stream)} >>\n{stream}\nendobj\n"

    body = obj1 + obj2 + obj3 + obj4
    xref_offset = len(pdf_content) + len(body)

    pdf = f"{pdf_content}{body}xref\n0 5\n0000000000 65535 f \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF"

    OUTPUT_PATH.write_bytes(pdf.encode("latin-1", errors="replace"))
    print(f"✅ Minimal PDF created: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_pdf()
