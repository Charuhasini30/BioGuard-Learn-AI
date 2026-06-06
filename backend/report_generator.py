from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import io

def generate_pdf(data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "BioGuard AI - Ecosystem Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.drawString(50, 680, f"Region Health Score: {data.get('biodiversity_health_score')}")
    c.drawString(50, 660, f"Risk Level: {data.get('species_risk_level')}")
    c.drawString(50, 640, f"Ecosystem Stability Index: {data.get('ecosystem_stability_index')}")
    c.drawString(50, 620, f"Resilience Index: {data.get('biodiversity_resilience_index')}")

    # AI Insight Section
    c.drawString(50, 580, "AI Conservation Insight:")

    insight = "Maintain habitat corridors and reduce fragmentation to improve biodiversity resilience."

    c.setFont("Helvetica-Oblique", 11)
    c.drawString(50, 560, insight)

    c.save()
    buffer.seek(0)
    return buffer