from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def create_pdf(report_text, filename="incident_report.pdf"):

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    content = [
        Paragraph(report_text, styles["BodyText"])
    ]

    doc.build(content)

    return filename