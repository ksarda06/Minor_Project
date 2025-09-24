# utils/report.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import datetime
import os

def save_report_pdf(report_text, patient_name="Unknown", out_path="reports"):
    os.makedirs(out_path, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(out_path, f"{patient_name}_report_{ts}.pdf")
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    x = 40
    y = height - 60
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, f"Patient Report - {patient_name}")
    c.setFont("Helvetica", 10)
    y -= 30
    for line in report_text.split("\n"):
        if y < 60:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)
        c.drawString(x, y, line)
        y -= 14
    c.save()
    return filename

