from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from report_generator import generate_pdf

app = FastAPI()

@app.post("/generate-report")
def create_report(data: dict):

    pdf_buffer = generate_pdf(data)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=bio_guard_report.pdf"
        }
    )