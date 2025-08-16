import pdfplumber

with pdfplumber.open("Ameesha Priya Software Development Engineer.pdf") as pdf:
    full_text = ""
    for page in pdf.pages:
        full_text += page.extract_text() + "\n"

print(full_text)
