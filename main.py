import requests
import io
from pdfminer.high_level import extract_text # This will now be from pdfminer.rtl
# from pdfminer.layout import LAParams # You can uncomment and use this for layout tuning if needed

url = "https://iqraonline.net/wp-content/uploads/2018/10/%D8%B5%D8%AF-%D8%B1%D9%88%D8%B6%D9%87-%D8%AC%D8%B9%D9%84%DB%8C.pdf"

try:
    res = requests.get(url)
    res.raise_for_status() # Check for HTTP errors

    pdf_content_bytes = res.content
    pdf_file_object = io.BytesIO(pdf_content_bytes)

    # The extract_text function from pdfminer.rtl will handle the BiDi reordering
    extracted_text = extract_text(pdf_file_object)

    print(extracted_text)

except requests.exceptions.RequestException as e:
    print(f"Error downloading PDF: {e}")
except Exception as e:
    print(f"An error occurred during PDF text extraction: {e}")