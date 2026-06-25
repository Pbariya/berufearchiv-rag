import os
import json
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract



# CONFIGURATION

PDF_DIRECTORY         = ".DATA/archivdaten"
OUTPUT_JSON_DIRECTORY = "./Archivdaten_jsons"

# Tesseract language string: German primary, English fallback.
TESSERACT_LANG = "deu+eng"

# Resolution for PDF-to-image conversion. 300 DPI is standard for OCR accuracy.
DPI = 300

# Local installation paths
POPPLER_PATH = r'D:\UNI KOBLENZ\Thesis\Implementation\poppler\Library\bin'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\tesseract.exe'

# TEXT EXTRACTION
#Convert a single PDF to images and apply Tesseract OCR on each page.

def extract_text_from_pdf(pdf_path, dpi=DPI, lang=TESSERACT_LANG,
                           poppler_path=None):
    print(f"  Processing: {pdf_path.name}")
    try:
        if poppler_path:
            images = convert_from_path(str(pdf_path), dpi=dpi, fmt='jpeg',
                                       poppler_path=poppler_path)
        else:
            images = convert_from_path(str(pdf_path), dpi=dpi, fmt='jpeg')

        page_texts = []
        for i, image in enumerate(images, 1):
            text = pytesseract.image_to_string(image, lang=lang, config='--psm 1')
            page_texts.append(text.strip())
            print(f"    Page {i}/{len(images)} processed")

        full_text = "\n\n".join(page_texts)
        print(f"    Extracted {len(full_text)} characters")
        return full_text, len(images)

    except Exception as e:
        print(f"    ERROR: {e}")
        return "", 0

# BATCH PROCESSING
#Process all PDF files in a directory and write each to an individual JSON.

def extract_pdfs_to_individual_jsons(pdf_directory, output_json_dir,
                                      dpi=DPI, lang=TESSERACT_LANG,
                                      poppler_path=None):
    pdf_dir  = Path(pdf_directory)
    json_dir = Path(output_json_dir)

    if not pdf_dir.exists():
        print(f"ERROR: Directory not found: {pdf_directory}")
        return

    json_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {pdf_directory}")
        return

    print(f"\nBerufearchiv OCR Pipeline")
    print(f"{'='*60}")
    print(f"Input directory  : {pdf_directory}")
    print(f"Output directory : {output_json_dir}")
    print(f"Documents found  : {len(pdf_files)}")
    print(f"OCR language     : {lang}  |  DPI: {dpi}")
    print(f"{'='*60}\n")

    successful   = 0
    failed       = 0
    failed_files = []

    for pdf_path in pdf_files:
        text, page_count = extract_text_from_pdf(
            pdf_path, dpi=dpi, lang=lang, poppler_path=poppler_path
        )

        if text and text.strip():
            json_data = {
                "source":    pdf_path.name,
                "full_path": str(pdf_path),
                "text":      text,
                "metadata":  {
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "page_count": page_count,
                },
            }
            json_filename = pdf_path.stem + ".json"
            json_path     = json_dir / json_filename

            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"    Saved: {json_filename}\n")
                successful += 1
            except Exception as e:
                print(f"    ERROR saving JSON: {e}\n")
                failed += 1
                failed_files.append(pdf_path.name)
        else:
            failed += 1
            failed_files.append(pdf_path.name)
            print(f"    WARNING: No text extracted from {pdf_path.name}\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"Extraction complete")
    print(f"  Succeeded : {successful}")
    print(f"  Failed    : {failed}")
    if failed_files:
        print(f"\n  Failed files:")
        for fn in failed_files:
            print(f"    - {fn}")

    if successful > 0:
        json_files = sorted(json_dir.glob("*.json"))
        total_mb   = sum(f.stat().st_size for f in json_files) / (1024 * 1024)
        print(f"\n  JSON files written : {len(json_files)}")
        print(f"  Total size         : {total_mb:.2f} MB")
        print(f"{'='*60}\n")

# ENTRY POINT

if __name__ == "__main__":
    extract_pdfs_to_individual_jsons(
        pdf_directory=PDF_DIRECTORY,
        output_json_dir=OUTPUT_JSON_DIRECTORY,
        dpi=DPI,
        lang=TESSERACT_LANG,
        poppler_path=POPPLER_PATH,
    )
