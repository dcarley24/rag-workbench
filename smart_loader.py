import os
import re
from unstructured.partition.pdf import partition_pdf
from docx import Document

def is_likely_reference(text):
    text = text.strip()
    if re.match(r'^\s*[\(\[]\d+[\)\]]', text):
        return True
    if re.search(r'\(\d{4}\)', text) and len(text) < 250:
        if re.search(r'[A-Z][a-z]+,\s[A-Z]\.|et al\.', text):
            return True
    return False

def find_reference_section_start(elements):
    reference_keywords = ['references', 'bibliography', 'citations']
    for i, el in enumerate(elements):
        element_text = el.text.strip().lower()
        if len(element_text) < 20 and element_text in reference_keywords:
            return i
    return -1

def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def load_docx(file_path):
    doc = Document(file_path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

def load_pdf(file_path):
    """
    MODIFICATION: Switched the default strategy to "fast" for much better performance.
    """
    try:
        elements = partition_pdf(filename=file_path, strategy="fast")
    except Exception as e:
        print(f"PDF parsing failed for {os.path.basename(file_path)}: {e}")
        return []

    ref_start_index = find_reference_section_start(elements)
    if ref_start_index != -1:
        elements = elements[:ref_start_index]

    clean_content = []
    for el in elements:
        # Additional filter for section headings which are not useful context
        if el.text and el.text.strip() and not is_likely_reference(el.text) and not re.match(r'^\d\s|\d\.\d', el.text.strip()):
            clean_content.append(el.text.strip())

    return clean_content

def smart_load(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            return load_pdf(file_path)
        #...(rest of the function is the same)
        elif ext == '.docx':
            return load_docx(file_path)
        elif ext == '.txt':
            return load_txt(file_path)
        else:
            print(f"Skipping unsupported file type: {ext}")
            return []
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return []
