import re

def clean_text(s):
    if not s:
        return ""
    s = s.strip().lower()
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s)

def concat_career_text(candidate):
    parts = []
    profile = candidate.get("profile", {})
    
    headline = profile.get("headline")
    if headline:
        parts.append(headline)
        
    summary = profile.get("summary")
    if summary:
        parts.append(summary)
        
    career = candidate.get("career_history", [])
    for job in career:
        title = job.get("title")
        if title:
            parts.append(title)
        desc = job.get("description")
        if desc:
            parts.append(desc)
            
    return " ".join(parts)

def tokenize_for_bm25(text):
    if not text:
        return []
    cleaned = re.sub(r'[^\w\s]', '', text)
    return cleaned.lower().split()

def extract_raw_text_from_docx(path):
    import docx
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    tables_text = []
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                tables_text.append(" | ".join(row_text))
    return "\n".join(paragraphs + tables_text)

def extract_raw_text_from_file(path):
    if path.lower().endswith(".docx"):
        return extract_raw_text_from_docx(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

