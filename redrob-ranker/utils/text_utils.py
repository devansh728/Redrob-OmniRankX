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
