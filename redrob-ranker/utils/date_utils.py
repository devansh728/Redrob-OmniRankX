import datetime

REFERENCE_TODAY = datetime.date(2026, 6, 19)

def parse_date(s):
    if not s:
        return None
    try:
        if isinstance(s, str):
            return datetime.date.fromisoformat(s[:10])
    except Exception:
        pass
    return None

def months_between(start, end=None):
    if not start:
        return 0
    if not end:
        end = REFERENCE_TODAY
    diff_years = end.year - start.year
    diff_months = end.month - start.month
    total = diff_years * 12 + diff_months
    return max(0, total)

def total_career_months(career_history):
    if not career_history:
        return 0
    return sum(item.get("duration_months", 0) for item in career_history)
