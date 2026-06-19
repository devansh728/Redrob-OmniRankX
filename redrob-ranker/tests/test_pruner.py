from pipeline import pruner
from config import settings

def test_pruner_pass():
    candidate = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "John Doe",
            "headline": "Lead ML Engineer",
            "summary": "Building search engine and NLP systems",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": "ML Engineer",
            "current_company": "Razorpay",
            "current_company_size": "1001-5000",
            "current_industry": "Fintech"
        },
        "career_history": [
            {
                "company": "Razorpay",
                "title": "ML Engineer",
                "start_date": "2021-06-19",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "Fintech",
                "company_size": "1001-5000",
                "description": "Building NLP rankings and vector embeddings"
            }
        ],
        "education": [
            {
                "institution": "IIT",
                "degree": "BTech",
                "field_of_study": "CS",
                "start_year": 2017,
                "end_year": 2021
            }
        ],
        "redrob_signals": {
            "profile_completeness_score": 100,
            "signup_date": "2021-06-19",
            "last_active_date": "2026-06-18",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 2,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 10,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 40},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 50,
            "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
    }
    res = pruner.run([candidate], settings)
    assert len(res) == 1

def test_pruner_honeypot():
    candidate = {
        "candidate_id": "CAND_0000002",
        "profile": {
            "anonymized_name": "Honeypot Candidate",
            "headline": "Lead ML Engineer",
            "summary": "Building search engine and NLP systems",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": "ML Engineer",
            "current_company": "Razorpay",
            "current_company_size": "1001-5000",
            "current_industry": "Fintech"
        },
        "career_history": [
            {
                "company": "Razorpay",
                "title": "ML Engineer",
                "start_date": "2021-06-19",
                "end_date": None,
                "duration_months": 12,
                "is_current": True,
                "industry": "Fintech",
                "company_size": "1001-5000",
                "description": "Building NLP rankings and vector embeddings"
            }
        ],
        "education": [
            {
                "institution": "IIT",
                "degree": "BTech",
                "field_of_study": "CS",
                "start_year": 2017,
                "end_year": 2023
            }
        ],
        "redrob_signals": {
            "profile_completeness_score": 100,
            "signup_date": "2021-06-19",
            "last_active_date": "2023-06-18",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 2,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 10,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 40},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 98,
            "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
    }
    res = pruner.run([candidate], settings)
    assert len(res) == 0
