from pipeline import semantic
from config import settings

def test_semantic_run():
    candidates = [
        {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": "A",
                "headline": "Senior AI Engineer",
                "summary": "Expert in NLP and IR ranking vectors",
                "location": "Pune",
                "country": "India",
                "years_of_experience": 5.0,
                "current_title": "AI Engineer",
                "current_company": "Razorpay",
                "current_company_size": "501-1000",
                "current_industry": "Fintech"
            },
            "career_history": [],
            "redrob_signals": {
                "skill_assessment_scores": {
                    "python": 90,
                    "nlp": 85
                }
            }
        },
        {
            "candidate_id": "CAND_0000002",
            "profile": {
                "anonymized_name": "B",
                "headline": "Java Backend Developer",
                "summary": "Building standard spring boot services",
                "location": "Noida",
                "country": "India",
                "years_of_experience": 4.0,
                "current_title": "Java Dev",
                "current_company": "TCS",
                "current_company_size": "10001+",
                "current_industry": "IT Services"
            },
            "career_history": [],
            "redrob_signals": {
                "skill_assessment_scores": {
                    "python": 20
                }
            }
        }
    ]
    res = semantic.run(candidates, None, settings)
    assert len(res) == 2
    assert "semantic_score" in res[0]
    assert "semantic_score" in res[1]
    assert res[0]["semantic_score"] >= res[1]["semantic_score"]
