from pipeline import behavioral
from config import settings

def test_behavioral_scoring():
    candidates = [
        {
            "candidate_id": "CAND_0000001",
            "redrob_signals": {
                "last_active_date": "2026-06-18",
                "recruiter_response_rate": 0.9,
                "github_activity_score": 90.0,
                "interview_completion_rate": 0.95,
                "notice_period_days": 15,
                "expected_salary_range_inr_lpa": {"min": 40, "max": 50},
                "open_to_work_flag": True
            }
        },
        {
            "candidate_id": "CAND_0000002",
            "redrob_signals": {
                "last_active_date": "2025-01-01",
                "recruiter_response_rate": 0.2,
                "github_activity_score": 10.0,
                "interview_completion_rate": 0.2,
                "notice_period_days": 90,
                "expected_salary_range_inr_lpa": {"min": 80, "max": 90},
                "open_to_work_flag": False
            }
        }
    ]
    res = behavioral.run(candidates, settings)
    assert len(res) == 2
    assert "behavioral_score" in res[0]
    assert "behavioral_score" in res[1]
    assert res[0]["behavioral_score"] > res[1]["behavioral_score"]
