from pipeline import trajectory
from config import settings

def test_trajectory_scoring():
    candidates = [
        {
            "candidate_id": "CAND_0000001",
            "career_history": [
                {
                    "company": "Swiggy",
                    "title": "Senior ML Engineer",
                    "start_date": "2024-01-01",
                    "end_date": None,
                    "duration_months": 30,
                    "is_current": True,
                    "industry": "Internet",
                    "company_size": "1001-5000"
                },
                {
                    "company": "TCS",
                    "title": "Software Engineer",
                    "start_date": "2021-01-01",
                    "end_date": "2023-12-31",
                    "duration_months": 36,
                    "is_current": False,
                    "industry": "IT Services",
                    "company_size": "10001+"
                }
            ]
        },
        {
            "candidate_id": "CAND_0000002",
            "career_history": [
                {
                    "company": "TCS",
                    "title": "Software Engineer",
                    "start_date": "2025-01-01",
                    "end_date": None,
                    "duration_months": 10,
                    "is_current": True,
                    "industry": "IT Services",
                    "company_size": "10001+"
                }
            ]
        }
    ]
    res = trajectory.run(candidates, settings)
    assert len(res) == 2
    assert "trajectory_score" in res[0]
    assert "trajectory_score" in res[1]
    assert res[0]["trajectory_score"] > res[1]["trajectory_score"]
