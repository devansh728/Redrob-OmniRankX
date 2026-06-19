from pipeline import fusion
from config import settings

def test_fusion_and_reasoning():
    candidates = [
        {
            "candidate_id": "CAND_0000002",
            "semantic_score": 0.8,
            "trajectory_score": 0.8,
            "behavioral_score": 0.8,
            "profile": {"location": "Noida"},
            "redrob_signals": {
                "willing_to_relocate": True,
                "preferred_work_mode": "hybrid",
                "notice_period_days": 15,
                "github_activity_score": 80.0,
                "open_to_work_flag": True,
                "last_active_date": "2026-06-18"
            }
        },
        {
            "candidate_id": "CAND_0000001",
            "semantic_score": 0.8,
            "trajectory_score": 0.8,
            "behavioral_score": 0.8,
            "profile": {"location": "Noida"},
            "redrob_signals": {
                "willing_to_relocate": True,
                "preferred_work_mode": "hybrid",
                "notice_period_days": 15,
                "github_activity_score": 80.0,
                "open_to_work_flag": True,
                "last_active_date": "2026-06-18"
            }
        }
    ]
    res = fusion.run(candidates, settings)
    assert len(res) == 2
    assert res[0]["rank"] == 1
    assert res[1]["rank"] == 2
    assert res[0]["candidate_id"] == "CAND_0000001"
    assert res[1]["candidate_id"] == "CAND_0000002"
    assert "reasoning" in res[0]
    assert len(res[0]["reasoning"]) > 0
