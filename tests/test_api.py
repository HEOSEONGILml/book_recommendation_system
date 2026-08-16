from fastapi.testclient import TestClient

from recommendation_service.main import create_app


def test_recommendation_api() -> None:
    response = TestClient(create_app()).post(
        "/v1/recommendations/carousels",
        headers={"X-Request-Id": "r_api"},
        json={
            "user_id": "u1", "session_id": "s1", "carousel_type": "LIBRARY_SIMILAR",
            "limit": 5, "non_personalized_work_ids": ["w_001"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "r_api"
    assert len(body["items"]) == 5
    assert body["metadata"]["arm"] == "A0"
