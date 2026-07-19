from fastapi.testclient import TestClient

from app.main import create_app


def test_health_endpoint_is_available() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ndis_route_is_registered() -> None:
    app = create_app()
    paths = {route.path for route in app.routes}

    assert "/api/v1/ndis-navigation/plan" in paths
    assert "/api/v1/ndis-navigation/document-plan" in paths
    assert "/api/v1/patient-chat/message" in paths
    assert "/api/v1/demo/profile" in paths
    assert "/api/v1/demo/profile/approval" in paths
    assert "/api/v1/demo/history" in paths
    assert "/api/v1/demo/chat-history" in paths
    assert "/api/v1/demo/providers" in paths
    assert "/api/v1/demo/referrals" in paths
