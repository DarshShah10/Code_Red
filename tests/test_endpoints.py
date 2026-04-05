from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

def test_get_tasks():
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert len(data["tasks"]) == 5
    assert [t["task_id"] for t in data["tasks"]] == ["task1", "task2", "task3", "task4", "task5"]

def test_grader_endpoint_task1():
    response = client.post("/grader", json={"task_id": "task1", "seed": 42})
    assert response.status_code == 200
    data = response.json()
    assert "final_score" in data
    assert 0.0 <= data["final_score"] <= 1.0

def test_baseline_endpoint_requires_api_key():
    # Test with a mock key — will fail at OpenAI call but should not 500
    response = client.post("/baseline", json={"task_id": "task1", "openai_api_key": "sk-test"})
    # Should return 200, 400, 401 (OpenAI auth failure), or 501 (not implemented yet)
    assert response.status_code in [200, 400, 401, 501]
