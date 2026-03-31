from unittest.mock import patch, MagicMock


def test_run_baseline_agent_runs_episode():
    """run_baseline_agent should return a float score 0.0-1.0."""
    # Mock OpenAI to avoid real API calls
    mock_response = MagicMock()
    mock_output_item = MagicMock()
    mock_content = MagicMock()
    mock_content.type = "function_call"
    mock_content.name = "dispatch_ambulance"
    mock_content.arguments = '{"ambulance_id": "AMB_1", "patient_id": "P1", "target_node": "NH45_BYPASS"}'
    mock_output_item.content = [mock_content]
    mock_response.output = [mock_output_item]

    # Patch the openai module's responses attribute directly
    import openai
    mock_responses = MagicMock()
    mock_responses.create = MagicMock(return_value=mock_response)

    with patch.object(openai, "responses", mock_responses):
        from codered_env.inference import run_baseline_agent
        score = run_baseline_agent(task_id="task1", seed=0, api_key="sk-test")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
