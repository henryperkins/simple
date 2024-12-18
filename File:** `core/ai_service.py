async def test_ai_service():
    """Test the AI service with a simple prompt."""
    prompt = "Generate a summary for the following code:\n\nprint('Hello, World!')"
    response = await ai_service._make_api_call_with_retry(prompt, None)
    print(response)
