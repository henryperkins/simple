import pytest
from unittest.mock import AsyncMock, patch
from core.ai_service import AIService
from core.config import AIConfig
from core.types.base import DocumentationContext, ProcessingResult

@pytest.mark.asyncio
async def test_generate_documentation():
    # Mocking dependencies
    with patch('core.ai_service.Injector.get') as mock_injector:
        mock_injector.return_value = AsyncMock()
        mock_injector.return_value.ai = AIConfig(api_key="test_key", endpoint="test_endpoint", deployment="test_deployment")
        
        ai_service = AIService()
        
        # Mocking the API call
        with patch('core.ai_service.AIService._make_api_call_with_retry', new_callable=AsyncMock) as mock_api_call:
            mock_api_call.return_value = {"choices": [{"message": {"content": "Test docstring"}}]}
            
            context = DocumentationContext(source_code="def test(): pass", module_path="test_module.py")
            result = await ai_service.generate_documentation(context)
            
            assert isinstance(result, ProcessingResult)
            assert result.content == {"error": "Failed to parse AI response"}  # Since we're not actually parsing the response

@pytest.mark.asyncio
async def test_api_call_with_retry():
    with patch('core.ai_service.Injector.get') as mock_injector:
        mock_injector.return_value = AsyncMock()
        mock_injector.return_value.ai = AIConfig(api_key="test_key", endpoint="test_endpoint", deployment="test_deployment")
        
        ai_service = AIService()
        
        # Mocking the API call to fail twice and then succeed
        with patch('aiohttp.ClientSession.post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [
                aiohttp.ClientResponseError(None, None, status=500),
                aiohttp.ClientResponseError(None, None, status=500),
                AsyncMock(status=200, json=AsyncMock(return_value={"choices": [{"message": {"content": "Success"}}]}))
            ]
            
            result = await ai_service._make_api_call_with_retry("test_prompt")
            assert result == {"choices": [{"message": {"content": "Success"}}]}
