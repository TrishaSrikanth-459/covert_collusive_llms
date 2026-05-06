"""
Mock OpenAI client for testing without API calls.
Usage: set USE_MOCK_OPENAI=1 in environment to enable.
"""
import asyncio
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock

class MockChatCompletionMessage:
    def __init__(self, content: str):
        self.content = content

class MockChatCompletionChoice:
    def __init__(self, message: MockChatCompletionMessage):
        self.message = message

class MockChatCompletionUsage:
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20, total_tokens: int = 30):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

class MockChatCompletion:
    """
    Mock object returned by the chat.completions.create method.
    """
    def __init__(self, response_text: str = "Mock response", usage: Optional[MockChatCompletionUsage] = None):
        self.choices = [MockChatCompletionChoice(MockChatCompletionMessage(response_text))]
        self.usage = usage or MockChatCompletionUsage()

class MockCompletions:
    """
    Mock for the completions endpoint (used as client.chat.completions).
    """
    def __init__(self):
        self.default_response = "Mock response"

    def set_default_response(self, text: str):
        self.default_response = text

    def create(self, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> MockChatCompletion:
        """
        Synchronous mock method that returns a predictable response.
        You can extend this to simulate different behaviors based on input.
        """
        # Optionally inspect messages to return different responses
        # For now, return a generic mock.
        return MockChatCompletion(self.default_response)

class MockChat:
    """
    Mock for the chat endpoint (used as client.chat).
    """
    def __init__(self):
        self.completions = MockCompletions()

class MockOpenAIClient:
    """
    Mock client that replaces the real OpenAI client.
    """
    def __init__(self):
        self.chat = MockChat()