```python
import os
from langchain_openai import ChatOpenAI


class LLM:
    @staticmethod
    def get_llm1():
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set."
            )

        return ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model="openrouter/free",
            temperature=0.8
        )
```
