import json

from deepeval.models import DeepEvalBaseLLM
from langchain_cohere import ChatCohere


class CohereJudge(DeepEvalBaseLLM):
    def __init__(self):
        # self.model_name = "command-a-03-2025"
        self.model_name = "command-a-plus-05-2026"
        self.llm = ChatCohere(model=self.model_name)

    def load_model(self):
        return self.llm

    def _parse_response(self, response: str, schema):
        # Remove Markdown code fences if the model returns them
        response = response.strip()

        if response.startswith("```json"):
            response = response[7:]

        elif response.startswith("```"):
            response = response[3:]

        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        data = json.loads(response)

        return schema.model_validate(data)

    def generate(self, prompt: str, schema=None):
        response = self.llm.invoke(prompt)

        if schema is not None:
            return self._parse_response(response.content, schema)

        return response.content

    async def a_generate(self, prompt: str, schema=None):
        response = await self.llm.ainvoke(prompt)

        if schema is not None:
            return self._parse_response(response.content, schema)

        return response.content

    def get_model_name(self):
        return self.model_name