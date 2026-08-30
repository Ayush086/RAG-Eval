import json
import os

from groq import Groq, AsyncGroq
from pydantic import BaseModel

from deepeval.models.base_model import DeepEvalBaseLLM


class GroqModel(DeepEvalBaseLLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

        api_key = os.environ["GROQ_API_KEY"]

        self.client = Groq(api_key=api_key)
        self.async_client = AsyncGroq(api_key=api_key)

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    def _build_messages(self, prompt: str, schema: BaseModel | None):
        if schema is None:
            return [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

        schema_json = json.dumps(
            schema.model_json_schema(),
            indent=2,
        )

        return [
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge. "
                    "Return ONLY a valid JSON object. "
                    "Do not use markdown. "
                    "Do not wrap the JSON in ```json fences. "
                    "The JSON must conform to this schema:\n\n"
                    f"{schema_json}"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    def generate(
        self,
        prompt: str,
        schema: BaseModel | None = None,
    ):
        messages = self._build_messages(prompt, schema)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={
                "type": "json_object",
            },
            reasoning_effort="low",
        )

        output = response.choices[0].message.content

        if not output:
            raise RuntimeError("Groq returned an empty response.")

        if schema is None:
            return output

        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Groq returned invalid JSON:\n{output}"
            ) from e

        return schema.model_validate(data)

    async def a_generate(
        self,
        prompt: str,
        schema: BaseModel | None = None,
    ):
        messages = self._build_messages(prompt, schema)

        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={
                "type": "json_object",
            },
            reasoning_effort="low",
        )

        output = response.choices[0].message.content

        if not output:
            raise RuntimeError("Groq returned an empty response.")

        if schema is None:
            return output

        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Groq returned invalid JSON:\n{output}"
            ) from e

        return schema.model_validate(data)