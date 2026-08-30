import json
import os

from groq import Groq, AsyncGroq
from pydantic import BaseModel

from deepeval.models.base_model import DeepEvalBaseLLM


class GroqModel(DeepEvalBaseLLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

        self.client = Groq(
            api_key=os.environ["GROQ_API_KEY"]
        )

        self.async_client = AsyncGroq(
            api_key=os.environ["GROQ_API_KEY"]
        )

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    @staticmethod
    def _parse_response(
        output: str,
        schema: BaseModel | None
    ):
        if schema is None:
            return output

        output = output.strip()

        # Remove markdown JSON fences if the model adds them.
        if output.startswith("```json"):
            output = output[len("```json"):].strip()

        elif output.startswith("```"):
            output = output[len("```"):].strip()

        if output.endswith("```"):
            output = output[:-3].strip()

        data = json.loads(output)

        return schema.model_validate(data)

    def generate(
        self,
        prompt: str,
        schema: BaseModel | None = None
    ):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation judge. "
                        "Return ONLY a valid JSON object. "
                        "Do not use Markdown or code fences."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={
                "type": "json_object"
            },
        )

        output = response.choices[0].message.content

        return self._parse_response(output, schema)

    async def a_generate(
        self,
        prompt: str,
        schema: BaseModel | None = None
    ):
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation judge. "
                        "Return ONLY a valid JSON object. "
                        "Do not use Markdown or code fences."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={
                "type": "json_object"
            },
        )

        output = response.choices[0].message.content

        return self._parse_response(output, schema)
