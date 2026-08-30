import os
import cohere

from deepeval.models.base_model import DeepEvalBaseLLM


class CohereModel(DeepEvalBaseLLM):

    def __init__(
        self,
        model_name: str,
        api_key: str,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.client = cohere.ClientV2(api_key=api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.message.content[0].text

    async def a_generate(self, prompt: str) -> str:
        response = await self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.message.content[0].text

    def get_model_name(self):
        return self.model_name