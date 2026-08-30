import json
import os

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)


from src.retriever import build_retriever
from models.groq_model import GroqModel


load_dotenv()


GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = "openai/gpt-oss-20b"
THRESHOLD = 0.7
os.environ["DEEPEVAL_DISABLE_CACHE"] = "1"


judge = GroqModel(
    model_name=JUDGE_MODEL
)


with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


retriever = build_retriever()

test_cases = []

for g in goldens:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [
        doc.page_content for doc in retrieved
    ]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )


metrics = [
    ContextualRecallMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
    ),
    ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
    ),
]


evaluate(
    test_cases=test_cases,
    metrics=metrics,

    # Prevent Groq rate-limit errors
    async_config=AsyncConfig(
    run_async=True,
    max_concurrent=1,
    throttle_value=20,
),

    hyperparameters={
        "retriever": "base_k5",
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH,
    },
)