"""
src/generator.py — the GENERATOR component.

Given a query and context (retrieved chunks), produce an answer grounded in
the context. The prompt is faithfulness-first: answer ONLY from the context,
and abstain when the context doesn't contain the answer.

    from src.generator import generate
    answer = generate("what is drift?", ["chunk text 1", "chunk text 2"])

There are two entry points:
  - generate(query, context)        -> returns the full answer string (default)
  - generate_stream(query, context) -> yields the answer in chunks as it is
                                       produced, for streaming UIs and for
                                       measuring time-to-first-token (TTFT)
Both share the exact same prompt, model, and chain — the only difference is
that one waits for the whole answer and the other emits it token-by-token.
"""

# from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = ChatCohere(model="command-a-03-2025")

# faithfulness-first prompt: ground every claim in the context, abstain if unsure
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful teaching assistant for a course on LLM evaluations. Answer the student's question using ONLY the information in the context provided below.

Rules:
- Use only information present in the context. Do not add outside knowledge.
- If the context does not contain enough information to answer, say exactly:
"I don't have enough information in the course material to answer that."
- Keep the answer clear and concise

<COURSE_CONTEXT>
{context}
</COURSE_CONTEXT>

<STUDENT_QUESTION>
{question}
</STUDENT_QUESTION>

Answer:
"""
)


chain = prompt | llm | StrOutputParser()


def generate(query: str, context: list[str]) -> str:
    """Generate a grounded answer from the query and context chunks."""
    context_text = "\n\n".join(context)
    return chain.invoke({"question": query, "context": context_text})


def generate_stream(query: str, context: list[str]):
    """
    Stream the grounded answer chunk-by-chunk as it is generated.

    Same prompt / model / chain as generate() — we just call .stream() instead
    of .invoke(). Because the chain ends in StrOutputParser(), each yielded
    chunk is already a plain str, so no .content unpacking is needed.

    Yields:
        str: successive pieces of the answer. Empty chunks are skipped so the
             caller can clock time-to-first-token on the first *visible* token.
    """
    context_text = "\n\n".join(context)
    for chunk in chain.stream({"question": query, "context": context_text}):
        if chunk:                      # skip empty leading chunks
            yield chunk


if __name__ == "__main__":
    ctx = [
        "Online eval means evaluating your system on live production traffic "
        "after deployment. It works without an answer key, unlike offline eval."
    ]

    # non-streaming
    print(generate("what is online eval?", ctx))

    # streaming (prints tokens as they arrive)
    print("\n--- streaming ---")
    for piece in generate_stream("what is online eval?", ctx):
        print(piece, end="", flush=True)
    print()