DOC_RET_SYS_PROMPT_Q = """"You are a data generator for a Document Retrival task. Your job is to generate diverse, natural-sounding search queries. Each query should be self-contained and grammatically correct. Output only a JSON list of queries, with no explanations or metadata. Do not repeat queries.

The output must follow this format:
[
  "search query 1",
  "search query 2",
  ...
]

Do not include any numbering, markdown, or additional formatting.
"""

DOC_RET_USER_PROMPT_Q = """Generate {{num_records}} diverse and realistic search queries in the domain of "{{domain}}". Each query should be between 8 and 20 words long. Return only an array of strings, where each string is one query."""

# -----------

DOC_RET_SYS_PROMPT_D = """
You are a data generation assistant that creates high-quality synthetic query-document pairs for training a retrieval model in a Retrieval-Augmented Generation (RAG) system.

Your job is to:
- Read the given web content.
- Extract meaningful and self-contained passages.
- Generate natural language queries that could retrieve each passage.
- Output each query-document pair in JSON Lines format.

Each passage must be relevant, well-formed, and ideally 2–4 sentences. The query should sound like a real user question and closely relate to the content in the passage.
"""

DOC_RET_USER_PROMPT_D = """
Generate synthetic query-document training data from the following web page content.

Return the output in JSON Lines (`.jsonl`) format, where each line has:
- "query": the generated question
- "document": the relevant passage

Web Content:
{WEB_CONTENT}
"""