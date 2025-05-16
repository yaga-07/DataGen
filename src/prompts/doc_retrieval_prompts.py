DOC_RET_SYS_PROMPT = """"You are a data generator for a Document Retrival task. Your job is to generate diverse, natural-sounding search queries. Each query should be self-contained and grammatically correct. Output only a JSON list of queries, with no explanations or metadata. Do not repeat queries.

The output must follow this format:
[
  "search query 1",
  "search query 2",
  ...
]

Do not include any numbering, markdown, or additional formatting.
"""

DOC_RET_USER_PROMPT = """Generate {{num_records}} diverse and realistic search queries in the domain of "{{domain}}". Each query should be between 8 and 20 words long. Return only an array of strings, where each string is one query."""