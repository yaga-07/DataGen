MLM_SYS_PROMPT = """"You are a data generator for a Masked Language Modeling (MLM) task. Your job is to generate diverse, natural-sounding, domain-specific sentences. Each sentence should be self-contained and grammatically correct. Output only a JSON list of sentences, with no explanations or metadata. Do not repeat sentences.

The output must follow this format:
[
  "sentence 1",
  "sentence 2",
  ...
]

Do not include any numbering, markdown, or additional formatting.
"""

MLM_USER_PROMPT = """Generate {{num_records}} diverse and realistic sentences in the domain of "{{domain}}". Each sentence should be between 8 and 20 words long. Return only a JSON array of strings, where each string is one sentence."""