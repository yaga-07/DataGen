import random
import json
from typing import List, Dict
from ..prompts.mlm_prompts import MLM_SYS_PROMPT, MLM_USER_PROMPT
from ..core.base_task import BaseTask

class MLMTask(BaseTask):
    """
    Task for generating Masked Language Modeling (MLM) data.
    """

    def mask_text(self, text: str):
        words = text.split()
        candidates = [w for w in words if w[0].isalpha()]
        if not candidates:
            return None, None
        target = random.choice(candidates)
        masked = text.replace(target, "[MASK]", 1)
        return masked, text

    def generate_data(self) -> List[Dict]:
        """
        Generates MLM data using the provided LLM model.

        Returns:
            List[Dict]: List of structured data samples, each as a dictionary.
        """
        results = []
        max_retries = 3
        for _ in range(self.num_records):
            for attempt in range(max_retries):
                prompt = {
                    "system": MLM_SYS_PROMPT,
                    "user": MLM_USER_PROMPT.format(num_records=1, domain=self.domain)
                }
                response = self.model.generate_response(prompt)
                try:
                    sentences = json.loads(response)
                    if isinstance(sentences, list) and all(isinstance(s, str) for s in sentences):
                        break
                except Exception:
                    sentences = None
                if attempt == max_retries - 1:
                    sentences = []
            for sentence in sentences or []:
                masked, original = self.mask_text(sentence)
                if masked:
                    results.append({
                        "masked_text": masked,
                        "original_text": original
                    })
        return results
