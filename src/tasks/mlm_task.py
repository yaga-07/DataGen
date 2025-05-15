import random
import json
from typing import List, Dict
from ..prompts.mlm_prompts import MLM_SYS_PROMPT, MLM_USER_PROMPT
from ..core.base_task import BaseTask
from src.utils.color_logger import get_color_logger

logger = get_color_logger(name="mlm_task")

# List of common stopwords to avoid masking
STOPWORDS = {
    "is", "am", "are", "was", "were", "be", "been", "being",
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "for", "nor", "so", "yet", "to", "of", "in", "on", "at", "by",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "from", "up", "down", "out",
    "off", "over", "under", "again", "further", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "don't", "should", "now"
}

class MLMTask(BaseTask):
    """
    Task for generating Masked Language Modeling (MLM) data.
    """

    def __init__(self, model, domain, num_records, mask_pct: float = 0.15):
        super().__init__(model, domain, num_records)
        self.mask_pct = mask_pct

    def mask_text(self, text: str, mask_pct: float = None):
        """
        Masks a percentage of meaningful words in the text.

        Args:
            text (str): The input sentence.
            mask_pct (float): Percentage of eligible words to mask.

        Returns:
            Tuple[str, str]: (masked_text, original_text)
        """
        if mask_pct is None:
            mask_pct = self.mask_pct
        words = text.split()
        # Identify eligible words (not stopwords, alphabetic, length > 2)
        eligible_indices = [
            i for i, w in enumerate(words)
            if w.lower() not in STOPWORDS and w.isalpha() and len(w) > 2
        ]
        if not eligible_indices:
            return None, None
        num_to_mask = max(1, int(len(eligible_indices) * mask_pct))
        mask_indices = random.sample(eligible_indices, min(num_to_mask, len(eligible_indices)))
        masked_words = words[:]
        for idx in mask_indices:
            masked_words[idx] = "[MASK]"
        masked = " ".join(masked_words)
        return masked, text

    def generate_data(self) -> List[Dict]:
        """
        Generates MLM data using the provided LLM model.

        Returns:
            List[Dict]: List of structured data samples, each as a dictionary.
        """
        results = []
        max_retries = 3
        batch_size = 50
        total = self.num_records
        generated = 0

        while generated < total:
            current_batch_size = min(batch_size, total - generated)
            sentences = []
            for attempt in range(max_retries):
                prompt = [
                    {
                        "role": "system",
                        "content": MLM_SYS_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": MLM_USER_PROMPT.replace("{{num_records}}", str(current_batch_size)).replace("{{domain}}", self.domain),
                    }
                ]
                logger.debug(f"Prompt (batch {generated // batch_size + 1}, attempt {attempt+1}):\n{json.dumps(prompt, indent=2)}")
                response = self.model.generate_response(prompt)
                logger.debug(f"Raw LLM response (batch {generated // batch_size + 1}, attempt {attempt+1}):\n{response}")
                try:
                    batch_sentences = eval(response)
                    if isinstance(batch_sentences, list) and all(isinstance(s, str) for s in batch_sentences):
                        sentences = batch_sentences
                        break
                except Exception:
                    sentences = []
            # Only add up to the number of records needed
            for sentence in sentences:
                if generated >= total:
                    break
                masked, original = self.mask_text(sentence)
                if masked:
                    results.append({
                        "text": original,
                        "masked_text": masked,
                    })
                    generated += 1
        return results
