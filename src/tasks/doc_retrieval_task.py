from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core import BaseLLM

import random
import json
from typing import List, Dict
from ..prompts.doc_retrieval_prompts import DOC_RET_SYS_PROMPT, DOC_RET_USER_PROMPT
from ..core.base_task import BaseTask
from src.utils.color_logger import get_color_logger
from src.utils.utils import backoff_retry
from src.utils.data_saver import save_data

logger = get_color_logger(name=__name__,)

class DocumentRetrievalTask(BaseTask):
    """
    Task for generating Document Retrieval data.
    """

    def __init__(self, model: "BaseLLM", domain: str, num_records: int, save_intermediate_queries: bool = False):
        super().__init__(model, domain, num_records)
        self.save_intermediate_queries = save_intermediate_queries

    def _generate_search_queries(self) -> List[Dict]:
        """
        Generates intermediate search queries for Document Retrieval data using the provided LLM model.

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
                        "content": DOC_RET_SYS_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": DOC_RET_USER_PROMPT.replace("{{num_records}}", str(current_batch_size)).replace("{{domain}}", self.domain),
                    }
                ]
                logger.debug(f"Prompt (batch {generated // batch_size + 1}, attempt {attempt+1}):\n{json.dumps(prompt, indent=2)}")
                def call_llm():
                    return self.model.generate_response(prompt)
                try:
                    response = backoff_retry(
                        call_llm,
                        max_retries=max_retries,
                        base_delay=1,
                        max_delay=8,
                        exceptions=(Exception,),
                        logger=logger
                    )
                    logger.debug(f"Raw LLM response (batch {generated // batch_size + 1}, attempt {attempt+1}):\n{response}")
                    batch_sentences = eval(response)
                    if isinstance(batch_sentences, list) and all(isinstance(s, str) for s in batch_sentences):
                        sentences = batch_sentences
                        generated += len(sentences)
                        results.extend(sentences)
                        break
                except Exception as e:
                    logger.warning(f"Batch {generated // batch_size + 1}, attempt {attempt+1} failed: {e}")
                    sentences = []
        self.intermediate_queries = results
        if self.save_intermediate_queries:
            save_data(
                data=results,
                folder=f"intermediate_queries_{self.__class__.__name__}",
                filename=f"intermediate_queries_{self.domain}.json",
                format="jsonl",
            )
            logger.info(f"Saved intermediate queries to intermediate_queries/intermediate_queries_{self.domain}.json")
        else:
            logger.info("Intermediate queries not saved.")
        return results
    
    def generate_data(self) -> List[Dict]:
        """
        Generates Document Retrieval data using the provided LLM model.

        Returns:
            List[Dict]: List of structured data samples, each as a dictionary.
        """
        pass
