from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core import BaseLLM

import random
import json
from math import ceil
from typing import List, Dict
from duckduckgo_search import DDGS
from src.prompts.doc_retrieval_prompts import (
    DOC_RET_SYS_PROMPT_Q, 
    DOC_RET_USER_PROMPT_Q,
    DOC_RET_SYS_PROMPT_D,
    DOC_RET_USER_PROMPT_D
    )
from src.core.base_task import BaseTask, AutoTask
from src.utils.color_logger import get_color_logger
from src.utils.utils import backoff_retry, fetch_and_parse, extract_json_from_markdown
from src.utils.data_saver import save_data

logger = get_color_logger(name=__name__, level="DEBUG")

@AutoTask.register("doc_retrieval")
class DocumentRetrievalTask(BaseTask):
    """
    Task for generating Document Retrieval data.
    """

    def __init__(self, model: "BaseLLM", domain: str, num_records: int, save_intermediate_results: bool = False):
        super().__init__(model, domain, num_records)
        self.save_intermediate_results = save_intermediate_results
        self.max_retries = 3
        self.batch_size = 50
        self.web_search_engine = DDGS()

    def _generate_search_queries(self) -> List[Dict]:
        """
        Generates intermediate search queries for Document Retrieval data using the provided LLM model.

        Returns:
            List[Dict]: List of structured data samples, each as a dictionary.
        """
        results = []
        total = ceil(self.num_records / 2)
        generated = 0

        while generated < total:
            current_batch_size = min(self.batch_size, total - generated)
            sentences = []
            for attempt in range(self.max_retries):
                prompt = [
                    {
                        "role": "system",
                        "content": DOC_RET_SYS_PROMPT_Q,
                    },
                    {
                        "role": "user",
                        "content": DOC_RET_USER_PROMPT_Q.replace("{{num_records}}", str(current_batch_size)).replace("{{domain}}", self.domain),
                    }
                ]
                logger.debug(f"Prompt (batch {generated // self.batch_size + 1}, attempt {attempt+1}):\n{json.dumps(prompt, indent=2)}")
                def call_llm():
                    return self.model.generate_response(prompt)
                try:
                    response = backoff_retry(
                        self.model.generate_response,
                        max_retries=self.max_retries,
                        base_delay=1,
                        max_delay=8,
                        exceptions=(Exception,),
                        logger=logger,
                        messages=prompt,
                    )
                    logger.debug(f"Raw LLM response (batch {generated // self.batch_size + 1}, attempt {attempt+1}):\n{response}")
                    batch_sentences = eval(response)
                    if isinstance(batch_sentences, list) and all(isinstance(s, str) for s in batch_sentences):
                        sentences = batch_sentences
                        generated += len(sentences)
                        results.extend(sentences)
                        break
                except Exception as e:
                    logger.warning(f"Batch {generated // self.batch_size + 1}, attempt {attempt+1} failed: {e}")
                    sentences = []
        self.intermediate_queries = results
        if self.save_intermediate_results:
            save_data(
                data=results,
                folder=f"intermediate_results_{self.__class__.__name__}",
                filename=f"intermediate_queries_{self.domain}.json",
                format="jsonl",
            )
            logger.info(f"Saved intermediate queries to intermediate_queries/intermediate_queries_{self.domain}.json")
        else:
            logger.info("Intermediate queries not saved.")
        return results
    
    def _get_search_queries_web(self, intermediate_queries: List[Dict]) -> List[Dict]:
        """
        This method does web search for inter mediate queries.
        Args:
            intermediate_queries (List[Dict]): List of intermediate queries.
        Returns:
            List[Dict]: List of structured data samples, each as a dictionary.
        """
        results = []
        for query in intermediate_queries:
            try:
                search_results=backoff_retry(
                        self.web_search_engine.text,
                        max_retries=self.max_retries,
                        base_delay=1,
                        max_delay=8,
                        exceptions=(Exception,),
                        logger=logger,
                        keywords=query,
                        max_results=1
                    )
                
                results.append({
                    "query": query,
                    "results": search_results
                })
            except Exception as e:
                logger.error(f"Error during web search for query '{query}': {e}")
                results.append({
                    "query": query,
                    "results": []
                })
        self.intermediate_queries_web = results
        if self.save_intermediate_results:
            save_data(
                data=results,
                folder=f"intermediate_results_{self.__class__.__name__}",
                filename=f"intermediate_queries_web_{self.domain}.json",
                format="jsonl",
            )
            logger.info(f"Saved intermediate queries to intermediate_queries_web/intermediate_queries_web_{self.domain}.json")
        else:
            logger.info("Intermediate queries not saved.")
        return results
    
    def _generate_document_retrieval_data(self, intermediate_web_results: List[Dict]) -> List[Dict]:
        """
        Generates Document Retrieval data using the provided LLM model.

        Args:
            intermediate_web_results (List[Dict]): List of intermediate web results.

        Returns:
            List[Dict]: List of structured data samples, each as a dictionary.
        """
        results = []
        generated = 0
        total = self.num_records
        max_outer_loops = 5  # Prevent infinite loop
        outer_loops = 0
        while generated < total and outer_loops < max_outer_loops:
            for web_rel in intermediate_web_results:
                if generated >= total:
                    break
                try:
                    web_content = web_rel.get("results", [])
                    if web_content:
                        body = web_content[0].get("body", "")
                        url = web_content[0].get("href", "")
                        if not url:
                            logger.warning(f"No URL found for query '{web_rel.get('query')}'.")
                            continue
                        main_text = fetch_and_parse(url)
                        if main_text:
                            bodymain_text = f"{body}\n\n{main_text}"
                            prompt = [
                                {
                                    "role": "system",
                                    "content": DOC_RET_SYS_PROMPT_D,
                                },
                                {
                                    "role": "user",
                                    "content": DOC_RET_USER_PROMPT_D.replace("{{WEB_CONTENT}}", bodymain_text),
                                }
                            ]
                            logger.debug(f"Prompt for document retrieval:\n{json.dumps(prompt, indent=2)}")
                            for attempt in range(self.max_retries):
                                try:
                                    def call_llm():
                                        return self.model.generate_response(prompt)
                                    response = backoff_retry(
                                        self.model.generate_response,
                                        max_retries=self.max_retries,
                                        base_delay=1,
                                        max_delay=8,
                                        exceptions=(Exception,),
                                        logger=logger,
                                        messages=prompt,
                                    )
                                    logger.debug(f"Raw LLM response (attempt {attempt+1}):\n{response}")
                                    response = extract_json_from_markdown(response)
                                    if not response:
                                        raise ValueError("No JSON found in the response.")
                                    batch_results = response
                                    if isinstance(batch_results, list):
                                        results.extend(batch_results)
                                        generated += len(batch_results)
                                        break
                                except Exception as e:
                                    logger.warning(f"Attempt {attempt+1} failed for document retrieval for query '{web_rel.get('query')}': {e}")
                                    continue
                        else:
                            logger.warning(f"No main text found for URL '{url}' in query '{web_rel.get('query')}'.")
                            continue
                    else:
                        logger.warning(f"No content found for query '{web_rel.get('query')}'.")
                        continue
                except Exception as e:
                    logger.error(f"Error during document retrieval for query '{web_rel.get('query')}': {e}")
            outer_loops += 1
            if generated >= total:
                break
            if outer_loops >= max_outer_loops:
                logger.warning(f"Reached max outer loop retries ({max_outer_loops}) in _generate_document_retrieval_data. Generated {generated} records out of {total}.")
                break
        # Truncate results if more than needed
        return results[:total]

    def generate_data(self) -> List[Dict]:
        """
        Generates Document Retrieval data using the provided LLM model.

        Returns:
            List[Dict]: List of structured data samples, each as a dictionary.
        """
        queries = self._generate_search_queries()
        logger.info(f"Generated {len(queries)} search queries.")
        queries_web = self._get_search_queries_web(queries)
        logger.info(f"Generated {len(queries_web)} search queries with web search.")
        doc_retrieval_data = self._generate_document_retrieval_data(queries_web)
        return doc_retrieval_data
