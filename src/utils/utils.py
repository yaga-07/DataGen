import time
import random

def backoff_retry(func, max_retries=3, base_delay=1, max_delay=10, exceptions=(Exception,), logger=None, *args, **kwargs):
    """
    Retry a function with exponential backoff.

    Args:
        func: The function to call.
        max_retries: Maximum number of retries.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        exceptions: Tuple of exception classes to catch.
        logger: Optional logger for warnings.
        *args, **kwargs: Arguments to pass to func.

    Returns:
        The result of func(*args, **kwargs) if successful.

    Raises:
        The last exception if all retries fail.
    """
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            if attempt == max_retries:
                if logger:
                    logger.error(f"Max retries reached. Raising exception: {e}")
                raise
            if logger:
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, max_delay)
