import time
import random
import requests
from bs4 import BeautifulSoup, Comment

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


def is_visible_text(element):
    """Filter function to exclude invisible or script/style text."""
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def extract_main_text(soup):
    """
    Attempts to extract the main textual content from a parsed HTML soup.
    Prioritizes <article>, <main>, and the largest div by text content.
    """
    # Try common semantic containers
    for tag in ['article', 'main']:
        section = soup.find(tag)
        if section:
            texts = section.find_all(string=True)
            visible_texts = filter(is_visible_text, texts)
            return '\n'.join(t.strip() for t in visible_texts if t.strip())

    # Fallback: choose the <div> with the most visible text
    divs = soup.find_all('div')
    max_text = ''
    for div in divs:
        texts = div.find_all(string=True)
        visible_texts = list(filter(is_visible_text, texts))
        combined = '\n'.join(t.strip() for t in visible_texts if t.strip())
        if len(combined) > len(max_text):
            max_text = combined

    return max_text

def fetch_and_parse(url):
    """
    Fetches the content of a URL, parses it using BeautifulSoup, and extracts meaningful text.

    Args:
        url: The URL to fetch.

    Returns:
        A string containing the main textual content of the page, or None if an error occurs.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()

        text_content = extract_main_text(soup)
        return text_content if text_content.strip() else None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        print(f"Error parsing content: {e}")
        return None