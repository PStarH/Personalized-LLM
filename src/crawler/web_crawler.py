import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

class WebCrawler:
    """
    Crawls the web using Bing to retrieve data relevant to the user's content.
    """

    def __init__(self):
        # Initialize any required resources
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/85.0.4183.102 Safari/537.36'
        }

    def retrieve_data(self, query, max_results=10):
        """
        Retrieves data from Bing based on the query.

        Args:
            query (str): The search query to retrieve data for.
            max_results (int): Number of search results to retrieve.

        Returns:
            str: Aggregated crawled data from the retrieved URLs.
        """
        search_url = "https://www.bing.com/search"
        params = {'q': query, 'count': max_results}
        crawled_data = ""

        try:
            response = self.session.get(search_url, headers=self.headers, params=params)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            result_elements = soup.find_all('li', class_='b_algo')

            count = 0
            for element in result_elements:
                if count >= max_results:
                    break
                link = element.find('a')
                if link and 'href' in link.attrs:
                    url = link['href']
                    print(f"Retrieving content from: {url}")
                    page_content = self.fetch_page_content(url)
                    crawled_data += page_content + "\n"
                    count += 1
                    # Be polite and avoid overwhelming servers
                    time.sleep(1)
        except requests.RequestException as e:
            print(f"An error occurred while searching Bing: {e}")

        return crawled_data.strip()

    def fetch_page_content(self, url):
        """
        Fetches and extracts textual content from a given URL.

        Args:
            url (str): The URL of the webpage to fetch.

        Returns:
            str: Extracted textual content from the webpage.
        """
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return text
        except requests.RequestException as e:
            print(f"Failed to retrieve content from {url}: {e}")
            return ""