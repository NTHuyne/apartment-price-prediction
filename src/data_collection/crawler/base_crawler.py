from abc import ABC, abstractmethod

class BaseCrawler(ABC):
    def __init__(self, url):
        """Initializes a BaseCrawler with a given url

        Args:
            url (str): The url to be crawled
        Raises:
            ValueError: If the url is not a string"""
        self.url = url
    
    @abstractmethod
    def fetch_data(self):
        """This method should be overridden by subclasses to fetch data from a given url
        
        Returns:
            list of dict: A list of dictionaries where each dictionary represents a single item
        """
        pass

    def run(self):
        raw_data = self.fetch_data()
        return raw_data