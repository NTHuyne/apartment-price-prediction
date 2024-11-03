from ...utils import *
from ..base_crawler import BaseCrawler
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, Timeout, RequestException
from urllib3.util.retry import Retry
import time


class AlonhadatCrawler(BaseCrawler):
    def __init__(self, url):
        super().__init__(url)
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    def fetch_data(self):
        
        item_urls = []
        apartments = []

        session = requests.Session()
        retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        base_url = f"{self.url}/nha-dat/can-ban/can-ho-chung-cu/1/ha-noi/trang--{{}}.html"
    
        # Loop through pages
        for page_num in range(3, 300):
            url = base_url.format(page_num)

            try:
                # Request each page with a timeout
                response = session.get(url, headers=self.headers, timeout=1)
                response.raise_for_status()  # Raise exception for 4xx/5xx errors
                soup = BeautifulSoup(response.text, 'html.parser')

                pre_url = 'https://alonhadat.com.vn'
                for div_tag in soup.find_all('div', class_='ct_title'):
                    a_tag = div_tag.find('a')  # Find the <a> tag within the <div class="ct_title">
                    if a_tag and 'href' in a_tag.attrs:  # Ensure the <a> tag and href attribute exist
                        full_url = pre_url + a_tag['href']
                        item_urls.append(full_url)

                # Delay between page requests
                time.sleep(1)  # 2-second delay to avoid hitting the server too fast
            
            except (ConnectionError, Timeout) as e:
                print(f"Error fetching page {page_num}: {e}")
                continue  # Skip to the next page

        # Loop through item URLs and extract data
        for urls in item_urls:
            try:
                apartment_response = session.get(urls, headers=self.headers, timeout=1)
                apartment_response.raise_for_status()
                apartment_soup = BeautifulSoup(apartment_response.content, 'html.parser')

                table = apartment_soup.find('table')
                cells = table.find_all('td')

                title = apartment_soup.find('div', class_='title').find('h1').text.strip()
                price = apartment_soup.find('span', class_='price').find('span', class_='value').text.strip()
                area = apartment_soup.find('span', class_='square').find('span', class_='value').text.strip()
                address = apartment_soup.find('div', class_='address').find('span', class_='value').text.strip()

                huong = cells[3].text
                phongAn = cells[5].text
                dgTrcNha = cells[9].text
                nhaBep = cells[11].text
                type = cells[13].text
                phapLy = cells[15].text
                sanThg = cells[17].text
                chNgang = cells[19].text
                soLau = cells[21].text
                carSlot = cells[23].text
                chDai = cells[25].text
                noBed = cells[27].text
                chinhChu = cells[29].text

                product = {
                    'title': title,
                    'price': price,
                    'area': area,
                    'address': address,
                    'huong': huong,
                    'phongAn': phongAn,
                    'dgTrcNha': dgTrcNha,
                    'nhaBep': nhaBep,
                    'type': type,
                    'phapLy': phapLy,
                    'sanThg': sanThg,
                    'chNgang': chNgang,
                    'soLau': soLau,
                    'carSlot': carSlot,
                    'chDai': chDai,
                    'noBed': noBed,
                    'chinhChu': chinhChu
                }
                apartments.append(product)

                # Delay between item requests
                time.sleep(1)  # 1-second delay between item requests

            except (ConnectionError, Timeout, RequestException) as e:
                print(f"Error fetching item {urls}: {e}")
                continue  # Skip to the next item
        return apartments
    
    def run(self):
        raw_data = self.fetch_data()
        create_dir('data/raw/alonhadat')
        save_to_file(raw_data, 'data/raw/alonhadat/alonhadat.json')
        return raw_data

if __name__ == "__main__":
    alonhadatCrawler = AlonhadatCrawler("https://alonhadat.com.vn")
    alonhadatCrawler.run()

"""
from ...utils import *
from ..base_crawler import BaseCrawler
import time
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, Timeout, RequestException
from urllib3.util.retry import Retry

import requests
from bs4 import BeautifulSoup
import json


session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246'
}

response = session.get('https://alonhadat.com.vn', headers=headers)
print(response.status_code)

item_urls = []
apartments = []

if response.status_code == 200:
    base_url = 'https://alonhadat.com.vn/nha-dat/can-ban/can-ho-chung-cu/1/ha-noi/trang--{}.html'
    
    # Loop through pages
    for page_num in range(3, 6):
        url = base_url.format(page_num)

        try:
            # Request each page with a timeout
            response = requests.get(url, headers=headers, timeout=1)
            response.raise_for_status()  # Raise exception for 4xx/5xx errors
            soup = BeautifulSoup(response.content, 'html.parser')

            pre_url = 'https://alonhadat.com.vn'
            for div_tag in soup.find_all('div', class_='ct_title'):
                a_tag = div_tag.find('a')  # Find the <a> tag within the <div class="ct_title">
                if a_tag and 'href' in a_tag.attrs:  # Ensure the <a> tag and href attribute exist
                    full_url = pre_url + a_tag['href']
                    item_urls.append(full_url)

            # Delay between page requests
            time.sleep(5)  # 2-second delay to avoid hitting the server too fast
        
        except (ConnectionError, Timeout) as e:
            print(f"Error fetching page {page_num}: {e}")
            continue  # Skip to the next page

    # Loop through item URLs and extract data
    for urls in item_urls:
        try:
            apartment_response = requests.get(urls, headers=headers, timeout=1)
            apartment_response.raise_for_status()
            apartment_soup = BeautifulSoup(apartment_response.text, 'html.parser')

            table = apartment_soup.find('table')
            cells = table.find_all('td')

            title = apartment_soup.find('div', class_='title').find('h1').text.strip()
            price = apartment_soup.find('span', class_='price').find('span', class_='value').text.strip()
            area = apartment_soup.find('span', class_='square').find('span', class_='value').text.strip()
            address = apartment_soup.find('div', class_='address').find('span', class_='value').text.strip()

            duAn = apartment_soup.find('span', class_='project')
            if duAn:
                duAn = duAn.find('a').text.strip()
            else:
                duAn = 'N/A'  # Fallback value when `duAn` is missing

            try:
            # Ensure we can access the required indices
                huong = cells[3].text if len(cells) > 3 else 'N/A'
                phongAn = cells[5].text if len(cells) > 5 else 'N/A'
                dgTrcNha = cells[9].text if len(cells) > 9 else 'N/A'
                nhaBep = cells[11].text if len(cells) > 11 else 'N/A'
                type = cells[13].text if len(cells) > 13 else 'N/A'
                phapLy = cells[15].text if len(cells) > 15 else 'N/A'
                sanThg = cells[17].text if len(cells) > 17 else 'N/A'
                chNgang = cells[19].text if len(cells) > 19 else 'N/A'
                soLau = cells[21].text if len(cells) > 21 else 'N/A'
                carSlot = cells[23].text if len(cells) > 23 else 'N/A'
                chDai = cells[25].text if len(cells) > 25 else 'N/A'
                noBed = cells[27].text if len(cells) > 27 else 'N/A'
                chinhChu = cells[29].text if len(cells) > 29 else 'N/A'

                product = {
                    'title': title,
                    'price': price,
                    'area': area,
                    'address': address,
                    'huong': huong,
                    'duAn': duAn,
                    'phongAn': phongAn,
                    'dgTrcNha': dgTrcNha,
                    'nhaBep': nhaBep,
                    'type': type,
                    'phapLy': phapLy,
                    'sanThg': sanThg,
                    'chNgang': chNgang,
                    'soLau': soLau,
                    'carSlot': carSlot,
                    'chDai': chDai,
                    'noBed': noBed,
                    'chinhChu': chinhChu
                }
                apartments.append(product)
        
            except IndexError as e:
                print(f"Skipping item {urls} due to missing data: {e}")
                continue  # Skip to the next item

            # Delay between item requests
            time.sleep(1)  # 1-second delay between item requests

        except (ConnectionError, Timeout, RequestException) as e:
            print(f"Error fetching item {urls}: {e}")
            continue  # Skip to the next item

# Save data to JSON file
print(len(item_urls), len(apartments))
with open('alonhadat2.json', 'w', encoding='utf-8') as json_file:
    json.dump(apartments, json_file, ensure_ascii=False, indent=4)

print("Data written")

class HomedyCrawler(BaseCrawler):
    def fetch_data(self):
        pass
"""