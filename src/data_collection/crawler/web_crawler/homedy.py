from ...utils import *
from ..base_crawler import BaseCrawler
import requests
from bs4 import BeautifulSoup

class HomedyCrawler(BaseCrawler):
    def __init__(self, url):
        super().__init__(url)
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    def fetch_data(self):
        all_data = []
        base_url = f"{self.url}/ban-can-ho-ha-noi/p1"
        while True:
            print(f"Crawling: {base_url}")
            response = requests.get(base_url, headers=self.headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                page_nav = soup.find('div', class_='page-nav')
                product_contents = soup.find_all('div', class_='product-content')
                all_links = []
                all_product_details = []
                for product in product_contents:
                    href = product.find('a', class_='title')['href']
                    all_links.append(href)
                    product_details = product.find('ul', class_='product-unit')
                    all_product_details.append(product_details)
                all_links = [f"{self.url}{link}" for link in all_links]
                for link, detail in zip(all_links, all_product_details):
                    price = detail.find('span', class_='price').text
                    acreage = detail.find('span', class_='acreage').text
                    if len(detail.find_all('li')) > 3:
                        unit_price = detail.find_all('li')[2].text
                    else:  
                        unit_price = ''
                    response = requests.get(link, headers=self.headers)
                    if response.status_code == 200:
                        try:
                            soup = BeautifulSoup(response.content, 'html.parser')
                            title = soup.find('div', class_='product-detail-top-left').find('h1').text
                            real_estate = soup.find('div', class_='address').find('a').text
                            address = ', '.join([span.text.strip() for span in soup.find('div', class_='address').find_all('span')])
                            description = soup.find('div', class_='description').text
                            data = {
                                'title': title,
                                'real_estate': real_estate,
                                'address': address,
                                'price': price,
                                'acreage': acreage,
                                'unit_price': unit_price,
                                'description': description
                            }
                            if soup.find('div', class_='product-attributes'):
                                attributes = soup.find_all('div', class_='product-attributes--item')
                                for attribute in attributes:
                                    key = attribute.find_all('span')[0].text
                                    value = attribute.find_all('span')[1].text
                                    data[key] = value
                            data['url'] = link
                            data['source'] = self.url
                            all_data.append(data)
                        except:
                            print(f'{link} failed')
                            continue
                    else:
                        continue
            else:
                print(f"Failed to retrieve the page {base_url}. Status code: {response.status_code}")
            if page_nav.find('a', rel='next'):
                base_url = f"{self.url}{page_nav.find('a', rel='next')['href']}"
                if base_url == "https://homedy.com/ban-can-ho-ha-noi/p201":
                    break
            else:
                break
        return all_data
    
    def run(self):
        raw_data = self.fetch_data()
        create_dir('data/raw/homedy')
        save_to_file(raw_data, 'data/raw/homedy/homedy.json')
        return raw_data

if __name__ == "__main__":
    homedyCrawler = HomedyCrawler("https://homedy.com")
    homedyCrawler.run()