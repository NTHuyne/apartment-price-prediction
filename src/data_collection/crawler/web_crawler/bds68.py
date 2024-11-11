import requests
import sys
from bs4 import BeautifulSoup
import os
import json 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_crawler import BaseCrawler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import save_to_file, create_dir

class Bds68Crawler(BaseCrawler):
    def __init__(self, url):
        super().__init__(url)
        self.total_pages = self.get_total_pages() 

    def get_total_pages(self):
        try:
            response = requests.get(self.url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                last_page_link = soup.find('a', title='Go to last Page')
                if last_page_link:
                    return int(last_page_link['href'].split('=')[-1])
                else:
                    print("No last page link found, defaulting to 1 page.")
                    return 1
            else:
                print(f"Failed to retrieve the main page for pagination (Status code: {response.status_code})")
                return 1
        except Exception as e:
            print(f"An error occurred while fetching total pages: {e}")
            return 1

    def fetch_data(self, page=0):
        try:
            page_url = f"{self.url}?pg={page}"
            response = requests.get(page_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                a_elements = soup.select('.div_title_mobile a[href^="/ban-chung-cu/ha-noi/"]')
                data = [{'text': a.get_text(strip=True), 'url': a['href']} for a in a_elements]
                if data:
                    return data
                else:
                    print(f"No data found on page {page}.")
                    return []
            else:
                print(f"Failed to retrieve data from page {page} (Status code: {response.status_code})")
                return []
        except Exception as e:
            print(f"An error occurred on page {page}: {e}")
            return []

    def fetch_detailed_data(self, relative_url):
        try:
            full_url = f"https://bds68.com.vn{relative_url}"
            response = requests.get(full_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                features_div = soup.find('div', class_='prop-features')
                if not features_div:
                    print(f"No prop-features found at {full_url}")
                    return None
                
                title_tag = soup.find('h1', class_='detail-prop-title')
                title = title_tag.get_text(strip=True) if title_tag else ''

                loai_tin_rao = features_div.find(string=lambda x: 'Loại Tin Rao:' in x)
                loai_tin_rao = loai_tin_rao.replace('Loại Tin Rao: ', '').strip() if loai_tin_rao else ''

                location_div = soup.find('div', class_='prop-address')
                location = location_div.find('span').get_text(strip=True) if location_div else ''
                
                du_an_div = features_div.find(string=lambda x: 'Dự Án:' in x)
                if du_an_div:
                    project_div = du_an_div.find_parent('div') 
                    project_link = project_div.find('a')
                    du_an = project_link.get_text(strip=True) if project_link else ''
                else:
                    du_an = ''
                
                price_div = features_div.find('div', string=lambda x: x and 'Giá:' in x)
                price = price_div.get_text(strip=True).replace('Giá: ', '').strip() if price_div else ''
                
                area_div = features_div.find('div', string=lambda x: x and 'Diện Tích:' in x)
                area = area_div.get_text(strip=True).replace('Diện Tích: ', '').strip() if area_div else ''

                description_div = soup.find('div', class_='prop-description')
                description_tag = description_div.find('p') if description_div else None
                description = description_tag.get_text(strip=True).replace('\n', ' ') if description_tag else ''

                bedrooms_div = features_div.find(string=lambda x: 'Số Phòng Ngủ:' in x)
                bedrooms = bedrooms_div.replace('Số Phòng Ngủ: ', '').strip() if bedrooms_div else ''
                
                bathrooms_div = features_div.find(string=lambda x: 'Số Phòng Tắm:' in x)
                bathrooms = bathrooms_div.replace('Số Phòng Tắm: ', '').strip() if bathrooms_div else ''

                known_attributes = {'Loại Tin Rao', 'Dự Án', 'Giá', 'Diện Tích'}

                attributes = soup.find_all('div', class_='col-sm-6 col-xs-12')
                variable_data = {}
                for attribute in attributes:
                    attribute_text = attribute.get_text(strip=True)
                    if ':' in attribute_text:
                        key, value = attribute_text.split(':', 1)
                        key = key.strip()
                        if key not in known_attributes:  
                            variable_data[key] = value.strip()
                
                data =  {
                    'Url': full_url,
                    'Source': 'https://bds68.com.vn/',
                    'Tên': title,
                    'Loại tin rao': loai_tin_rao,
                    'Dự án': du_an,
                    'Vị trí': location,
                    'Giá': price,
                    'Diện tích': area,
                    'Mô tả': description
                }

                data.update(variable_data)
                return data
            else:
                print(f"Failed to retrieve detailed data from {full_url} (Status code: {response.status_code})")
                return None
        except Exception as e:
            print(f"An error occurred while fetching detailed data from {full_url}: {e}")
            return None

    def fetch_all_pages(self):
        all_data = []
        for page in range(1, self.total_pages + 1):
            print(f"Fetching page {page}...")
            page_data = self.fetch_data(page=page)
            if not page_data: 
                print(f"No data found on page {page}. Continuing to the next page.")
                continue
            for item in page_data:
                detailed_data = self.fetch_detailed_data(item['url'])
                if detailed_data:
                    all_data.append(detailed_data)
        return all_data

if __name__ == "__main__":
    url = "https://bds68.com.vn/ban-chung-cu/ha-noi"
    crawler = Bds68Crawler(url)
    print(f"Total pages: {crawler.total_pages}")
    all_property_data = crawler.fetch_all_pages()  
    create_dir('data/raw/bds68')
    save_to_file(all_property_data, "data/raw/bds68/bds68.json")
    print(f"Total properties retrieved: {len(all_property_data)}")
