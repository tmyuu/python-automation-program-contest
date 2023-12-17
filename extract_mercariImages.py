from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
import os
from selenium.common.exceptions import TimeoutException
import re

def extract_id_with_prefix_from_url(url):
    pattern = r"/(m\d+)_"
    match = re.search(pattern, url)
    return match.group(1) if match else ""

def find_index_of_matching_element(category_list, query):
    try:
        return category_list.index(query)
    except ValueError:
        return -1

def mercari_scraping(response):
    category_list = ['スニーカー', 'スリッポン', 'サンダル', 'パンプス', 'ブーツ', 'ドレスシューズ', 'ローファー', '長靴']
    category_list_eng = ['sneakers', 'slip-on', 'sandal', 'pumps', 'boots', 'dress-shoes', 'loafers', 'rain-shoes']
    search_query = category_list[find_index_of_matching_element(category_list_eng, response)]
    
    options = webdriver.ChromeOptions()
    service = webdriver.chrome.service.Service(ChromeDriverManager().install())
    browser = webdriver.Chrome(service=service, options=options)

    image_dir = "/Users/mypc/Desktop/mercari_images"
    os.makedirs(image_dir, exist_ok=True)

    count = 0
    try:
        for page in range(3):  # 3ページ分処理
            page_token = f"v1%{page + 1}A"  # page_tokenの値を更新
            url = f"https://jp.mercari.com/search?keyword={search_query}&order=desc&sort=created_time&status=on_sale&page_token={page_token}"

            browser.get(url)
            time.sleep(6)

            try:
                image_elements = WebDriverWait(browser, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, f".sc-bcd1c877-2.gWHcGv img"))
                )
                image_urls = [element.get_attribute('src') for element in image_elements]
            except TimeoutException:
                image_elements = None

            for i, img in enumerate(image_elements):
                if count >= 115:
                    break
                img_url = img.get_attribute('src')
                if img_url:
                    response = requests.get(img_url)
                    image_path = os.path.join(image_dir, f'{extract_id_with_prefix_from_url(img_url)}.jpg')
                    with open(image_path, 'wb') as file:
                        file.write(response.content)
                count += 1
    except Exception as e:
        print(f"an error occurred: {e}")
    finally:
        browser.quit()
        print("Completed downloading images.")
