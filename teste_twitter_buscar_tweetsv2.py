from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time

# Configurando o serviço do ChromeDriver
service = Service(ChromeDriverManager().install())

# Instanciando o driver do Chrome
driver = webdriver.Chrome(service=service)

# URL do perfil do Twitter que você deseja extrair tweets
url = 'https://twitter.com/MercedesAMGF1BR'

# Abrindo a página no navegador
driver.get(url)

# Esperando até que os tweets sejam carregados (espera explícita)
try:
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.tweet')))
except Exception as e:
    print("Timeout! Os tweets não foram carregados a tempo.")
    driver.quit()
    exit()

# Extraindo o HTML da página
html = driver.page_source

# Parseando o HTML usando BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')

# Encontrando todos os elementos que contêm os tweets
tweet_containers = soup.find_all('div', class_='tweet')

# Iterando sobre os elementos para extrair o texto de cada tweet
for tweet in tweet_containers:
    tweet_text = tweet.find('div', class_='tweet-text').text.strip()
    print(tweet_text)

# Fechando o navegador
driver.quit()
