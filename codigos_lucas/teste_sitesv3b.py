import feedparser
import openai
import requests
import nltk
import numpy as np
import tensorflow as tf

from sumy.parsers.html import HtmlParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore

# Baixar as stopwords em português, se ainda não tiver sido feito
nltk.download('stopwords')

# Defina sua chave da API do OpenAI
openai.api_key = '---'

# Função para obter o conteúdo HTML de um link
def get_html_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Erro ao obter o conteúdo HTML do URL {url}: {e}")
        return None

# Função para coletar todas as palavras de um texto
def collect_words(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    return words

# Coleta todas as palavras das notícias
def collect_all_words(rss_feeds):
    all_words = []
    for site, rss_feed in rss_feeds.items():
        feed = feedparser.parse(rss_feed)
        for entry in feed.entries:
            html_content = get_html_content(entry.link)
            if html_content:
                soup = BeautifulSoup(html_content, 'html.parser')
                text = soup.get_text()
                words = collect_words(text)
                all_words.extend(words)
    return all_words


# Lista para armazenar os títulos das notícias
def get_titles_list(rss_feeds, max_news=5):
    titles_list = []
    for site, rss_feed in rss_feeds.items():
        feed = feedparser.parse(rss_feed)
        for i, entry in enumerate(feed.entries[:max_news]):
            titles_list.append(entry.title)
    return titles_list

# Lista de feeds RSS dos sites que serão buscadas as notícias
rss_feeds = {
    "Greenpeace Brasil": "https://www.greenpeace.org/brasil/feed/",
    "G1 - Natureza": "https://g1.globo.com/rss/g1/natureza/",
    "Instituto Akatu": "https://www.akatu.org.br/feed/",
    "O Eco": "https://www.oeco.org.br/feed/",
}

# Lista de palavras relacionadas ao meio ambiente que queremos buscar
environment_related_words = [
    "queimadas", "enchente", "poluição", "desmatamento", "biodiversidade", "sustentabilidade",
    "reciclagem", "desflorestamento", "aquecimento global", "preservação", "conservação",
    "impacto ambiental", "energia renovável", "carbono", "plástico", "floresta", "água",
    "ar", "fauna", "flora", "clima", "contaminação", "resíduos", "ecossistema",
    "despejo ilegal de resíduos", "extração ilegal de madeira", "caça furtiva", "pesca predatória",
    "tráfico de animais silvestres", "despejo de produtos químicos tóxicos", "vazamento de petróleo",
    "incêndios criminosos", "corrupção ambiental", "desvio de recursos naturais",
    "exploração ilegal de recursos naturais", "contrabando de espécies protegidas", "urbanização descontrolada",
    "construções irregulares em áreas de preservação", "agronegócio predatório", "mineração ilegal",
    "pescaria ilegal em áreas protegidas", "envenenamento de rios e mananciais", "contaminação de alimentos",
    "desmatamento para fins comerciais", "uso indiscriminado de agrotóxicos", "sobreexplotação de recursos hídricos",
    "evasão de fiscalização ambiental", "suborno para autorizações ambientais", "impacto ambiental de indústrias poluidoras",
    "destruição de habitats naturais", "assoreamento de rios e lagos", "atropelamento de fauna silvestre",
    "derramamento de lixo em áreas protegidas", "exploração sexual de recursos naturais",
    "pesquisa científica ilegal em fauna e flora", "furtos de patrimônio natural",
    "introdução de espécies exóticas invasoras", "grilagem de terras", "contrabando de produtos florestais",
    "roubo de madeira", "extração ilegal de areia e cascalho", "modificação genética de espécies",
    "descarte irregular de produtos eletrônicos", "produção clandestina de carvão vegetal",
    "pesca com explosivos ou veneno",
    "desmatamento", "queimada", "incêndio", "inundações", "poluente", "resíduo", "erosão", "desertificação",
    "contaminante", "contaminação", "degradação", "seca", "salinização", "desertificação", "assoreamento",
    "impacto", "alteração", "degradante", "emissão"
]

# Lista para armazenar os títulos das notícias
titles_list = []

# Função para imprimir as principais notícias de um feed RSS com resumo

def print_top_news_rss_with_summary(site_name, url, max_news=5):
    global titles_list  # Usando a lista global dentro da função

    # Analisa o feed RSS
    feed = feedparser.parse(url)

    # Imprime cabeçalho
    print(f"Principais notícias de {site_name}:")
    print("="*50)

    # Itera pelas entradas do feed (notícias)
    for i, entry in enumerate(feed.entries[:max_news]):
        print(f"Notícia {i+1}:")
        print(entry.title, '.')  # Imprime o título da notícia
        titles_list.append(entry.title)  # Adiciona o título à lista
        print("\nResumo:")
        summary = get_summary(entry.link)  # Obtém o resumo da notícia
        print(summary)  # Imprime o resumo
        print("-"*50)


# Função para obter o resumo do conteúdo de um link
def get_summary(url):
    parser = HtmlParser.from_url(url, Tokenizer(
        "portuguese"))  # Parseia o HTML da página
    summarizer = LexRankSummarizer()  # Inicializa o sumarizador
    # Obtém um resumo do conteúdo
    summary = summarizer(parser.document, sentences_count=3)
    # Retorna o resumo como uma string
    return ' '.join([str(sentence) for sentence in summary])

# Coletar todas as palavras das notícias
all_words = collect_all_words(rss_feeds)

# Define as stopwords em português
stop_words = set(stopwords.words('portuguese'))

# Remove stopwords
filtered_words = [word for word in all_words if word not in stop_words]

# Identificar e contar as palavras relacionadas ao meio ambiente
environment_words = [
    word for word in filtered_words if word in environment_related_words]
word_freq = nltk.FreqDist(environment_words)

# Obter as palavras mais comuns e suas frequências
top_words, frequencies = zip(*word_freq.most_common(10))

# Dividir os dados em conjuntos de treinamento e teste
words_array = np.array(top_words)
frequencies_array = np.array(frequencies)
X_train, X_test, y_train, y_test = train_test_split(
    words_array, frequencies_array, test_size=0.2, random_state=42)

# Convert text data to numerical data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Find the length of the longest sequence in X_train
sequence_length = max([len(seq) for seq in X_train])

# Pad sequences
X_train = pad_sequences(X_train, padding='post', maxlen=sequence_length)
X_test = pad_sequences(X_test, padding='post', maxlen=sequence_length)

# Also make sure to convert your labels to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Implementação da rede neural
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

loss, mae = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, MAE: {mae}')

predictions = model.predict(X_test)

# Imprimir a lista de títulos e suas frequências
prompt_text = ""  # Define a variável prompt_text
prompt_text += "Palavras e suas frequências:\n"
for word, freq in zip(top_words, frequencies):
    prompt_text += f"- {word}: {freq} vezes\n"

# Convertendo as palavras em formato de dicionário para facilitar o envio para o GPT
word_dict = dict(zip(top_words, frequencies))

# Convertendo a lista de títulos em uma única string
titles_string = '\n'.join(titles_list)

# Adicionar os títulos ao prompt_text
prompt_text += "\nTítulos das notícias coletadas:\n\n"
prompt_text += titles_string
prompt_text += "\n\nO que você pode concluir a respeito? Quero uma análise detalhada, ampla, falando das notícias, números e também das implicações, tendências e possíveis ações a serem tomadas em relação ao tema."

print("Texto para GPT:\n", prompt_text)

# Enviando o texto para o GPT para gerar o relatório
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # Tentar um modelo diferente
    messages=[
        {"role": "user", "content": prompt_text},
    ],
    temperature=0.85,  # Controla a aleatoriedade da saída do modelo
    max_tokens=1200  # Limita o comprimento da saída do modelo
)

summary = "No output from the model"

if response and response.choices and len(response.choices) > 0:
    summary = response.choices[0].message['content'].strip()

print("\n\nRelatório das notícias relacionadas ao Meio Ambiente:")
print('\n', summary)


"""
1.Importa as bibliotecas necessárias, como feedparser, openai, requests, nltk, numpy, tensorflow, sumy, BeautifulSoup, entre outras.
2.Define funções para obter o conteúdo HTML de um link, coletar todas as palavras de um texto, imprimir as principais notícias de um feed RSS com resumo, obter o resumo do conteúdo de um link, entre outras funções auxiliares.
3.Baixa as stopwords em português usando o NLTK.
4.Define uma chave de API para o OpenAI.
5.Cria uma lista de feeds RSS de sites que contêm notícias relacionadas ao meio ambiente e uma lista de palavras relacionadas ao meio ambiente que serão buscadas nas notícias.
6.Define uma lista para armazenar os títulos das notícias.
7.Itera sobre os feeds RSS, coleta o conteúdo HTML de cada notícia, extrai o texto, coleta todas as palavras das notícias, remove as stopwords e identifica e conta as palavras relacionadas ao meio ambiente.
8.Divide os dados em conjuntos de treinamento e teste, converte os dados de texto em dados numéricos usando Tokenizer e padroniza as sequências.
9.Implementa uma rede neural utilizando o TensorFlow e o Keras para prever a frequência das palavras relacionadas ao meio ambiente.
10.Cria um texto para enviar ao GPT (OpenAI's Generative Pre-trained Transformer) contendo as palavras e suas frequências, os títulos das notícias coletadas e uma solicitação para uma análise detalhada das notícias.
11.Envia o texto para o GPT para gerar o relatório das notícias relacionadas ao meio ambiente, incluindo análise, números, implicações, tendências e possíveis ações a serem tomadas.
12.Imprime o relatório gerado pelo GPT.

No geral, o código é um exemplo de como coletar notícias relacionadas a um tópico específico, analisar essas notícias e gerar um relatório automatizado utilizando técnicas de processamento de linguagem natural e aprendizado de máquina.

"""