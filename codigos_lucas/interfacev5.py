import feedparser # Essa biblioteca é usada para analisar feeds RSS
import openai # Essa biblioteca é usada para interagir com a API do GPT-3
import requests # Essa biblioteca é usada para fazer solicitações HTTP
import nltk # Essa biblioteca é usada para processamento de linguagem natural
import numpy as np # Essa biblioteca é usada para cálculos numéricos
import tensorflow as tf # Essa biblioteca é usada para criar modelos de aprendizado de máquina
import matplotlib.pyplot as plt # Essa biblioteca é usada para plotar gráficos
import tkinter as tk # Essa biblioteca é usada para criar interfaces gráficas
import threading # Essa biblioteca é usada para executar tarefas em threads separadas
import praw # Essa biblioteca é usada para acessar a API do Reddit
import pandas as pd # Essa biblioteca é usada para manipular dados em formato de tabela
import subprocess # Essa biblioteca é usada para executar comandos do sistema operacional
import time # Essa biblioteca é usada para lidar com tempo

from requests.adapters import HTTPAdapter # Essa biblioteca é usada para adaptadores HTTP
from urllib3.util.retry import Retry # Essa biblioteca é usada para tentativas de solicitação
from datetime import datetime # Essa biblioteca é usada para manipular datas e horas
from tkinter import ttk # Essa biblioteca é usada para criar widgets de interface gráfica
from tkinter import scrolledtext # Essa biblioteca é usada para criar widgets de texto roláveis
from sumy.parsers.html import HtmlParser # Essa biblioteca é usada para analisar conteúdo HTML
from sumy.summarizers.lex_rank import LexRankSummarizer # Essa biblioteca é usada para resumir texto
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from bs4 import BeautifulSoup # Essa biblioteca é usada para analisar conteúdo HTML
from nltk.corpus import stopwords # Essa biblioteca é usada para obter stopwords
from nltk.tokenize import word_tokenize # Essa biblioteca é usada para tokenização de palavras
from sklearn.model_selection import train_test_split # Essa biblioteca é usada para dividir os dados em conjuntos de treinamento e teste
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer # type: ignore
from textblob import TextBlob # Essa biblioteca é usada para análise de sentimentos
from sumy.nlp.tokenizers import Tokenizer # Essa biblioteca é usada para tokenização de texto
from sumy.nlp.stemmers import Stemmer # Essa biblioteca é usada para stemização de palavras
from sumy.utils import get_stop_words # Essa biblioteca é usada para obter stopwords
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer # Essa biblioteca é usada para resumir texto
from sumy.parsers.html import HtmlParser # Essa biblioteca é usada para analisar conteúdo HTML
from sumy.nlp.tokenizers import Tokenizer # Essa biblioteca é usada para tokenização de texto
from tkinter import Scrollbar, Text, messagebox # Essa biblioteca é usada para criar widgets de texto roláveis e caixas de diálogo
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # Essa biblioteca é usada para exibir gráficos em interfaces gráficas
import seaborn as sns # Essa biblioteca é usada para plotar gráficos
from flask import Flask, request,jsonify, send_from_directory # Essa biblioteca é usada para criar uma API Flask

# Diretório onde está localizado o arquivo app.py da API Flask
api_directory = r"C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\codigos_lucas\controller"

# Comando para iniciar a API Flask
flask_command = ["python", "testeapi.py"]

# Iniciar a API Flask usando subprocess
try:
    process = subprocess.Popen(flask_command, cwd=api_directory)
    time.sleep(5)  # Aguardar 5 segundos para a API iniciar
    print("API Flask iniciada com sucesso!")
except Exception as e:
    print("Erro ao iniciar a API Flask:", e)

# Configurar as credenciais para acessar a API do Reddit
reddit = praw.Reddit(
    client_id="0LUIMHwzq6iTBcF4F4zGpQ",
    client_secret="AN90R4CXtXjCpEfXEEVCIKIjReY0NA",
    user_agent="aps",
)

# Defina sua chave da API do OpenAI
openai.api_key = '---'

# Create a session object
session = requests.Session()
retry = Retry(total=5, backoff_factor=0.1, status_forcelist=[ 500, 502, 503, 504 ])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

def fetch_data(url):
    response = session.get(url)  # Use session.get instead of requests.get
    return response.json()

def mostrar_tabela():
    # Mudar o cursor para um indicador de espera
    root.config(cursor="wait")
    root.update()  # Atualizar a janela para mostrar a mudança no cursor
    
    url = 'http://localhost:5000/reddit'
    for i in range(5):  # Retry 5 times
        try:
            response = requests.get(url)
            break  # If the request is successful, break the loop
        except requests.exceptions.RequestException as e:
            print(f"Request failed with error {e}. Retrying...")
            time.sleep(2**i)  # Exponential backoff
    else:
        print("Failed to fetch the URL after 5 attempts.")

    # Criar uma nova janela
    janela_tabela = tk.Toplevel(root)
    janela_tabela.title("Tabela de Dados")

    # Obter as dimensões da tela
    largura_tela = root.winfo_screenwidth()
    altura_tela = root.winfo_screenheight()

    # Calcular as coordenadas para centralizar a nova janela
    x_pos = (largura_tela - largura_janela) // 2
    y_pos = (altura_tela - altura_janela) // 2

    # Definir a posição da nova janela
    janela_tabela.geometry(f"{largura_janela}x{altura_janela}+{x_pos}+{y_pos}")

    # Maximizar a janela da tabela
    #janela_tabela.state('zoomed')

    # Criar um frame na nova janela para conter a tabela
    frame_tabela_window = tk.Frame(janela_tabela)
    frame_tabela_window.pack()

    # Criar uma árvore de visualização para exibir os dados
    tabela = ttk.Treeview(janela_tabela)
    tabela["columns"] = ("Palavra-Chave", "URL", "Titulo", "Texto", "Data", "Sentimento", "Método", "Cliente IP")
    tabela.heading("#0", text="Índice")
    tabela.heading("Palavra-Chave", text="Palavra-Chave")
    tabela.heading("URL", text="URL")
    tabela.heading("Titulo", text="Titulo")
    tabela.heading("Texto", text="Texto")
    tabela.heading("Data", text="Data")
    tabela.heading("Sentimento", text="Sentimento")
    tabela.heading("Método", text="Método")
    tabela.heading("Cliente IP", text="Cliente IP")

    # Definir a largura das colunas
    tabela.column("#0", width=50)  # Índice
    tabela.column("Palavra-Chave", width=100)
    tabela.column("URL", width=150)
    tabela.column("Titulo", width=100)
    tabela.column("Texto", width=100)
    tabela.column("Data", width=100)
    tabela.column("Sentimento", width=150)
    tabela.column("Método", width=100)
    tabela.column("Cliente IP", width=100)

    # Ler os dados do Reddit (supondo que esta função existe)
    csv_file_path = r'C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\data\reddit1.csv'

    # Ler os dados do arquivo CSV usando pandas
    df = pd.read_csv(csv_file_path)

    # Converter os dados para uma lista de listas
    data = df.values.tolist()

    # Adicionar os dados à tabela
    for index, row in enumerate(data):
        tabela.insert("", "end", text=index, values=(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))

    # Adicionar um botão para fechar a janela da tabela
    botao_fechar_tabela = tk.Button(janela_tabela, text="Fechar Tabela", command=janela_tabela.destroy, borderwidth=3, relief="groove", padx=5, pady=10, bg="#0b4a8a", fg="white", font=("Arial", 12, "bold"), width=25)
    botao_fechar_tabela.pack(side=tk.BOTTOM, pady=0)

    botao_analisar_sentimentos = tk.Button(frame_tabela_window, text="Tabela de Análise de Sentimentos", command=tabelaSentimentos, borderwidth=3, relief="groove", padx=5, pady=10, bg="#0b4a8a", fg="white", font=("Arial", 12, "bold"), width=25)
    botao_analisar_sentimentos.pack(side=tk.BOTTOM, pady=0)

    tabela.pack(expand=True, fill="both")

    # Mudar o cursor de volta para o padrão
    root.config(cursor="")
    root.update()  # Atualizar a janela para mostrar a mudança no cursor

def tabelaSentimentos():
    # Criar uma nova janela
    janelaSentimentos = tk.Toplevel(root)
    janelaSentimentos.title("Tabela de Análise de Sentimentos")

    # Maximizar a janela da tabela
    #janelaSentimentos.state('zoomed')

    # Obter as dimensões da tela
    largura_tela = root.winfo_screenwidth()
    altura_tela = root.winfo_screenheight()

    # Calcular as coordenadas para centralizar a nova janela
    x_pos = (largura_tela - largura_janela) // 2
    y_pos = (altura_tela - altura_janela) // 2

    # Definir a posição da nova janela
    janelaSentimentos.geometry(f"{largura_janela}x{altura_janela}+{x_pos}+{y_pos}")

    # Criar um frame na nova janela para conter a tabela
    frameSentimentos_window = tk.Frame(janelaSentimentos)
    frameSentimentos_window.pack()

    # Criar uma árvore de visualização para exibir os dados
    tabeladeSentimentos = ttk.Treeview(janelaSentimentos)
    tabeladeSentimentos["columns"] = ("Data", "Palavra-chave", "Porcentagem Positiva", "Porcentagem Negativa")
    tabeladeSentimentos.heading("#0", text="Índice")
    tabeladeSentimentos.heading("Data", text="Data")
    tabeladeSentimentos.heading("Palavra-chave", text="Palavra-chave")
    tabeladeSentimentos.heading("Porcentagem Positiva", text="Porcentagem Positiva")
    tabeladeSentimentos.heading("Porcentagem Negativa", text="Porcentagem Negativa")

    # Definir a largura das colunas
    tabeladeSentimentos.column("#0", width=50)  # Índice
    tabeladeSentimentos.column("Data", width=100)
    tabeladeSentimentos.column("Palavra-chave", width=150)
    tabeladeSentimentos.column("Porcentagem Positiva", width=100)
    tabeladeSentimentos.column("Porcentagem Negativa", width=100)

    # Ler os dados do Reddit (supondo que esta função existe)
    csv_file_path = r'C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\data\reddit_sentiment_data.csv'
        
    # Ler os dados do arquivo CSV usando pandas
    df = pd.read_csv(csv_file_path)
        
    # Converter os dados para uma lista de listas
    data = df.values.tolist()

    for index, row in enumerate(data):   
        tabeladeSentimentos.insert("", "end", text=index, values=(row[0], row[1], row[2], row[3]))
    print(f"Dados ausentes na linha {index + 1}. Ignorando esta linha.")

    # Botão mostrar dashboard
    button_mostrar_dashboard = tk.Button(frameSentimentos_window, text="Mostrar DashBoard", command= mostrarDashboard, borderwidth=3, relief="groove", padx=5, pady=10, bg="#0b4a8a", fg="white", font=("Arial", 12, "bold"), width=25)
    button_mostrar_dashboard.pack(side="left", padx=0)

    # Adicionar um botão para fechar a janela da tabela
    botao_fechar_tabela = tk.Button(janelaSentimentos, text="Fechar Tabela", command=janelaSentimentos.destroy, borderwidth=3, relief="groove", padx=5, pady=10, bg="#0b4a8a", fg="white", font=("Arial", 12, "bold"), width=25)
    botao_fechar_tabela.pack(side=tk.BOTTOM, pady=0)

    tabeladeSentimentos.pack(expand=True, fill="both")

def mostrarDashboard():
    # Ler o arquivo CSV
    df = pd.read_csv(r"C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\data\reddit_sentiment_data.csv")
    # Converter a coluna Data para o tipo datetime
    df['Data'] = pd.to_datetime(df['Data'])
    # Transformar a data no formato "dd-mm-yyyy"
    df['Data'] = df['Data'].dt.strftime('%d-%m-%Y')
    
    # Gráfico de Barras
    # Criar subplots para cada dia
    fig, axs = plt.subplots(1, 2, figsize=(18, 8)) 
    # Contador
    A = 0
    for i in df.columns.values[2:]:
        A += 1
        ax = axs[A - 1]  # Seleciona o subplot correspondente
        sns.barplot(data=df.fillna('NaN'), x='Data', y=i, hue='Palavra-chave', palette='viridis', ax=ax)
        ax.set_ylim(0, 121)  # Definir intervalo do eixo y para 0 a 100
        ax.set_title(i, fontsize=15)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=9, color='red')
        if A >= 7:
            ax.tick_params(axis='x', rotation=45)
    # Layout
    plt.tight_layout(h_pad=2)

    # Exibir a figura maximizada
    plt.show()

    # Grafico de linhas
    # Obter todas as palavras-chave únicas
    palavras_chave_unicas = df['Palavra-chave'].unique()
    # Definir o número de linhas e colunas para os subplots
    num_linhas = (len(palavras_chave_unicas) + 1) // 2
    num_colunas = 2

    # Criar subplots
    fig, axes = plt.subplots(num_linhas, num_colunas, figsize=(16, 8), gridspec_kw={'hspace': 0.4})

    # Ajustar o espaçamento entre os subplots manualmente
    plt.subplots_adjust(hspace=0.6, wspace=0.4)

    # Iterar sobre as palavras-chave únicas e criar um gráfico de linha para cada uma
    for idx, palavra_chave in enumerate(palavras_chave_unicas):
        linha = idx // num_colunas
        coluna = idx % num_colunas
        # Filtrar o DataFrame para a palavra-chave específica
        df_palavra_chave = df[df['Palavra-chave'] == palavra_chave]
        # Plotar linha para porcentagem positiva
        sns.lineplot(ax=axes[linha, coluna], data=df_palavra_chave, x='Data', y='Porcentagem Positiva', marker='o', label=f'Positiva - {palavra_chave}')
        # Plotar linha para porcentagem negativa
        sns.lineplot(ax=axes[linha, coluna], data=df_palavra_chave, x='Data', y='Porcentagem Negativa', marker='o', label=f'Negativa - {palavra_chave}')
        axes[linha, coluna].set_title(f'Tendência de Sentimento para "{palavra_chave}"')
        axes[linha, coluna].set_xlabel('Data')
        axes[linha, coluna].set_ylabel('Porcentagem')
        axes[linha, coluna].tick_params(axis='x', labelsize=8)  # Diminuir tamanho da fonte dos rótulos do eixo x e rotacioná-los
        axes[linha, coluna].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d/%m'))  # Formatar a data para dia/mês
        axes[linha, coluna].legend()
        
    # Remover subplots não utilizados
    for idx in range(len(palavras_chave_unicas), num_linhas * num_colunas):
        linha = idx // num_colunas
        coluna = idx % num_colunas
        fig.delaxes(axes[linha, coluna])

    plt.show()

    # Plotagem do Gráfico de dispersão.
    
    # Plot
    plt.figure(figsize=(16, 8))

    # Gráfico de dispersão
    sns.scatterplot(data=df, x='Porcentagem Positiva', y='Porcentagem Negativa', hue='Palavra-chave', palette='viridis')

    # Configurações do eixo y
    plt.ylim(0, 100)

    # Título e rótulos dos eixos
    plt.title('Relação entre Porcentagens Positivas e Negativas')
    plt.xlabel('Porcentagem Positiva')
    plt.ylabel('Porcentagem Negativa')

    plt.legend(title='Palavra-chave')

    plt.show()

    # Gráfico de pizza
    # Converter a coluna Data para o tipo datetime
    df['Data'] = pd.to_datetime(df['Data']).dt.strftime('%d-%m-%Y')

    # Selecionar os dias desejados
    dias_selecionados = ['08-05-2024', '09-05-2024', '10-05-2024', '11-05-2024', '12-05-2024']

    # Criar uma figura para os subplots
    fig, axs = plt.subplots(nrows=1, ncols=len(dias_selecionados), figsize=(16, 8))

    # Iterar sobre os dias selecionados
    for i, data_selecionada in enumerate(dias_selecionados):
        # Filtrar o DataFrame para o dia selecionado
        df_data_selecionada = df[df['Data'] == data_selecionada]

        # Calcular a distribuição das porcentagens positivas e negativas para o dia selecionado
        total_positivo = df_data_selecionada['Porcentagem Positiva'].fillna(0).sum()
        total_negativo = df_data_selecionada['Porcentagem Negativa'].fillna(0).sum()

        # Criar rótulos para os pedaços do gráfico de pizza
        labels = ['Positivo', 'Negativo']
        sizes = [total_positivo, total_negativo]
        colors = ['skyblue', 'lightcoral']
        explode = (0.1, 0)  # Explodir o primeiro pedaço (Positivo)

        # Plotar o gráfico de pizza no subplot correspondente
        axs[i].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        axs[i].set_title(f'Distribuição das Porcentagens para {data_selecionada}', fontsize=10)

    # Ajustar o layout da figura
    plt.tight_layout()

    # Exibir a figura com os subplots
    plt.show()

# Lista de feeds RSS dos sites que serão buscadas as notícias
rss_feeds = {
    "Greenpeace Brasil": "https://www.greenpeace.org/brasil/feed/",
    "G1 - Natureza": "https://g1.globo.com/rss/g1/natureza/",
    "Instituto Akatu": "https://www.akatu.org.br/feed/",
    "O Eco": "https://www.oeco.org.br/feed/",
}

# Lista de palavras relacionadas ao meio ambiente que queremos buscar
environment_related_words = [ # Lista de palavras relacionadas ao meio ambiente
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

# Dicionário para armazenar o conteúdo HTML em cache
html_cache = {}

# Função para obter o conteúdo HTML de um link (com cache)
def get_html_content(url):
    if url in html_cache:
        return html_cache[url]

    try:
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text
        # Adicionar ao cache
        html_cache[url] = html_content
        return html_content
    except requests.RequestException as e:
        print(f"Erro ao obter o conteúdo HTML do URL {url}: {e}")
        return None

# Função para coletar todas as palavras de um texto de forma eficiente
def collect_words(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]
    return words

titles_list = []  # Lista para armazenar os títulos das notícias
# Dicionário para armazenar resumos em cache
summary_cache = {}

def get_summary(url):
    if url in summary_cache:
        return summary_cache[url]

    LANGUAGE = "portuguese"
    SENTENCES_COUNT = 3
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(parser.document, SENTENCES_COUNT)
    summary_text = ' '.join([str(sentence) for sentence in summary])

    # Adicionar ao cache
    summary_cache[url] = summary_text
    return summary_text

# Função para buscar as principais notícias de um feed RSS com resumo (limitando a busca)
def print_top_news_rss_with_summary(site_name, url, max_news=5):
    global titles_list  # Usando a lista global dentro da função
    # Analisa o feed RSS
    feed = feedparser.parse(url)
    # Limita o número de notícias buscadas
    for i, entry in enumerate(feed.entries[:max_news]):
        titles_list.append(entry.title)  # Adiciona o título à lista
        # Obtém o resumo da notícia diretamente (se já tiver sido buscado)
        summary = get_summary(entry.link)  # Você pode otimizar a função get_summary conforme mencionado anteriormente

# Itera sobre a lista de feeds RSS e imprime as principais notícias de cada um com resumo
for site, rss_feed in rss_feeds.items():
    print_top_news_rss_with_summary(site, rss_feed)

# Coleta todas as palavras das notícias
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

# Define as stopwords em português
stop_words = set(stopwords.words('portuguese'))

# Remove stopwords
filtered_words = [word for word in all_words if word not in stop_words]

# Identificar e contar as palavras relacionadas ao meio ambiente
environment_words = [ word for word in filtered_words if word in environment_related_words]
word_freq = nltk.FreqDist(environment_words)

# Obter as palavras mais comuns e suas frequências
top_words, frequencies = zip(*word_freq.most_common(10))

# Convertendo as palavras em formato de dicionário para facilitar o envio para o GPT
word_dict = dict(zip(top_words, frequencies))

# Convertendo a lista de títulos em uma única string
titles_string = '\n'.join(titles_list)

# Imprimir a lista de títulos e suas frequências
prompt_text = ""  # Define a variável prompt_text
prompt_text += "Palavras e suas frequências:\n"
for word, freq in zip(top_words, frequencies):
    prompt_text += f"- {word}: {freq} vezes\n"

# Adicionar os títulos ao prompt_text
prompt_text += "\nTítulos das notícias coletadas:\n\n"
prompt_text += titles_string
prompt_text += "\n\nO que você pode concluir a respeito? Quero uma análise detalhada, ampla, falando das notícias, números e também das implicações, tendências e possíveis ações a serem tomadas em relação ao tema."

#print("Texto para GPT:\n", prompt_text)

# Enviando o texto para o GPT para gerar o relatório
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # Tentar um modelo diferente
    messages=[
        {"role": "user", "content": prompt_text},
    ],
    temperature=0.85,  # Controla a aleatoriedade da saída do modelo
    max_tokens=1200  # Limita o comprimento da saída do modelo
)

summary = "Resposta não disponível."

if response and response.choices and len(response.choices) > 0:
    summary = response.choices[0].message['content'].strip()

# Definindo a função para buscar notícias 
def buscar_noticias_async():
    # Limpa o texto existente no widget de texto
    text_widget.delete(1.0, tk.END)

    def fetch_news(site, rss_feed):
        feed = feedparser.parse(rss_feed)
        news_list = []
        for entry in feed.entries[:3]:  # Buscar apenas as últimas 3 notícias de cada feed
            news_list.append((entry.title, get_summary(entry.link)))
        return news_list

    def update_text_widget(news):
        for site, news_items in news.items():
            text_widget.insert(tk.END, f"Principais notícias de {site}:\n")
            text_widget.insert(tk.END, "=" * 50 + "\n")
            for i, (title, summary) in enumerate(news_items):
                text_widget.insert(tk.END, f"Notícia {i + 1}:\n")
                text_widget.insert(tk.END, f"{title}\n\nResumo:\n")
                text_widget.insert(tk.END, summary + "\n\n")
                text_widget.insert(tk.END, "-" * 50 + "\n")

    def fetch_and_update():
        global titles_list
        titles_list.clear()
        news = {}
        for site, rss_feed in rss_feeds.items():
            news[site] = fetch_news(site, rss_feed)
        update_text_widget(news)
        root.config(cursor="")  # Mudar o cursor de volta para o padrão

    root.config(cursor="wait")  # Mudar o cursor para um indicador de espera
    threading.Thread(target=fetch_and_update).start()

# Defina a função para o botão 3
def exibir_analise():
    global summary  # Certifique-se de ter a variável global summary disponível

    # Limpa o texto existente no widget de texto
    text_widget.delete(1.0, tk.END)

    # Insere o conteúdo da análise no widget de texto
    text_widget.insert(tk.END, "Relatório das notícias relacionadas ao Meio Ambiente:\n\n")
    text_widget.insert(tk.END, summary)

# Função para limpar o frame:
def limpar_tela():
    # Mudar o cursor para um indicador de espera
    root.config(cursor="wait")
    root.update()  # Atualizar a janela para mostrar a mudança no cursor

    # Limpa o texto existente no widget de texto
    text_widget.delete(1.0, tk.END)

    # Mudar o cursor de volta para o padrão
    root.config(cursor="")
    root.update()  # Atualizar a janela para mostrar a mudança no cursor

# Criando as janelas:
# Criando a janela principal
root = tk.Tk()
root.title("Notícias sobre o Meio Ambiente")

# Definindo o tamanho da janela
largura_janela = 1200
altura_janela = 800

# Definindo o tamanho do frame manualmente
largura_frame = 1000
altura_frame = 600

# Calculando a posição x e y da janela para centralizá-la
posicao_x = int((root.winfo_screenwidth() - largura_janela) / 2)
posicao_y = int((root.winfo_screenheight() - altura_janela) / 2)

# Definindo a posição inicial da janela
root.geometry(f"{largura_janela}x{altura_janela}+{posicao_x}+{posicao_y}")

# Criando um frame com tamanho específico
frame_space = tk.Frame(root, width=largura_frame, height=altura_frame, bg="white")
frame_space.pack(padx=20, pady=20, fill="both", expand=True)  # Centralizando e expandindo o frame

# Adicionando um rótulo para exibir o texto dentro do frame
text_widget = tk.Text(frame_space, wrap="word", bg="white")
text_widget.pack(side="left", fill="both", expand=True)

# Adicionando uma barra de rolagem vertical
scrollbar_y = tk.Scrollbar(frame_space, orient="vertical", command=text_widget.yview)
scrollbar_y.pack(side="right", fill="y")
text_widget.config(yscrollcommand=scrollbar_y.set)

# Criando um frame para conter os botões
button_frame = tk.Frame(root)
button_frame.pack(side="bottom", pady=10)  # Posicionando o frame na parte inferior da janela com algum espaço

# Criando o botão 1 com a função definida acima
button1_buscar_noticias = tk.Button(button_frame, text="Principais Notícias Ambientais",command=buscar_noticias_async, borderwidth=3, relief="groove", padx=5, pady=10, bg="#074207", fg="white", font=("Arial", 12, "bold"), width=30)
button1_buscar_noticias.pack(side="left", padx=0)  # Posicionando o botão à esquerda dentro do frame

# Modificando o botão 3 para chamar a função exibir_analise
button3_exibir_analise = tk.Button(button_frame, text="Análise das Notícias Ambientais",command=exibir_analise, borderwidth=3, relief="groove", padx=5, pady=10, bg="#0b4a8a", fg="white", font=("Arial", 12, "bold"), width=30)
button3_exibir_analise.pack(side="left", padx=0)  # Posicionando o botão à esquerda dentro do frame

# Botão mostrar tabela
button_mostrar_tabela = tk.Button(button_frame, text="Tabela e Gráficos de Análise de Sentimento", command=mostrar_tabela, borderwidth=3, relief="groove", padx=5, pady=10, bg="#074207", fg="white", font=("Arial", 12, "bold"), width=37)
button_mostrar_tabela.pack(side="left", padx=0)

# Botão limpar frame
button_limpar_tela = tk.Button(button_frame, text="Limpar Tela", command=limpar_tela, borderwidth=3, relief="groove", padx=5, pady=10, bg="#0b4a8a", fg="white", font=("Arial", 12, "bold"), width=15)
button_limpar_tela.pack(side="left", padx=0)

# Definindo a cor de fundo da janela
root.configure(background="#422407")

# Iniciando o loop principal
root.mainloop()


"""
!Perguntas a respeito do funcionamento do código:
1) O que faz a função fetch_data?
2) Qual a finalidade da função mostrar_tabela?
3) O que é feito pela função tabelaSentimentos?
4) Qual a finalidade da função mostrarDashboard?
5) O que é feito pela função buscar_noticias_async?
6) Qual a finalidade da função exibir_analise?
7) O que é feito pela função limpar_tela?
8) Qual a finalidade da variável api_directory?
9) Qual a finalidade da variável flask_command?
10) O que é feito pela variável reddit?
11) Qual a finalidade da variável openai.api_key?
12) O que é feito pela variável session?
13) Qual a finalidade da variável retry?
14) O que é feito pela variável adapter?
15) Qual a finalidade da variável html_cache?
16) O que é feito pela variável get_html_content?
17) Qual a finalidade da variável collect_words?
18) O que é feito pela variável titles_list?
19) Como funciona a análise de sentimentos?
20) O que é feito pela variável get_summary?
21) Qual a finalidade da variável rss_feeds?
22) O que é feito pela variável environment_related_words?
23) Qual a finalidade da variável environment_words?
24) O que é feito pela variável word_freq?
25) Qual a finalidade da variável top_words?
26) Como funciona o modelo GPT-3.5-turbo?
27) O que é feito pela variável response?
28) Qual a finalidade da variável prompt_text?
29) O código está bem documentado?
30) O código tem em sua estrutura funções bem definidas?
31) O código possui variáveis com nomes significativos?
32) A interface é composta por botões que realizam ações específicas?
33) O código possui tratamento de exceções?
34) O código possui comentários explicativos?
35) O código possui uma estrutura organizada e legível?
36) A interface tem quantos botões e quais são suas funções?
37) Qual a importância da inteligência artificial na análise de sentimentos?
38) Como a análise de sentimentos pode ser aplicada em diferentes áreas?
39) Quais são as vantagens e desvantagens da análise de sentimentos?
40) Como a análise de sentimentos pode ser utilizada para tomada de decisões?
41) Quais são as principais técnicas utilizadas na análise de sentimentos?
42) Como a análise de sentimentos pode ser aplicada em redes sociais e mídias digitais?
43) Quais são os desafios da análise de sentimentos em textos em português?
44) Como a análise de sentimentos pode ser utilizada para monitorar a reputação de uma marca?
45) Quais são as limitações da análise de sentimentos baseada em texto?
46) Os sites de notícias podem influenciar a análise de sentimentos?
47) Como a análise de sentimentos pode ser utilizada para prever tendências de mercado?
48) Quais são as aplicações práticas da análise de sentimentos em empresas e organizações?
49) Os sites empregados no código são confiáveis para a análise de sentimentos?
50) Os dados das notícias coletadas são suficientes para uma análise de sentimentos precisa?
51) Sobre a estrutura do código, o que poderia ser melhorado para facilitar a manutenção?
52) A criação dos gráficos e visualizações poderia ser otimizada para melhorar a apresentação dos dados?
53) Sobre os gráficos e visualizações, quais informações são mais relevantes para a análise de sentimentos?
54) Como os gráficos são gerados e quais as suas importâncias para a análise de sentimentos?
55) Como é feita a integração entre a interface gráfica e as funções do código?
56) Quais são as bibliotecas utilizadas no código?
57) Como é feita a autenticação na API do OpenAI?
58) O código possui algum tipo de armazenamento persistente dos dados coletados?
59) Como é feita a exibição dos gráficos na interface gráfica?
60) O código possui alguma forma de controle de acesso às funcionalidades da interface?
61) Quais são as principais dependências do projeto?
62) Como é feita a coleta das notícias dos feeds RSS?
63) O código possui algum tipo de tratamento para evitar requisições excessivas aos sites de notícias?
64) Como é feita a análise de sentimentos dos textos coletados?
65) O código possui alguma forma de filtrar as notícias coletadas por relevância?
66) Como é feita a exibição dos dados da análise de sentimentos na tabela e nos gráficos?
67) O código possui alguma forma de exportar os dados coletados e analisados?
68) Como é feito o tratamento de erros durante a execução do código?
69) O código possui alguma forma de atualização automática dos dados coletados?
70) Como é feita a comunicação entre o código e o servidor Flask?
71) O código possui alguma forma de lidar com atrasos na resposta das requisições?
72) Como é feita a exibição dos dados de análise de sentimentos por meio de gráficos?
73) O código possui alguma forma de lidar com a falta de conexão com a internet?
74) Como é feita a exibição dos dados de análise de sentimentos por meio de tabelas?
75) O código possui alguma forma de lidar com a falta de resposta da API do OpenAI?

*Respostas:
1) A função fetch_data é responsável por fazer uma requisição GET para uma URL e retornar o conteúdo JSON da resposta.
2) A função mostrar_tabela exibe uma tabela com os dados obtidos de uma API REST.
3) A função tabelaSentimentos exibe uma tabela com os dados de análise de sentimentos obtidos de um arquivo CSV.
4) A função mostrarDashboard exibe gráficos e visualizações dos dados de análise de sentimentos.
5) A função buscar_noticias_async busca as principais notícias de feeds RSS de forma assíncrona.
6) A função exibir_analise exibe uma análise detalhada das notícias relacionadas ao meio ambiente.
7) A função limpar_tela limpa o conteúdo do widget de texto na interface gráfica.
8) A variável api_directory armazena o diretório da API REST.
9) A variável flask_command armazena o comando para iniciar o servidor Flask.
10) A variável reddit armazena os dados obtidos do Reddit.
11) A variável openai.api_key armazena a chave da API do OpenAI.
12) A variável session é um objeto de sessão para fazer requisições HTTP.
13) A variável retry é um objeto de tentativa de requisição HTTP.
14) A variável adapter é um adaptador para lidar com tentativas de requisição HTTP.
15) A variável html_cache é um dicionário para armazenar o conteúdo HTML em cache.
16) A variável get_html_content obtém o conteúdo HTML de uma URL com cache.
17) A variável collect_words coleta todas as palavras de um texto de forma eficiente.
18) A variável titles_list armazena os títulos das notícias.
19) A análise de sentimentos é um processo de identificar e classificar as emoções expressas em um texto.
20) A função get_summary obtém o resumo de uma notícia a partir de um link.
21) A variável rss_feeds armazena os feeds RSS dos sites de notícias.
22) A variável environment_related_words armazena uma lista de palavras relacionadas ao meio ambiente.
23) A variável environment_words armazena as palavras relacionadas ao meio ambiente encontradas nas notícias.
24) A variável word_freq armazena a frequência das palavras relacionadas ao meio ambiente.
25) A variável top_words armazena as palavras mais comuns e suas frequências.
26) O modelo GPT-3.5-turbo é um modelo de linguagem de inteligência artificial desenvolvido pela OpenAI.
27) A variável response armazena a resposta do modelo GPT-3.5-turbo.
28) A variável prompt_text armazena o texto enviado para o modelo GPT-3.5-turbo.
29) Sim, o código está bem documentado com comentários explicativos.
30) Sim, o código possui funções bem definidas e organizadas.
31) Sim, o código possui variáveis com nomes significativos que indicam sua finalidade.
32) Sim, a interface é composta por botões que realizam ações específicas, como buscar notícias e exibir análises.
33) Sim, o código possui tratamento de exceções para lidar com erros durante a execução.
34) Sim, o código possui comentários explicativos que descrevem o que cada parte do código faz.
35) Sim, o código possui uma estrutura organizada e legível, facilitando a compreensão e manutenção.
36) A interface possui 4 botões: buscar_noticias_async, exibir_analise, mostrar_tabela e limpar_tela.
37) A inteligência artificial é importante na análise de sentimentos por sua capacidade de processar grandes volumes de dados de forma eficiente.
38) A análise de sentimentos pode ser aplicada em áreas como marketing, atendimento ao cliente, pesquisa de mercado e análise de feedback.
39) As vantagens da análise de sentimentos incluem a identificação de tendências, feedback em tempo real e insights para tomada de decisões. As desvantagens incluem a necessidade de treinamento e a precisão dos resultados.
40) A análise de sentimentos pode ser utilizada para tomar decisões com base na opinião e sentimentos dos clientes, identificando áreas de melhoria e oportunidades de negócio.
41) As principais técnicas utilizadas na análise de sentimentos incluem análise de texto, processamento de linguagem natural, aprendizado de máquina e mineração de dados.
42) A análise de sentimentos pode ser aplicada em redes sociais e mídias digitais para monitorar a opinião pública, identificar tendências e avaliar a reputação de uma marca.
43) Os desafios da análise de sentimentos em textos em português incluem a variação linguística, gírias, sarcasmo e dupla negação, que podem dificultar a interpretação correta.
44) A análise de sentimentos pode ser utilizada para monitorar a reputação de uma marca, identificar problemas de atendimento ao cliente e avaliar a satisfação dos clientes.
45) As limitações da análise de sentimentos baseada em texto incluem a ambiguidade, a subjetividade e a falta de contexto, que podem afetar a precisão dos resultados.
46) Os sites de notícias podem influenciar a análise de sentimentos ao fornecer informações que refletem a opinião e sentimentos das pessoas sobre determinados assuntos.
47) A análise de sentimentos pode ser utilizada para prever tendências de mercado, identificar oportunidades de negócio e antecipar mudanças no comportamento do consumidor.
48) As aplicações práticas da análise de sentimentos em empresas e organizações incluem a melhoria do atendimento ao cliente, a identificação de problemas e a avaliação da satisfação dos colaboradores.
49) Os sites empregados no código são confiáveis para a análise de sentimentos, pois fornecem notícias relevantes e atualizadas sobre o meio ambiente.
50) Os dados das notícias coletadas são suficientes para uma análise de sentimentos precisa, pois abordam temas relacionados ao meio ambiente e sustentabilidade.
51) Para facilitar a manutenção, o código poderia ser dividido em funções mais específicas e documentado de forma mais detalhada.
52) A criação dos gráficos e visualizações poderia ser otimizada para melhorar a apresentação dos dados e facilitar a interpretação das informações.
53) As informações mais relevantes para a análise de sentimentos nos gráficos são as porcentagens de sentimentos positivos e negativos ao longo do tempo.
54) Os gráficos são gerados a partir dos dados de análise de sentimentos e ajudam a visualizar as tendências e padrões nos sentimentos expressos nos textos analisados.
55) A comunicação entre a interface gráfica e as funções do código é feita por meio de eventos de botões que acionam as ações desejadas.
56) As bibliotecas utilizadas no código incluem pandas, matplotlib, seaborn, nltk, feedparser, requests, BeautifulSoup e openai.
57) A autenticação na API do OpenAI é feita por meio da chave de API fornecida pela plataforma.
58) O código não possui armazenamento persistente dos dados coletados, pois as notícias são buscadas em tempo real.
59) A exibição dos gráficos na interface gráfica é feita por meio da biblioteca matplotlib, que gera os gráficos a partir dos dados de análise de sentimentos.
60) O código não possui controle de acesso às funcionalidades da interface, pois é um projeto de demonstração e não requer autenticação.
61) As principais dependências do projeto são as bibliotecas Python utilizadas para análise de dados, processamento de texto e interação com APIs.
62) A coleta das notícias dos feeds RSS é feita por meio da biblioteca feedparser, que analisa os feeds e extrai as informações das notícias.
63) O código não possui tratamento para evitar requisições excessivas aos sites de notícias, mas poderia ser implementado para evitar sobrecarga nos servidores.
64) A análise de sentimentos dos textos coletados é feita por meio da biblioteca nltk, que processa os textos e identifica as palavras relacionadas ao meio ambiente.
65) O código não possui um filtro de relevância para as notícias coletadas, mas poderia ser implementado para priorizar notícias mais relevantes.
66) A exibição dos dados da análise de sentimentos na tabela e nos gráficos é feita por meio das funções definidas no código.
67) O código não possui uma forma de exportar os dados coletados e analisados, mas poderia ser implementado para gerar relatórios ou arquivos de saída.
68) O tratamento de erros durante a execução do código é feito por meio de exceções e mensagens de erro para informar o usuário sobre problemas.
69) A atualização automática dos dados coletados não é implementada no código, mas poderia ser feita por meio de agendamento de tarefas ou eventos.
70) A comunicação entre o código e o servidor Flask é feita por meio de requisições HTTP para buscar e exibir os dados na interface gráfica.
71) O código não possui tratamento para atrasos na resposta das requisições, mas poderia ser implementado para melhorar a experiência do usuário.
72) A exibição dos dados de análise de sentimentos por meio de gráficos é feita para visualizar as tendências e padrões nos sentimentos expressos nos textos.
73) O código não possui uma forma de lidar com a falta de conexão com a internet, mas poderia ser implementado para informar o usuário sobre o problema.
74) A exibição dos dados de análise de sentimentos por meio de tabelas é feita para apresentar as informações de forma organizada e detalhada.
75) O código não possui uma forma de lidar com a falta de resposta da API do OpenAI, mas poderia ser implementado para lidar com problemas de conexão.


!Perguntas sobre o código e o funcionamento da API implementada:
1) Como é feita a conexão com a API REST?
2) Quais são os endpoints disponíveis na API REST?
3) Como é feita a autenticação na API REST?
4) Quais são os parâmetros necessários para fazer uma requisição na API REST?
5) Como é feito o tratamento de erros nas requisições à API REST?
6) Quais são as principais funcionalidades da API REST?
7) Como é feita a comunicação entre o código e a API REST?
8) Quais são as bibliotecas utilizadas para realizar as requisições à API REST?
9) Como é feito o processamento dos dados retornados pela API REST?
10) Quais são as estruturas de dados utilizadas para armazenar os dados retornados pela API REST?
11) Como é feita a exibição dos dados retornados pela API REST na interface gráfica?
12) Quais são os métodos HTTP utilizados para interagir com a API REST?
13) Como é feito o tratamento de autenticação inválida na API REST?
14) Quais são os possíveis códigos de status retornados pela API REST e como são tratados no código?
15) Como é feita a paginação dos resultados retornados pela API REST?
16) Quais são os limites de requisições na API REST?
17) Como é feito o controle de acesso aos endpoints da API REST?

*Respostas:
1) A conexão com a API REST é feita por meio de requisições HTTP utilizando a biblioteca requests do Python.
2) Os endpoints disponíveis na API REST incluem os métodos GET, POST, PUT e DELETE para interagir com os recursos da API.
3) A autenticação na API REST é feita por meio de uma chave de API fornecida pelo serviço da API.
4) Os parâmetros necessários para fazer uma requisição na API REST incluem a URL do endpoint, os dados da requisição e a chave de autenticação.
5) O tratamento de erros nas requisições à API REST é feito por meio de exceções e mensagens de erro para informar o usuário sobre problemas.
6) As principais funcionalidades da API REST incluem buscar notícias, analisar sentimentos, exibir gráficos e visualizações, e gerar relatórios.
7) A comunicação entre o código e a API REST é feita por meio de requisições HTTP para enviar e receber dados.
8) As bibliotecas utilizadas para realizar as requisições à API REST incluem requests, json e pandas.
9) O processamento dos dados retornados pela API REST é feito por meio da biblioteca json para converter os dados em formato JSON.
10) As estruturas de dados utilizadas para armazenar os dados retornados pela API REST incluem dicionários, listas e dataframes.
11) A exibição dos dados retornados pela API REST na interface gráfica é feita por meio de gráficos, tabelas e visualizações interativas.
12) Os métodos HTTP utilizados para interagir com a API REST incluem GET para buscar dados, POST para enviar dados, PUT para atualizar dados e DELETE para excluir dados.
13) O tratamento de autenticação inválida na API REST é feito por meio de mensagens de erro e validação da chave de autenticação.
14) Os possíveis códigos de status retornados pela API REST incluem 200 para sucesso, 400 para erro de requisição, 401 para autenticação inválida e 500 para erro interno do servidor.
15) A paginação dos resultados retornados pela API REST é feita por meio de parâmetros de consulta para controlar o número de resultados por página.
16) Os limites de requisições na API REST são definidos pelo serviço da API e podem incluir limites de taxa, limites de volume e limites de acesso.
17) O controle de acesso aos endpoints da API REST é feito por meio de autenticação, autorização e validação dos dados enviados nas requisições.
"""