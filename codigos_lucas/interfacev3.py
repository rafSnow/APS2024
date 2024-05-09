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
import os # Essa biblioteca é usada para interagir com o sistema operacional
import re # Essa biblioteca é usada para expressões regulares
import unicodedata # Essa biblioteca é usada para normalizar caracteres Unicode
import torch # Essa biblioteca é usada para criar modelos de aprendizado de máquina

from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer # Essa biblioteca é usada para trabalhar com modelos de linguagem
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

# Baixar as stopwords em português, se ainda não tiver sido feito
nltk.download('stopwords')

# Configurar as credenciais para acessar a API do Reddit
reddit = praw.Reddit(
    client_id="0LUIMHwzq6iTBcF4F4zGpQ",
    client_secret="AN90R4CXtXjCpEfXEEVCIKIjReY0NA",
    user_agent="aps",
)

# Carregar o modelo e o tokenizer pré-treinado
model_name = 'neuralmind/bert-base-portuguese-cased'
model_torch = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model_torch.load_state_dict(torch.load(r'C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\codigos\jupyter\model.bin'))

tokenizer_raf = AutoTokenizer.from_pretrained(r'C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\codigos\jupyter\tokenizer_dir')

# Defina sua chave da API do OpenAI
openai.api_key = '---'

# Função para limpar os dados
def Limpeza_dados(texto):
    # Remover links
    texto = re.sub(r'http\S+', '', texto)
    # Remover menções a usuários
    texto = re.sub(r'@\w+', '', texto)
    # Remover caracteres especiais e números
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    # Remover espaços extras
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

# Função para remover acentos
def remover_acentos(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

# Função para classificar o sentimento do texto
def classify_sentiment(text, tokenizer_raf, model_torch):
    # Limpeza de dados
    text = Limpeza_dados(text)
    text = remover_acentos(text)

    # Tokenização
    inputs = tokenizer_raf(text, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Passagem para a frente
    model_torch.eval()
    with torch.no_grad():
        outputs = model_torch(**inputs)

    # Obter a previsão do modelo
    _, preds = torch.max(outputs.logits, dim=1)

    # Converter a previsão em um sentimento
    if preds.item() == 1:
        return "Positivo"
    else:
        return "Negativo"

# Função para obter dados do Reddit
def obter_dados():
    # Lista de tópicos de interesse (palavras-chave)
    topics = ["desmatamento", "incendios", "barragem", "chuva", "poluição"]
    
    # Lista para armazenar os dados
    data = []
    dataSentiment = [] 
    
    # Adicionar os dados à lista
    for keyword in topics:
        # Contadores para postagens positivas e negativas
        positive_count = 0
        negative_count = 0

        for submission in reddit.subreddit("all").search(keyword, sort="hot", time_filter="month", limit=20):
            if submission.selftext.strip() != "" or submission.url.strip() != "":
                # Obtendo os metadados da mensagem
                url = submission.url
                titulo = submission.title
                texto = submission.selftext
                data_postagem = submission.created_utc

                # Convertendo a data para o formato legível
                data_formatada = datetime.utcfromtimestamp(data_postagem).strftime('%Y-%m-%d %H:%M:%S')

                # Classificar o sentimento do texto
                sentimento = classify_sentiment(titulo + " " + texto, tokenizer_raf, model_torch)

                # Adicionando os metadados à lista de dados
                data.append([keyword, url, titulo, texto, data_formatada, sentimento])
                # Atualizar os contadores com base no sentimento
                if sentimento == "Positivo":
                    positive_count += 1
                elif sentimento == "Negativo":
                    negative_count += 1
        # Calcular a porcentagem de notícias positivas e negativas
        total_count = positive_count + negative_count
        positive_percentage = (positive_count / total_count) * 100 if total_count > 0 else 0
        negative_percentage = (negative_count / total_count) * 100 if total_count > 0 else 0                    
            
        # Obter a data atual
        current_date = datetime.now().strftime("%Y-%m-%d")
    
        # Armazenar os resultados na lista de dados
        if keyword in topics:
            dataSentiment.append([current_date, keyword, positive_percentage, negative_percentage])
    
    # Criar um DataFrame com os dados
    df = pd.DataFrame(dataSentiment, columns=['Data', 'Palavra Chave', 'Porcentagem Positiva', 'Porcentagem Negativa'])

    # Salvar o DataFrame em um arquivo CSV
    csv_file_path = r'C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\data\reddit_sentiment_data.csv'

    # Salvar o DataFrame em um arquivo CSV (modo append)
    df.to_csv(csv_file_path, index=False, mode='a', header=not os.path.isfile(csv_file_path))

    print("Dados salvos com sucesso no arquivo CSV.")
    
    return data

def mostrar_tabela():
    # Mudar o cursor para um indicador de espera
    root.config(cursor="wait")
    root.update()  # Atualizar a janela para mostrar a mudança no cursor
    
    # Criar uma nova janela
    janela_tabela = tk.Toplevel(root)
    janela_tabela.title("Tabela de Dados")

    # Maximizar a janela da tabela
    #janela_tabela.state('zoomed')

    # Criar um frame na nova janela para conter a tabela
    frame_tabela_window = tk.Frame(janela_tabela)
    frame_tabela_window.pack()

    # Cria uma tabela usando o widget Treeview
    tabela = ttk.Treeview(frame_tabela_window)
    tabela["columns"] = ("Palavra-Chave", "URL", "Título", "Texto", "Data", "Sentimento")
    tabela.heading("#0", text="ID")
    tabela.heading("Palavra-Chave", text="Palavra-Chave")
    tabela.heading("URL", text="URL")
    tabela.heading("Título", text="Título")
    tabela.heading("Texto", text="Texto")
    tabela.heading("Data", text="Data")
    tabela.heading("Sentimento", text="Sentimento")

    # Definir a largura das colunas de forma padronizada
    tabela.column("#0", width=50)  # ID
    tabela.column("Palavra-Chave", width=150)
    tabela.column("URL", width=250)
    tabela.column("Título", width=200)
    tabela.column("Texto", width=500)
    tabela.column("Data", width=140)
    tabela.column("Sentimento", width=140)


    # Ler os dados do Reddit (supondo que esta função existe)
    data = obter_dados()

    # Adicionar os dados à tabela
    for index, row in enumerate(data):
        tabela.insert("", "end", text=index, values=(row[0], row[1], row[2], row[3], row[4], row[5]))

    # Adicionar um botão para fechar a janela da tabela
    botao_fechar_tabela = tk.Button(janela_tabela, text="Fechar Tabela", command=janela_tabela.destroy, borderwidth=3, relief="groove", padx=5, pady=10, bg="#0b4a8a", fg="black", font=("Arial", 12, "bold"), width=25)
    botao_fechar_tabela.pack(side=tk.BOTTOM, pady=0)

    # Adicionar um botão para analisar sentimentos
    def analisar_sentimento():
        # Obter o item selecionado na tabela
        selected_item = tabela.selection()[0]
        item_values = tabela.item(selected_item)['values']

        # Obter o texto do título e do texto
        titulo = item_values[2]
        texto = item_values[3]

        # Classificar o sentimento (supondo que esta função existe)
        sentimento = classify_sentiment(titulo + " " + texto, tokenizer_raf, model_torch)

        # Atualizar o valor na tabela
        tabela.item(selected_item, values=(item_values[0], item_values[1], item_values[2], item_values[3], item_values[4], sentimento))

    #Botão para analisar sentimentos
    # Desativar o botão
    botao_analisar_sentimentos = tk.Button(frame_tabela_window, text="Analisar Sentimentos", command=analisar_sentimento, borderwidth=3, relief="groove", padx=5, pady=10, bg="#0b4a8a", fg="black", font=("Arial", 12, "bold"), width=25)
    botao_analisar_sentimentos.pack(side=tk.BOTTOM, pady=0)


    tabela.pack(expand=True, fill="both")

    # Mudar o cursor de volta para o padrão
    root.config(cursor="")
    root.update()  # Atualizar a janela para mostrar a mudança no cursor

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

titles_list = []  # Lista para armazenar os títulos das notícias

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
    LANGUAGE = "portuguese"
    SENTENCES_COUNT = 3

    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    summary = summarizer(parser.document, SENTENCES_COUNT)

    return ' '.join([str(sentence) for sentence in summary])

# Lista de feeds RSS dos sites que serão buscadas as notícias
rss_feeds = {
    "Greenpeace Brasil": "https://www.greenpeace.org/brasil/feed/",
    "G1 - Natureza": "https://g1.globo.com/rss/g1/natureza/",
    "Instituto Akatu": "https://www.akatu.org.br/feed/",
    "O Eco": "https://www.oeco.org.br/feed/",
}


# Itera sobre a lista de feeds RSS e imprime as principais notícias de cada um com resumo
for site, rss_feed in rss_feeds.items():
    # Chama a função para imprimir as notícias com resumo de cada feed
    print_top_news_rss_with_summary(site, rss_feed)
    print("\n")  # Imprime uma linha em branco entre os resultados


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
environment_words = [
    word for word in filtered_words if word in environment_related_words]
word_freq = nltk.FreqDist(environment_words)

# Obter as palavras mais comuns e suas frequências
top_words, frequencies = zip(*word_freq.most_common(10))

# Definir dicionário de polaridades para as palavras
polarities = {}
for word in top_words:
    blob = TextBlob(word)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        polarities[word] = 'Boa'
    elif polarity < 0:
        polarities[word] = 'Ruim'
    else:
        polarities[word] = 'Neutra'

# Contagem das polaridades
polarity_counts = {'Boa': 0, 'Ruim': 0, 'Neutra': 0}
for polarity in polarities.values():
    polarity_counts[polarity] += 1

# Dividir os dados em conjuntos de treinamento e teste
words_array = np.array(top_words)
frequencies_array = np.array(frequencies)
X_train, X_test, y_train, y_test = train_test_split(
    words_array, frequencies_array, test_size=0.2, random_state=42)

# Convert text data to numerical data
tokenizer = KerasTokenizer(num_words=5000)
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
model_tensor = Sequential()
model_tensor.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model_tensor.add(Dense(64, activation='relu'))
model_tensor.add(Dense(1, activation='linear'))
model_tensor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model_tensor.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

loss, mae = model_tensor.evaluate(X_test, y_test)
print(f'Loss: {loss}, MAE: {mae}')

predictions = model_tensor.predict(X_test)

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

# print("Texto para GPT:\n", prompt_text)

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

# print("\n\nRelatório das notícias relacionadas ao Meio Ambiente:")
# print('\n', summary)

# # Função para fechar a janela ao clicar no botão "Fechar"
# def close_window():
#     root.destroy()

# # Criar uma janela principal
# root = tk.Tk()
# root.title("Relatório das Notícias")

# # Criar um widget de texto rolável para exibir o relatório
# report_text = scrolledtext.ScrolledText(root, width=80, height=30)
# report_text.pack()

# # Definir o texto do relatório
# report_text.insert(tk.END, f"\n\nRelatório das notícias relacionadas ao Meio Ambiente:\n\n{summary}\n\n")

# # Criar um botão para fechar a janela
# close_button = tk.Button(root, text="Fechar", command=close_window)
# close_button.pack()

# # Iniciar o loop principal da interface gráfica
# root.mainloop()

# Imprimir as polaridades das palavras
# print("\nPolaridades das Palavras:")
# for word, polarity in polarities.items():
#     print(f"{word}: {polarity}")

# Gráfico de barras das palavras mais repetidas
plt.figure(figsize=(10, 6))
plt.bar(top_words, frequencies)
plt.xlabel('Palavras')
plt.ylabel('Frequência')
plt.title('Palavras Mais Repetidas Relacionadas ao Meio Ambiente')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Gráfico de barras horizontais da análise de sentimento das palavras
plt.figure(figsize=(8, 6))
plt.barh(list(polarity_counts.keys()), list(polarity_counts.values()), color=['green', 'red', 'gray'])
plt.xlabel('Contagem')
plt.ylabel('Sentimento')
plt.title('Distribuição da Análise de Sentimento das Palavras')
plt.grid(axis='x')
plt.show()

# Criando as janelas:

# Definindo a função para buscar notícias em uma thread separada
def buscar_noticias():
    global titles_list  # Utilizando a lista global

    # Limpa o texto existente no widget de texto
    text_widget.delete(1.0, tk.END)

    # Itera sobre os sites e imprime as notícias com resumo
    for site, rss_feed in rss_feeds.items():
        text_widget.insert(tk.END, f"Principais notícias de {site}:\n")
        text_widget.insert(tk.END, "=" * 50 + "\n")
        
        feed = feedparser.parse(rss_feed)

        for i, entry in enumerate(feed.entries[:5]):  # Imprime apenas as 5 principais notícias
            text_widget.insert(tk.END, f"Notícia {i + 1}:\n")
            text_widget.insert(tk.END, f"{entry.title}\n\nResumo:\n")
            summary = get_summary(entry.link)
            text_widget.insert(tk.END, summary + "\n\n")
            titles_list.append(entry.title)  # Adiciona o título à lista

            text_widget.insert(tk.END, "-" * 50 + "\n")

# Função para plotar o gráfico em uma nova janela
def plotar_grafico():
    # Criando uma nova janela para o gráfico
    janela_grafico = tk.Toplevel(root)
    janela_grafico.title("Gráfico de Barras")

    # Criando uma figura para o gráfico
    fig, ax = plt.subplots(figsize=(11, 8))

    # Plotando o gráfico de barras
    ax.bar(top_words, frequencies)
    ax.set_xlabel('Palavras')
    ax.set_ylabel('Frequência')
    ax.set_title('Palavras Mais Repetidas Relacionadas ao Meio Ambiente')
    ax.tick_params(axis='x', rotation=45)  # Ajustando a rotação dos ticks no eixo x

    # Ajustando a posição do subplot e o layout da legenda
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Ajuste a proporção inferior para acomodar a legenda

    # Adicionando o gráfico à nova janela usando FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=janela_grafico)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Defina a função para o botão 3
def exibir_analise():
    global summary  # Certifique-se de ter a variável global summary disponível

    # Limpa o texto existente no widget de texto
    text_widget.delete(1.0, tk.END)

    # Insere o conteúdo da análise no widget de texto
    text_widget.insert(tk.END, "Relatório das notícias relacionadas ao Meio Ambiente:\n\n")
    text_widget.insert(tk.END, summary)

# Criando a janela principal
root = tk.Tk()
root.title("Notícias sobre o Meio Ambiente")

# Obtendo as dimensões da tela
largura_tela = root.winfo_screenwidth()
altura_tela = root.winfo_screenheight()

# Definindo o tamanho da janela
largura_janela = 1000
altura_janela = 600

# Calculando a posição x e y da janela para centralizá-la
posicao_x = int((largura_tela - largura_janela) / 2)
posicao_y = int((altura_tela - altura_janela) / 2)

# Definindo a posição inicial da janela
root.geometry(f"{largura_janela}x{altura_janela}+{posicao_x}+{posicao_y}")

# Exibir a janela e atualizá-la para obter suas dimensões
root.update()

# Obtendo as dimensões da janela principal
window_width = root.winfo_width()
window_height = root.winfo_height()

# Obtendo as dimensões do frame
frame_width = window_width // 1.6  # Define a largura do frame como 1000 pixels
frame_height = 400 # Define a altura do frame como 500 pixels

# Calculando as coordenadas x e y para centralizar o frame horizontalmente e 20 pixels a partir da borda superior
x_frame = (window_width - frame_width) // 2
y_frame = 20

# Criando um frame para o espaço em branco
frame_space = tk.Frame(root, width=frame_width, height=frame_height, bg="white")
frame_space.place(x=x_frame, y=y_frame)  # Definindo a posição do frame

# Adicionando um rótulo para exibir o texto dentro do frame
text_widget = tk.Text(frame_space, wrap="word", bg="white")
text_widget.pack(side="left", fill="y")

# Adicionando uma barra de rolagem vertical
scrollbar_y = tk.Scrollbar(frame_space, orient="vertical", command=text_widget.yview)
scrollbar_y.pack(side="right", fill="y")
text_widget.config(yscrollcommand=scrollbar_y.set)

# Criando um frame para conter os botões
button_frame = tk.Frame(root)
button_frame.pack(side="bottom", pady=10)  # Posicionando o frame na parte inferior da janela com algum espaço

# Criando o botão 1 com a função definida acima
button1_buscar_noticias = tk.Button(button_frame, text="Principais Notícias", command=buscar_noticias, borderwidth=3, relief="groove", padx=5, pady=10, bg="#074207", fg="black", font=("Arial", 12, "bold"), width=25)
button1_buscar_noticias.pack(side="left", padx=0)  # Posicionando o botão à esquerda dentro do frame

# Modificando o botão 2 para chamar a função plotar_grafico
button2_plotar_grafico = tk.Button(button_frame, text="Gráfico de Notícias", command=plotar_grafico, borderwidth=3, relief="groove", padx=5, pady=10, bg="#cc990e", fg="black", font=("Arial", 12, "bold"), width=25)
button2_plotar_grafico.pack(side="left", padx=0)  # Posicionando o botão à esquerda dentro do frame

# Modificando o botão 3 para chamar a função exibir_analise
button3_exibir_analise = tk.Button(button_frame, text="Análise Escrita das Notícias", command=exibir_analise, borderwidth=3, relief="groove", padx=5, pady=10, bg="#0b4a8a", fg="black", font=("Arial", 12, "bold"), width=25)
button3_exibir_analise.pack(side="left", padx=0)  # Posicionando o botão à esquerda dentro do frame

# Botão mostrar tabela
button_mostrar_tabela = tk.Button(button_frame, text="Mostrar Tabela", command=mostrar_tabela, borderwidth=3, relief="groove", padx=5, pady=10, bg="#0b4a8a", fg="black", font=("Arial", 12, "bold"), width=25)
button_mostrar_tabela.pack(side="left", padx=0)


# Definindo a cor de fundo da janela
root.configure(background="#422407")

# Iniciando o loop principal
root.mainloop()

"""
# O código acima cria uma interface gráfica simples com três botões que permitem buscar as principais notícias, plotar um gráfico de barras das palavras mais repetidas e exibir uma análise escrita das notícias. A análise escrita é gerada pelo GPT-3 com base nos títulos das notícias coletadas.
# Ele também exibe as principais notícias de diferentes sites relacionados ao meio ambiente, extrai palavras-chave, analisa o sentimento das palavras e gera um relatório detalhado das notícias.
# A interface gráfica é criada usando a biblioteca Tkinter, e os gráficos são plotados usando a biblioteca Matplotlib.
# O código é executado em uma única thread, mas a busca de notícias é feita em uma thread separada para evitar bloqueios na interface gráfica.
# A análise escrita das notícias é gerada pelo GPT-3 usando a API da OpenAI.
# O código pode ser executado em qualquer ambiente Python com as bibliotecas necessárias instaladas.
# Uma curiosidade sobre o código é que ele utiliza a análise de sentimentos para identificar se as palavras relacionadas ao meio ambiente têm uma conotação positiva, negativa ou neutra.
# E pode ser usado para acompanhar as notícias sobre o meio ambiente e gerar relatórios automatizados com base nessas notícias.
"""