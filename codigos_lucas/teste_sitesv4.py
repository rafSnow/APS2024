import feedparser  # Para analisar feeds RSS
import requests  # Para fazer solicitações HTTP
import matplotlib.pyplot as plt  # Para plotar gráficos
import nltk  # Para processamento de linguagem natural
import openai  # Para uso da API OpenAI
import tkinter as tk

from sumy.parsers.html import HtmlParser  # Para analisar o HTML de uma página
from sumy.nlp.tokenizers import Tokenizer  # Para tokenização do texto
from sumy.summarizers.lex_rank import LexRankSummarizer  # Para sumarização de texto
from bs4 import BeautifulSoup  # Para analisar HTML
from nltk.corpus import stopwords  # Lista de palavras irrelevantes
from nltk.tokenize import word_tokenize  # Para tokenizar texto
from tkinter import Scrollbar, Text, messagebox

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

# Função para imprimir as principais notícias de um feed RSS com resumo no rótulo
def print_top_news_rss_with_summary_to_label(site_name, url, max_news=5):
    global titles_list  # Usando a lista global dentro da função
    
    # Analisa o feed RSS
    feed = feedparser.parse(url)
    
    # Concatena os resultados em uma string
    result_text = ""
    # Imprime cabeçalho
    result_text += f"Principais notícias de {site_name}:\n"
    result_text += "=" * 50 + "\n"
    
    # Itera pelas entradas do feed (notícias)
    for i, entry in enumerate(feed.entries[:max_news]):
        result_text += f"Notícia {i+1}:\n"
        result_text += entry.title + '.\n'  # Título da notícia
        titles_list.append(entry.title)  # Adiciona o título à lista
        result_text += "\nResumo:\n"
        summary = get_summary(entry.link)  # Obtém o resumo da notícia
        result_text += summary + "\n"  # Adiciona o resumo
        result_text += "-" * 50 + "\n"
    
    # Atualiza o texto do rótulo com os resultados
    text_widget.delete(1.0, tk.END)  # Limpa o texto atual
    text_widget.insert(tk.END, result_text)

def get_summary(url):
    parser = HtmlParser.from_url(url, Tokenizer("portuguese"))  # Parseia o HTML da página
    summarizer = LexRankSummarizer()  # Inicializa o sumarizador
    summary = summarizer(parser.document, sentences_count=3)  # Obtém um resumo do conteúdo
    return ' '.join([str(sentence) for sentence in summary])  # Retorna o resumo como uma string

# Display the window and update it to get its dimensions
root.update()

# Getting the dimensions of the main window
window_width = root.winfo_width()
window_height = root.winfo_height()

# Getting the dimensions of the frame
frame_width = window_width // 1.6  # Set the frame width to half of the window width
frame_height = 400

# Calculating the x and y coordinates to center the frame horizontally and 20 pixels from the top edge
x_frame = (window_width - frame_width) // 2
y_frame = 20

# Creating a frame for the blank space
frame_space = tk.Frame(root, width=frame_width, height=frame_height, bg="white")
frame_space.place(x=x_frame, y=y_frame)  # Setting the position of the frame

# Adding a label to display text within the frame
text_widget = tk.Text(frame_space, wrap="word", bg="white")
text_widget.pack(side="left", fill="y")

# Adding vertical scrollbar
scrollbar_y = tk.Scrollbar(frame_space, orient="vertical", command=text_widget.yview)
scrollbar_y.pack(side="right", fill="y")
text_widget.config(yscrollcommand=scrollbar_y.set)

# Função para imprimir os dados no rótulo
def imprimir_dados_no_rotulo():
    # Itera sobre a lista de feeds RSS e imprime as principais notícias de cada um com resumo
    result_text = ""
    for site, rss_feed in rss_feeds.items():
        result_text += f"Principais notícias de {site}:\n"
        result_text += "=" * 50 + "\n"
        feed = feedparser.parse(rss_feed)
        for i, entry in enumerate(feed.entries[:5]):
            result_text += f"Notícia {i+1}:\n"
            result_text += entry.title + '.\n'  # Título da notícia
            result_text += "\nResumo:\n"
            summary = get_summary(entry.link)  # Obtém o resumo da notícia
            result_text += summary + "\n"  # Adiciona o resumo
            result_text += "-" * 50 + "\n"
    
    # Atualiza o texto do rótulo com os resultados
    text_widget.delete(1.0, tk.END)  # Limpa o texto atual
    text_widget.insert(tk.END, result_text)

# Lista de feeds RSS dos sites que serão buscadas as notícias
rss_feeds = {
    "Greenpeace Brasil": "https://www.greenpeace.org/brasil/feed/",
    "G1 - Natureza": "https://g1.globo.com/rss/g1/natureza/",
    "Instituto Akatu": "https://www.akatu.org.br/feed/",
    "O Eco": "https://www.oeco.org.br/feed/",
}

# Definindo funções para os botões
def funcao_botao1():
    for site, rss_feed in rss_feeds.items():
        print_top_news_rss_with_summary_to_label(site, rss_feed)

def funcao_botao2():
    messagebox.showinfo("Botão 2", "Você pressionou o Botão 2!")

def mostrar_mensagem():
    messagebox.showinfo("Título da Mensagem", "Esta é uma mensagem de exemplo!")

# Criando um frame para conter os botões
button_frame = tk.Frame(root)
button_frame.pack(side="bottom", pady=10)  # Posiciona o frame na parte inferior da janela com algum espaço

# Criando os botões dentro do frame com largura definida
button1_buscar_noticias = tk.Button(button_frame, text="Principais Notícias", command=imprimir_dados_no_rotulo, borderwidth=3, relief="groove", padx=5, pady=10, bg="#074207", fg="black", font=("Arial", 12, "bold"), width=25)
button1_buscar_noticias.pack(side="left", padx=0)  # Posiciona o botão à esquerda dentro do frame

button2_plotar_grafico = tk.Button(button_frame, text="Gráfico de Notícias", command=funcao_botao2, borderwidth=3, relief="groove", padx=5, pady=10, bg="#cc990e", fg="black", font=("Arial", 12, "bold"), width=25)
button2_plotar_grafico.pack(side="left", padx=0)  # Posiciona o botão à esquerda dentro do frame

button3_exibir_analise = tk.Button(button_frame, text="Análise Escrita das Notícias", command=mostrar_mensagem, borderwidth=3, relief="groove", padx=5, pady=10, bg="#0b4a8a", fg="black", font=("Arial", 12, "bold"), width=25)
button3_exibir_analise.pack(side="left", padx=0)  # Posiciona o botão à esquerda dentro do frame

# Definindo a cor de fundo da janela
root.configure(background="#422407")

# Iniciando o loop principal
root.mainloop()
















# # Baixe as stopwords em português, se ainda não tiver sido feito
# nltk.download('stopwords')

# # Defina sua chave da API do OpenAI
# openai.api_key = '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'

# # Função para obter o conteúdo HTML de um link
# def get_html_content(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         return response.text
#     except requests.RequestException as e:
#         print(f"Erro ao obter o conteúdo HTML do URL {url}: {e}")
#         return None

# # Função para coletar todas as palavras de um texto
# def collect_words(text):
#     words = word_tokenize(text)
#     words = [word.lower() for word in words if word.isalnum()]
#     return words

# titles_list = []  # Lista para armazenar os títulos das notícias

# # Função para imprimir as principais notícias de um feed RSS com resumo
# def print_top_news_rss_with_summary(site_name, url, max_news=5):
#     global titles_list  # Usando a lista global dentro da função
    
#     # Analisa o feed RSS
#     feed = feedparser.parse(url)
    
#     # Imprime cabeçalho
#     print(f"Principais notícias de {site_name}:")
#     print("="*50)
    
#     # Itera pelas entradas do feed (notícias)
#     for i, entry in enumerate(feed.entries[:max_news]):
#         print(f"Notícia {i+1}:")
#         print(entry.title,'.')  # Imprime o título da notícia
#         titles_list.append(entry.title)  # Adiciona o título à lista
#         print("\nResumo:")
#         summary = get_summary(entry.link)  # Obtém o resumo da notícia
#         print(summary)  # Imprime o resumo
#         print("-"*50)

# # Função para obter o resumo do conteúdo de um link
# def get_summary(url):
#     parser = HtmlParser.from_url(url, Tokenizer("portuguese"))  # Parseia o HTML da página
#     summarizer = LexRankSummarizer()  # Inicializa o sumarizador
#     summary = summarizer(parser.document, sentences_count=3)  # Obtém um resumo do conteúdo
#     return ' '.join([str(sentence) for sentence in summary])  # Retorna o resumo como uma string

# # Lista de feeds RSS dos sites que serão buscadas as notícias
# rss_feeds = {
#     "Greenpeace Brasil": "https://www.greenpeace.org/brasil/feed/",
#     "G1 - Natureza": "https://g1.globo.com/rss/g1/natureza/",
#     "Instituto Akatu": "https://www.akatu.org.br/feed/",
#     "O Eco": "https://www.oeco.org.br/feed/",
# }

# # Itera sobre a lista de feeds RSS e imprime as principais notícias de cada um com resumo
# for site, rss_feed in rss_feeds.items():
#     print_top_news_rss_with_summary(site, rss_feed)  # Chama a função para imprimir as notícias com resumo de cada feed
#     print("\n")  # Imprime uma linha em branco entre os resultados

# # Lista de palavras relacionadas ao meio ambiente que queremos buscar
# environment_related_words = [
#     "queimadas", "enchente", "poluição", "desmatamento", "biodiversidade", "sustentabilidade",
#     "reciclagem", "desflorestamento", "aquecimento global", "preservação", "conservação",
#     "impacto ambiental", "energia renovável", "carbono", "plástico", "floresta", "água",
#     "ar", "fauna", "flora", "clima", "contaminação", "resíduos", "ecossistema",
#     "despejo ilegal de resíduos", "extração ilegal de madeira", "caça furtiva", "pesca predatória",
#     "tráfico de animais silvestres", "despejo de produtos químicos tóxicos", "vazamento de petróleo",
#     "incêndios criminosos", "corrupção ambiental", "desvio de recursos naturais",
#     "exploração ilegal de recursos naturais", "contrabando de espécies protegidas", "urbanização descontrolada",
#     "construções irregulares em áreas de preservação", "agronegócio predatório", "mineração ilegal",
#     "pescaria ilegal em áreas protegidas", "envenenamento de rios e mananciais", "contaminação de alimentos",
#     "desmatamento para fins comerciais", "uso indiscriminado de agrotóxicos", "sobreexplotação de recursos hídricos",
#     "evasão de fiscalização ambiental", "suborno para autorizações ambientais", "impacto ambiental de indústrias poluidoras",
#     "destruição de habitats naturais", "assoreamento de rios e lagos", "atropelamento de fauna silvestre",
#     "derramamento de lixo em áreas protegidas", "exploração sexual de recursos naturais",
#     "pesquisa científica ilegal em fauna e flora", "furtos de patrimônio natural",
#     "introdução de espécies exóticas invasoras", "grilagem de terras", "contrabando de produtos florestais",
#     "roubo de madeira", "extração ilegal de areia e cascalho", "modificação genética de espécies",
#     "descarte irregular de produtos eletrônicos", "produção clandestina de carvão vegetal",
#     "pesca com explosivos ou veneno",
#     "desmatamento", "queimada", "incêndio", "inundações", "poluente", "resíduo", "erosão", "desertificação",
#     "contaminante", "contaminação", "degradação", "seca", "salinização", "desertificação", "assoreamento",
#     "impacto", "alteração", "degradante", "emissão"
# ]

# # Coleta todas as palavras das notícias
# all_words = []
# for site, rss_feed in rss_feeds.items():
#     feed = feedparser.parse(rss_feed)
#     for entry in feed.entries:
#         html_content = get_html_content(entry.link)
#         if html_content:
#             soup = BeautifulSoup(html_content, 'html.parser')
#             text = soup.get_text()
#             words = collect_words(text)
#             all_words.extend(words)

# # Remove stopwords
# stop_words = set(stopwords.words('portuguese'))
# filtered_words = [word for word in all_words if word not in stop_words]

# # Identifica e conta as palavras relacionadas ao meio ambiente
# environment_words = [word for word in filtered_words if word in environment_related_words]
# word_freq = nltk.FreqDist(environment_words)

# # Obtém as palavras mais comuns e suas frequências
# top_words = word_freq.most_common(10)
# top_words, frequencies = zip(*top_words)

# # Plota o gráfico de barras das palavras mais comuns relacionadas ao meio ambiente
# plt.figure(figsize=(10, 6))
# plt.bar(top_words, frequencies, color='green')
# plt.xlabel('Palavras relacionadas ao Meio Ambiente')
# plt.ylabel('Frequência\nx vezes')
# plt.title('Frequência das palavras relacionadas ao Meio Ambiente nas notícias')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# # Convertendo as palavras em formato de dicionário para facilitar o envio para o GPT
# word_dict = dict(zip(top_words, frequencies))

# # Convertendo a lista de títulos em uma única string
# titles_string = '\n'.join(titles_list)

# # Construa o texto para o prompt do GPT
# prompt_text = "Frequência das palavras nas últimas notícias do meio ambiente.\n"

# for word, freq in word_dict.items():
#     prompt_text += f"- {word}: {freq} vezes\n"

# prompt_text += "\nTítulos das notícias coletadas:\n\n"
# prompt_text += titles_string
# prompt_text += "\n\nO que você pode concluir a respeito? Quero uma análise detalhada, ampla, falando das notícias, números e também das implicações, tendências e possíveis ações a serem tomadas em relação ao tema."

# print("Texto para GPT:\n", prompt_text)

# # Enviando o texto para o GPT para gerar o relatório
# response = openai.ChatCompletion.create(
#     model = "gpt-3.5-turbo",  # Tentar um modelo diferente
#     messages=[
#         {"role": "user", "content": prompt_text},
#     ],
#     temperature=0.85,  # Controla a aleatoriedade da saída do modelo
#     max_tokens=1200  # Limita o comprimento da saída do modelo
# )

# summary = "No output from the model"

# if response and response.choices and len(response.choices) > 0:
#     summary = response.choices[0].message['content'].strip()

# print("\n\nRelatório das notícias relacionadas ao Meio Ambiente:")
# print('\n',summary)



