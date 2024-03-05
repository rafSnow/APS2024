# Importação das bibliotecas necessárias
import feedparser  # Para análise de feeds RSS
import requests  # Para fazer solicitações HTTP
from sumy.parsers.html import HtmlParser  # Para analisar o HTML de uma página
from sumy.nlp.tokenizers import Tokenizer  # Para tokenização do texto
from sumy.summarizers.lex_rank import LexRankSummarizer  # Para sumarização de texto

# Definição da função para imprimir as principais notícias de um feed RSS com resumo
def print_top_news_rss_with_summary(site_name, url, max_news=4):
    # Analisa o feed RSS
    feed = feedparser.parse(url)
    
    # Imprime cabeçalho
    print(f"Principais notícias de {site_name}:")
    print("="*50)
    
    # Itera pelas entradas do feed (notícias)
    for i, entry in enumerate(feed.entries[:max_news]):
        print(f"Notícia {i+1}:")
        print(entry.title,'.')  # Imprime o título da notícia
        print("\nResumo:")
        summary = get_summary(entry.link)  # Obtém o resumo da notícia
        print(summary)  # Imprime o resumo
        print("-"*50)

# Função para obter o resumo do conteúdo de um link
def get_summary(url):
    response = requests.get(url)  # Faz uma solicitação HTTP para obter o conteúdo da página
    parser = HtmlParser.from_url(url, Tokenizer("portuguese"))  # Parseia o HTML da página
    summarizer = LexRankSummarizer()  # Inicializa o sumarizador
    summary = summarizer(parser.document, sentences_count=3)  # Obtém um resumo do conteúdo
    return ' '.join([str(sentence) for sentence in summary])  # Retorna o resumo como uma string

# Lista de feeds RSS dos sites que serão buscadas as notícias
rss_feeds = {
    "Greenpeace Brasil": "https://www.greenpeace.org/brasil/feed/",
    "G1 - Natureza": "https://g1.globo.com/rss/g1/natureza/",
    "Instituto Akatu": "https://www.akatu.org.br/feed/",
    "O Eco": "https://www.oeco.org.br/feed/",
}

# Itera sobre a lista de feeds RSS e imprime as principais notícias de cada um com resumo
for site, rss_feed in rss_feeds.items():
    print_top_news_rss_with_summary(site, rss_feed)  # Chama a função para imprimir as notícias com resumo de cada feed
    print("\n")  # Imprime uma linha em branco entre os resultados
