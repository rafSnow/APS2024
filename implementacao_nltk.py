from bs4 import BeautifulSoup
import urllib.request
import nltk
from nltk.corpus import stopwords
from rich.traceback import install
install()
nltk.download('stopwords')

# #!código corrigido para print:
# tagged_sents = nltk.corpus.mac_morpho.tagged_sents() #carregaria as sentenças rotuladas do corpus mac_morpho fornecido pelo NLTK. 
# #Este corpus contém textos em português do Brasil já marcados com partes do discurso (POS tags).
# texto = 'A manhã está ensolarada' #define uma string de exemplo contendo a frase "A manhã está ensolarada".
# tokens = nltk.word_tokenize(texto) #usa a função word_tokenize() do NLTK para tokenizar a frase de exemplo em uma lista de tokens (palavras). 
# #Isso é necessário para preparar o texto para a marcação de POS.
# unigram_tagger = nltk.tag.UnigramTagger(tagged_sents) #Ela cria um etiquetador UnigramTagger usando as sentenças rotuladas do corpus mac_morpho 
# #como dados de treinamento. O etiquetador UnigramTagger é um modelo de marcação de POS simples e baseado em estatísticas que atribui tags de POS 
# #a cada palavra com base em observações anteriores.
# tags = unigram_tagger.tag(tokens) #usa o etiquetador UnigramTagger para marcar cada token na frase de exemplo com sua respectiva parte 
# #do discurso (POS tag).

# print("Palavra - Tag:")
# for palavra, tag in tags:
#     print(f'{palavra} - {tag}')
"""
Importará a biblioteca nltk.
Comentará o código que carrega os dados rotulados do corpus mac_morpho, pois não está sendo usado no momento.
Definirá uma string de exemplo texto contendo a frase "A manhã está ensolarada".
Tokenizará a frase usando nltk.word_tokenize() para obter uma lista de tokens (palavras).
Criará um etiquetador UnigramTagger chamado unigram_tagger.
Usará o etiquetador para marcar cada token na frase com sua respectiva parte do discurso (POS tag).
Imprimirá as palavras da frase junto com suas respectivas tags de POS.
"""

# #!código base:
# tagged_sents = nltk.corpus.mac_morpho.tagged_sents()
# texto = 'A manhã está ensolarada'
# tokens = nltk.word_tokenize(texto)
# unigram_tagger = nltk.tag.UnigramTagger(tagged_sents)
# unigram_tagger.tag(tokens)
"""
Importará a biblioteca nltk.
Comentará o código que carrega os dados rotulados do corpus mac_morpho, pois não está sendo usado no momento.
Definirá uma string de exemplo texto contendo a frase "A manhã está ensolarada".
Tokenizará a frase usando nltk.word_tokenize() para obter uma lista de tokens (palavras).
Criará um etiquetador UnigramTagger chamado unigram_tagger.
Usará o etiquetador para marcar cada token na frase com sua respectiva parte do discurso (POS tag).
"""

#!código base:
response = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')
html = response.read()
soup = BeautifulSoup(html,"html5lib")
text = soup.get_text(strip=True)
tokens = [t for t in text.split()]
freq = nltk.FreqDist(tokens)

for key,val in freq.items():
    if val > 10:
        print (str(key) + ':' + str(val))
freq.plot(20, cumulative=False)
"""
Usa o módulo urllib.request para abrir uma conexão com o URL 'https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial'.
Lê o conteúdo HTML da página web.
Usa a biblioteca BeautifulSoup para analisar o HTML.
Extrai todo o texto da página usando soup.get_text(strip=True). Isso remove todas as tags HTML e deixa apenas o texto puro.
Divide o texto em tokens (palavras) usando text.split().
Usa a biblioteca nltk para calcular a frequência de cada token (palavra) no texto.
Itera sobre o dicionário de frequência (freq.items()) e imprime as palavras que ocorrem mais de 10 vezes.
Usa o método plot() da classe FreqDist para criar um gráfico de frequência das 20 palavras mais comuns.
Portanto, o programa basicamente extrai o texto de uma página da Wikipedia sobre inteligência artificial, conta a frequência das palavras 
e imprime as palavras mais comuns, além de plotar um gráfico de frequência das 20 palavras mais comuns.
"""

# #!sem as stopwords:
# response = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')
# #utiliza o módulo urllib.request para abrir uma conexão com a URL 'https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial' e 
# # retorna um objeto de resposta.
# html = response.read()
# # lê o conteúdo da resposta da página web e armazena-o na variável html.
# soup = BeautifulSoup(html, 'html5lib')
# #utiliza a biblioteca BeautifulSoup para criar um objeto BeautifulSoup a partir do conteúdo HTML da página. O argumento 'html5lib' 
# # indica que o BeautifulSoup deve usar o parser 
# # HTML5 para analisar o HTML.
# text = soup.getText(strip = True)
# #extrai todo o texto da página HTML, removendo quaisquer espaços em branco desnecessários ao redor do texto, e armazena-o na variável text.
# tokens = nltk.word_tokenize(text)
# # utiliza a função word_tokenize() do NLTK para dividir o texto em uma lista de tokens (palavras).
# stop_words = set(stopwords.words('portuguese'))
# #define uma lista de stopwords em português, usando o conjunto de stopwords fornecido pelo NLTK para o idioma português.
# filtered_tokens = [t for t in tokens if t.lower() not in stop_words]
# #cria uma lista de tokens filtrados, removendo quaisquer tokens que sejam stopwords em português.
# freq = nltk.FreqDist(filtered_tokens)
# #utiliza a função FreqDist() do NLTK para calcular a frequência de cada token na lista de tokens filtrados e armazena o resultado em um 
# # objeto FreqDist chamado freq.
# for key, val in freq.items():
#     if val > 10:
#         print (str(key) + ':' + str(val))
# # itera sobre os itens do objeto FreqDist e imprime as palavras que ocorrem mais de 10 vezes, juntamente com sua frequência.
# freq.plot(20, cumulative=False)
#cria um gráfico de frequência das 20 palavras mais comuns na lista de tokens filtrados. 
# O parâmetro cumulative=False indica que o gráfico não deve ser cumulativo, ou seja, cada barra representa a 
# frequência de uma palavra individual.
"""
Importamos a biblioteca stopwords do NLTK e baixamos a lista de stopwords em português.
Utilizamos a lista de stopwords para filtrar as palavras do texto, mantendo apenas aquelas que não são stopwords.
Calculamos a frequência das palavras no texto filtrado.
Imprimimos as palavras mais frequentes (com frequência superior a 10) e plotamos um gráfico de frequência das 20 palavras mais 
comuns no texto após a remoção das stopwords.
"""