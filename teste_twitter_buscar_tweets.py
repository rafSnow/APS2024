# Importando a função install do módulo traceback do Rich para melhorar a visualização dos erros
from rich.traceback import install
install()

# Importando a biblioteca tweepy para interagir com a API do Twitter
import tweepy

# Substitua essas variáveis pelas suas próprias credenciais
consumer_key = '---'  # Chave de consumo da API do Twitter
consumer_secret = '---'  # Chave secreta de consumo da API do Twitter
access_token = '---'  # Token de acesso à conta do Twitter
access_token_secret = '---'  # Token secreto de acesso à conta do Twitter

# Autenticando usando as credenciais fornecidas
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Criando um objeto da API tweepy usando a autenticação
api = tweepy.API(auth)

try:
    # Tentando obter os últimos 10 tweets do usuário '@MercedesAMGF1BR'
    tweets = api.user_timeline(screen_name='@MercedesAMGF1BR', count=10)

    # Iterando sobre os tweets e imprimindo seus textos
    for tweet in tweets:
        print(tweet.text)
except tweepy.TweepError as e:
    # Se ocorrer um erro ao buscar os tweets, exibir uma mensagem de erro com o detalhe do erro
    print(f"Erro ao buscar tweets do usuário: {e}")

"""
É necessário criar um novo app na página de developer do twitter e a partir dos códigos do novo app, essa aplicação talvez possa rodar.
(Talvez) o app para essa aplicação de coletar os textos das postagens tenha que ser diferente do app que faz posts no perfil do twitter (teste_twitter.py)
Hoje, 02/03, cheguei ao limite de criação de app (3) por dia por usuário.
"""