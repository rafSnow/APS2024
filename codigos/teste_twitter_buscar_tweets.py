# Importando a função install do módulo traceback do Rich para melhorar a visualização dos erros
from rich.traceback import install
install()

# Importando a biblioteca tweepy para interagir com a API do Twitter
import tweepy

# Substitua essas variáveis pelas suas próprias credenciais
consumer_key = '-'  # Chave de consumo da API do Twitter
consumer_secret = '-'  # Chave secreta de consumo da API do Twitter
access_token = '-'  # Token de acesso à conta do Twitter
access_token_secret = '-'  # Token secreto de acesso à conta do Twitter

# Configure a autenticação
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Crie o objeto da API
api = tweepy.API(auth)

# Nome de usuário da conta do Twitter
username = 'MercedesAMGF1BR'

# Obtenha os últimos tweets
try:
    tweets = api.user_timeline(screen_name=username, count=4)
    for tweet in tweets:
        print(tweet.text)
except tweepy.TweepError as e:
    print(f'Erro ao acessar a API do Twitter: {e.response.status_code}')