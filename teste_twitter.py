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

# Criando um cliente tweepy.Client com as credenciais fornecidas
client = tweepy.Client(
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_token_secret=access_token_secret
)

try:
    # Tentando criar um tweet com o texto especificado
    client.create_tweet(text="Teste APS 2024")
    print("Tweet enviado com sucesso!")  # Exibindo uma mensagem de sucesso se o tweet for enviado com sucesso
except tweepy.TweepError as e:
    # Se ocorrer um erro ao enviar o tweet, exibir uma mensagem de erro com o detalhe do erro
    print(f"Erro ao enviar o tweet: {e}")

"""
Esse código funcionou, mas para isso, tive que criar um app na página de developer do twitter, e pegar as chaves de acesso à api para conseguir executar os comandos.
Esse aplicativo por sua vez, dava a permissão para gerenciar posts (criar e apagar) do meu perfil do twitter.
"""