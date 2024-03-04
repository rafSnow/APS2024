# Importe as bibliotecas necessárias
import tweepy

# Defina suas credenciais
consumer_key = '-'  # Chave de consumo da API do Twitter
consumer_secret = '-'  # Chave secreta de consumo da API do Twitter

# Configure a autenticação com OAuth 2.0
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

# Crie o objeto da API
api = tweepy.API(auth)

# Verifique se a autenticação foi bem-sucedida
if not api:
    print('Erro de autenticação!')
else:
    print('Autenticação bem-sucedida!')

# Agora você pode usar a API conforme necessário
