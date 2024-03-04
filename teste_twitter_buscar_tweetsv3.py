# Importando a função install do módulo traceback do Rich para melhorar a visualização dos erros
# from rich.traceback import install
# install()
import requests

# Substitua essas variáveis pelas suas próprias credenciais
consumer_key = '-'  # Chave de consumo da API do Twitter
consumer_secret = '-'  # Chave secreta de consumo da API do Twitter
access_token = '-'  # Token de acesso à conta do Twitter
access_token_secret = '-'  # Token secreto de acesso à conta do Twitter

# Configure as credenciais de autenticação
auth = (consumer_key, consumer_secret)

# Nome de usuário da conta do Twitter
username = 'MercedesAMGF1BR'

# URL da API do Twitter para obter os tweets de um usuário
api_url = f'https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name={username}&count=4'

# Faça uma solicitação GET para a API do Twitter
response = requests.get(api_url, auth=auth)

# Verifique se a solicitação foi bem-sucedida
if response.status_code == 200:
    # Obtenha os dados da resposta (tweets)
    tweets = response.json()

    # Imprima os últimos 4 tweets
    for tweet in tweets:
        print(tweet['text'])
        print('-' * 50)
else:
    print(f'Erro ao acessar a API do Twitter: {response.status_code}')
