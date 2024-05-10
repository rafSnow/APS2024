from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import praw
import os
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer # Essa biblioteca é usada para trabalhar com modelos de linguagem
import torch # Essa biblioteca é usada para criar modelos de aprendizado de máquina
import re
import unicodedata


app = Flask(__name__)

reddit = praw.Reddit(
    client_id="0LUIMHwzq6iTBcF4F4zGpQ",
    client_secret="AN90R4CXtXjCpEfXEEVCIKIjReY0NA",
    user_agent="aps",
)

# Carregar o modelo e o tokenizer pré-treinado
model_name = 'neuralmind/bert-base-portuguese-cased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.load_state_dict(torch.load(r'C:\Users\rafael.nsouza\Documents\GitHub\JucaBiluca\codigos\jupyter\model.bin'))

tokenizer = AutoTokenizer.from_pretrained(r'C:\Users\rafael.nsouza\Documents\GitHub\JucaBiluca\codigos\jupyter\tokenizer_dir')

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
def classify_sentiment(text, tokenizer, model):
    # Limpeza de dados
    text = Limpeza_dados(text)
    text = remover_acentos(text)

    # Tokenização
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Passagem para a frente
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Obter a previsão do modelo
    _, preds = torch.max(outputs.logits, dim=1)

    # Converter a previsão em um sentimento
    if preds.item() == 1:
        return "Positivo"
    else:
        return "Negativo"

def obter_dados(method, client_ip):
    # Lista de tópicos de interesse (palavras-chave)
    topics = ["desmatamento", "incendios", "barragem", "chuva", "poluição"]
    
    # Lista para armazenar os dados
    data = []
    
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

                text = titulo + " " + texto

                # Classificar o sentimento do texto
                sentimento = classify_sentiment(text, tokenizer, model)

                # Adicionando os metadados à lista de dados
                data.append([keyword, url, titulo, texto, data_formatada, sentimento, method, client_ip])
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
    
    # Criar um DataFrame com os dados
    df = pd.DataFrame(data, columns=['Palavra Chave', 'URL', 'Título', 'Texto', 'Data', 'Sentimento', 'Método', 'Cliente IP'])

    # Salvar o DataFrame em um arquivo CSV
    csv_file_path = r'C:\Users\rafael.nsouza\Documents\GitHub\JucaBiluca\data\reddit1.csv'

    # Salvar o DataFrame em um arquivo CSV (modo append)
    df.to_csv(csv_file_path, index=False, mode='a', header=not os.path.isfile(csv_file_path))

    print("Dados salvos com sucesso no arquivo CSV.")
    
    return data

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')
    
@app.route('/reddit', methods=['GET'])
def reddit_search():
    method = request.method
    client_ip = request.remote_addr
    dados = obter_dados(method, client_ip)
    return jsonify(dados)

if __name__ == "__main__":
    app.run(debug=True)