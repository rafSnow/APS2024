import os
import praw
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o modelo treinado e compilar ao mesmo tempo
model_path = r"C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\modelos\modelo_rnn.h5"
model = load_model(model_path, compile=True)

# Carregar o tokenizer
with open(r"C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\modelos\tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Função para classificar o sentimento de um texto
def classify_sentiment(text, tokenizer, model):
    # Tokenização e vetorização
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    # Prever o sentimento usando o modelo treinado
    prediction = model.predict(padded_sequence)[0][0]
    if prediction >= 0.5:
        return "Positivo"
    else:
        return "Negativo"

# Configurar as credenciais para acessar a API do Reddit
reddit = praw.Reddit(
    client_id="0LUIMHwzq6iTBcF4F4zGpQ",
    client_secret="AN90R4CXtXjCpEfXEEVCIKIjReY0NA",
    user_agent="aps",
)

# Lista de tópicos de interesse (palavras-chave)
topics = ["deforestation", "forestfires", "floods", "rain", "riverpollution"]

# Criar uma lista para armazenar os dados
data = []

# Iterar sobre cada palavra-chave
for keyword in topics:
    # Contadores para postagens positivas e negativas
    positive_count = 0
    negative_count = 0
    # Iterar sobre os posts do Reddit relacionados à palavra-chave
    for submission in reddit.subreddit("all").search(keyword, sort="hot", time_filter="day", limit=20):
        if submission.selftext.strip() != "" or submission.url.strip() != "":
            # Concatenar o título e o conteúdo do post
            text = submission.title + " " + submission.selftext
            # Classificar o sentimento do texto
            sentiment = classify_sentiment(text, tokenizer, model)
            # Atualizar os contadores com base no sentimento
            if sentiment == "Positivo":
                positive_count += 1
            elif sentiment == "Negativo":
                negative_count += 1
    # Calcular a porcentagem de notícias positivas e negativas
    total_count = positive_count + negative_count
    positive_percentage = (positive_count / total_count) * \
        100 if total_count > 0 else 0
    negative_percentage = (negative_count / total_count) * \
        100 if total_count > 0 else 0
    # Obter a data atual
    current_date = datetime.now().strftime("%Y-%m-%d")
    # Armazenar os resultados na lista de dados
    data.append(
        [current_date, keyword, positive_percentage, negative_percentage])

# Criar um DataFrame com os dados
df = pd.DataFrame(
    data, columns=['Date', 'Keyword', 'Positive Percentage', 'Negative Percentage'])

# Define the absolute path to the CSV file
csv_file_path = r"C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\data\reddit_sentiment_data.csv"

# Check if the directory exists and, if not, create it
directory = os.path.dirname(csv_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False, mode='a', header=not os.path.isfile(csv_file_path))

print("Dados salvos com sucesso no arquivo CSV.")

# Ler o arquivo CSV
df = pd.read_csv(
    r"C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\data\reddit_sentiment_data.csv")

# Converter a coluna Date para o tipo datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Criar uma figura e eixos para o gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Plotar as velas verdes (positivas) para cada palavra-chave sem incluir o parâmetro label
for keyword in df['Keyword'].unique():
    subset = df[df['Keyword'] == keyword]
    ax.bar(subset.index, subset['Positive Percentage'] / 100, color='green', width=0.5)
# Plotar as velas vermelhas (negativas) para cada palavra-chave sem incluir o parâmetro label
    ax.bar(subset.index, subset['Negative Percentage'] / 100, color='red', width=0.5, bottom=subset['Positive Percentage'] / 100)

# Adicionar legendas e título
ax.legend()
ax.set_xlabel('Data')
ax.set_ylabel('Porcentagem')
ax.set_title('Porcentagens positivas e negativas com base nas palavras-chave')

# Adicionar rótulos personalizados para os eixos x e y
ax.set_xticklabels(df.index.strftime('%Y-%m-%d'), rotation=45)
ax.set_yticklabels([f'{val}%' for val in range(0, 101, 10)])

# Mostrar o gráfico
plt.tight_layout()
plt.show()