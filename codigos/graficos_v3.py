import os
import praw
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o modelo treinado e compilar ao mesmo tempo
model_path = r"C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\modelos\modelo_rnn.h5"
model = load_model(model_path, compile=True)

# Carregar o tokenizer
with open(r"C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\modelos\tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Função para classificar o sentimento de um texto
def classify_sentiment(text, tokenizer, model):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)[0][0]
    return "Positivo" if prediction >= 0.5 else "Negativo"

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
    positive_percentage = (positive_count / total_count) * 100 if total_count > 0 else 0
    negative_percentage = (negative_count / total_count) * 100 if total_count > 0 else 0
    
    
     # Obter a data atual
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Armazenar os resultados na lista de dados
    data.append([current_date, keyword, positive_percentage, negative_percentage])

# Criar um DataFrame com os dados
df = pd.DataFrame(data, columns=['Date', 'Keyword', 'Sentiment', 'Percentage'])

# Define o caminho absoluto para o arquivo CSV
csv_file_path = r"C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\5Periodo\APS\DESENVOLVIMENTO\APS2024\data\reddit_sentiment_data.csv"

# Verificar se o diretório existe e, caso contrário, criá-lo
directory = os.path.dirname(csv_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# Salvar o DataFrame em um arquivo CSV
df.to_csv(csv_file_path, index=False, mode='a', header=not os.path.isfile(csv_file_path))

print("Dados salvos com sucesso no arquivo CSV.")

# Ler o arquivo CSV
df = pd.read_csv(csv_file_path)

# Converter a coluna Date para o tipo datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Plot
# Tamanho da figura
plt.figure(figsize=(18, 30))

# Contador
A = 0

for i in df.columns.values[2:]:
    A += 1
    plt.subplot(5, 2, A)
    ax = sns.barplot(data=df.fillna('NaN'), x='Date', y=i, hue='Keyword', palette='viridis', legend=False)
    plt.title(i, fontsize=15)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)
    if A >= 7:
        plt.xticks(rotation=45)

# Layout
plt.tight_layout(h_pad=2)
plt.show()
