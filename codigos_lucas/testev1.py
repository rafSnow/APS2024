import tkinter as tk
from tkinter import ttk
import praw
from datetime import datetime
import pandas as pd
import os
import re
import unicodedata
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer
import seaborn as sns
import matplotlib.pyplot as plt

# Configurar as credenciais para acessar a API do Reddit
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

def mostrar_dashboard():
    def plot_bar_chart():
        # Ler os dados do CSV
        df = pd.read_csv(r"C:\Users\rafael.nsouza\Documents\GitHub\JucaBiluca\data\reddit_sentiment_data.csv")
        
        # Criar subplots para cada dia
        fig, axs = plt.subplots(1, 2, figsize=(18, 8)) 

        # Contador
        A = 0
        for i in df.columns.values[2:]:
            A += 1
            ax = axs[A - 1]  # Seleciona o subplot correspondente
            sns.barplot(data=df.fillna('NaN'), x='Data', y=i, hue='Palavra-chave', palette='viridis', ax=ax)
            ax.set_ylim(0, 121)  # Definir intervalo do eixo y para 0 a 100
            ax.set_title(i, fontsize=15)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=9, color='red')
            if A >= 7:
                ax.tick_params(axis='x', rotation=45)
        # Ajustar a posição do gráfico para cima
        plt.subplots_adjust(top=5)

        # Adicionar o relatório
        report_text = """
        O gráfico na imagem mostra a porcentagem positiva e negativa associada a várias palavras-chave (desmatamento, incêndios, barragem, chuva e poluição) em diferentes datas.

        Observações com base na descrição do Gráfico:
        - Porcentagem Positiva: Este gráfico mostra a porcentagem de sentimentos positivos associados a cada palavra-chave em diferentes datas. As palavras-chave “chuva” e “Incêndios” têm as maiores porcentagens, indicando que elas são geralmente vistas de forma positiva nas discussões ou contextos analisados.
        - Porcentagem Negativa: Este gráfico mostra a porcentagem de sentimentos negativos associados a cada palavra-chave em diferentes datas. “Poluição” e “Desmatamento” dominam com altas porcentagens, indicando que esses temas são geralmente vistos de forma negativa.
        """

        fig.text(0.5, -0.05, report_text, ha='center', fontsize=12, wrap=True)

        # Layout
        plt.tight_layout(h_pad=2)
        plt.show()

    def plot_line_chart():
        # Código para plotar gráfico de linhas
        pass

    def plot_scatter_chart():
        # Código para plotar gráfico de dispersão
        pass

    def plot_pie_chart():
        # Código para plotar gráfico de pizza
        pass

    # Lista de tipos de gráficos
    tipos_graficos = ["Gráfico de Barras", "Gráfico de Linhas", "Gráfico de Dispersão", "Gráfico de Pizza"]

    # Criar uma janela secundária para os gráficos
    dashboard_window = tk.Toplevel(root)
    dashboard_window.title("Dashboard")

    # Criar um frame para os botões de seleção de gráfico
    frame_botoes = tk.Frame(dashboard_window)
    frame_botoes.pack(pady=10)

    # Criar os botões de seleção de gráfico
    for tipo_grafico in tipos_graficos:
        tk.Button(frame_botoes, text=tipo_grafico, command=lambda t=tipo_grafico: update_plot(t)).pack(side=tk.LEFT, padx=10)

    # Frame para exibir o gráfico selecionado
    frame_grafico = tk.Frame(dashboard_window)
    frame_grafico.pack(expand=True, fill="both")

    # Função para atualizar o frame do gráfico com base no tipo selecionado
    def update_plot(tipo_grafico):
        # Limpar o frame do gráfico
        for widget in frame_grafico.winfo_children():
            widget.destroy()

        # Exibir o gráfico selecionado
        if tipo_grafico == "Gráfico de Barras":
            plot_bar_chart()
        elif tipo_grafico == "Gráfico de Linhas":
            plot_line_chart()
        elif tipo_grafico == "Gráfico de Dispersão":
            plot_scatter_chart()
        elif tipo_grafico == "Gráfico de Pizza":
            plot_pie_chart()

    # Chamar a função update_plot para exibir o primeiro gráfico por padrão
    update_plot(tipos_graficos[0])

    # Criar uma janela secundária para os gráficos
    dashboard_window = tk.Toplevel(root)
    dashboard_window.title("Dashboard")

    # Criar um frame para os botões de seleção de gráfico
    frame_botoes = tk.Frame(dashboard_window)
    frame_botoes.pack(pady=10)

    # Botão para mostrar o gráfico de barras
    tk.Button(frame_botoes, text="Gráfico de Barras", command=plot_bar_chart).pack(side=tk.LEFT, padx=10)

    # Frame para exibir o gráfico selecionado
    frame_grafico = tk.Frame(dashboard_window)
    frame_grafico.pack(expand=True, fill="both")
    
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

# Função para obter dados do Reddit
def obter_dados():
    # Lista de tópicos de interesse (palavras-chave)
    topics = ["desmatamento", "incendios", "barragem", "chuva", "poluição"]
    
    # Lista para armazenar os dados
    data = []
    dataSentiment = [] 
    
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

                # Classificar o sentimento do texto
                sentimento = classify_sentiment(titulo + " " + texto, tokenizer, model)

                # Adicionando os metadados à lista de dados
                data.append([keyword, url, titulo, texto, data_formatada, sentimento])
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
    
        # Armazenar os resultados na lista de dados
        if keyword in topics:
            dataSentiment.append([current_date, keyword, positive_percentage, negative_percentage])
    
    # Criar um DataFrame com os dados
    df = pd.DataFrame(dataSentiment, columns=['Data', 'Palavra Chave', 'Porcentagem Positiva', 'Porcentagem Negativa'])

    # Salvar o DataFrame em um arquivo CSV
    csv_file_path = r'C:\Users\rafael.nsouza\Documents\GitHub\JucaBiluca\data\reddit1.csv'

    # Salvar o DataFrame em um arquivo CSV (modo append)
    df.to_csv(csv_file_path, index=False, mode='a', header=not os.path.isfile(csv_file_path))

    print("Dados salvos com sucesso no arquivo CSV.")
    
    return data

# Função para mostrar a tabela
def mostrar_tabela():
    # Mudar o cursor para um indicador de espera
    root.config(cursor="wait")
    root.update()  # Atualizar a janela para mostrar a mudança no cursor

    global frame_tabela
    
    # Oculta o frame principal
    frame_principal.pack_forget()
    
    # Cria um novo frame para a tabela
    frame_tabela = tk.Frame(root)
    frame_tabela.pack(expand=True, fill="both")
    
    # Cria uma tabela usando o widget Treeview
    tabela = ttk.Treeview(frame_tabela)
    tabela["columns"] = ("Palavra-Chave", "URL", "Título", "Texto", "Data", "Sentimento")
    tabela.heading("#0", text="ID")
    tabela.heading("Palavra-Chave", text="Palavra-Chave")
    tabela.heading("URL", text="URL")
    tabela.heading("Título", text="Título")
    tabela.heading("Texto", text="Texto")
    tabela.heading("Data", text="Data")
    tabela.heading("Sentimento", text="Sentimento")
    
    # Ler os dados do Reddit
    data = obter_dados()
    
    # Adicionar os dados à tabela
    for index, row in enumerate(data):
        tabela.insert("", "end", text=index, values=(row[0], row[1], row[2], row[3], row[4], row[5]))
    
    # Adicionar um botão para analisar sentimentos
    def analisar_sentimento():
        # Obter o item selecionado na tabela
        selected_item = tabela.selection()[0]
        item_values = tabela.item(selected_item)['values']
        
        # Obter o texto do título e do texto
        titulo = item_values[2]
        texto = item_values[3]
        
        # Classificar o sentimento
        sentimento = classify_sentiment(titulo + " " + texto, tokenizer, model)
        
        # Atualizar o valor na tabela
        tabela.item(selected_item, values=(item_values[0], item_values[1], item_values[2], item_values[3], item_values[4], sentimento))
    
    # Botão para analisar sentimentos
    botao_analisar_sentimentos = tk.Button(frame_tabela, text="Analisar Sentimentos", command=analisar_sentimento)
    botao_analisar_sentimentos.pack(side=tk.BOTTOM, pady=10)
    
    tabela.pack(expand=True, fill="both")
    
    # Mudar o cursor de volta para o padrão
    root.config(cursor="")
    root.update()  # Atualizar a janela para mostrar a mudança no cursor

# Criando a janela principal
root = tk.Tk()
root.title("Análise de Sentimentos no Reddit")

# Obtendo as dimensões da tela
largura_tela = root.winfo_screenwidth()
altura_tela = root.winfo_screenheight()

# Configurando as dimensões da janela para ocupar 100% da tela
root.geometry(f"{largura_tela}x{altura_tela}")

# Criando o frame principal
frame_principal = tk.Frame(root)
frame_principal.pack(expand=True, fill="both")

# Botão para mostrar a tabela
botao_mostrar_tabela = tk.Button(frame_principal, text="Mostrar Tabela", command=mostrar_tabela)
botao_mostrar_tabela.pack(pady=10)

# Botão para mostrar o dashboard
botao_mostrar_dashboard = tk.Button(frame_principal, text="Mostrar Dashboard", command=mostrar_dashboard)
botao_mostrar_dashboard.pack(pady=10)

# Botão para sair do programa
botao_sair = tk.Button(root, text="Sair", command=root.quit)
botao_sair.pack(pady=10)

# Executando o loop principal
root.mainloop()

