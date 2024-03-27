import pandas as pd
import matplotlib.pyplot as plt

# Dados de exemplo
data = {'Date': ['2022-01-01', '2022-01-02', '2022-01-03'],
        'Positive Percentage': [60, 40, 80],
        'Negative Percentage': [40, 60, 20]}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Criar uma figura e eixos para o gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Plotar as velas verdes (positivas)
ax.bar(df.index, df['Positive Percentage'], color='green', width=0.5, label='Positive Percentage')

# Plotar as velas vermelhas (negativas)
ax.bar(df.index, df['Negative Percentage'], color='red', width=0.5, label='Negative Percentage')

# Adicionar legendas e título
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Percentage')
ax.set_title('Positive and Negative Percentages')

# Mostrar o gráfico
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
