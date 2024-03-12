# Imports

# Manipulação de dados
import pandas as pd
import numpy as np

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno

# Estatística
import scipy
from scipy.stats import normaltest
from scipy.stats import chi2_contingency

# Engenharia de Atributos
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import category_encoders as ce

# Ignore Warning
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")



# Importando bibliotecas necessárias para manipulação de dados, visualização, estatísticas e engenharia de atributos.
import pandas as pd                    # Para manipulação de dados tabulares
import numpy as np                     # Para operações matemáticas eficientes
import matplotlib.pyplot as plt        # Para visualização de dados estáticos
import seaborn as sns                  # Para visualização de dados estatísticos
import plotly.express as px            # Para visualização interativa de dados
import plotly.graph_objects as go      # Para visualização interativa de dados
from plotly.subplots import make_subplots  # Para criar subplots interativos
import missingno                       # Para visualização de dados ausentes
import scipy                           # Para funções estatísticas
from scipy.stats import normaltest, chi2_contingency  # Para testes estatísticos
from sklearn.pipeline import Pipeline  # Para criação de pipelines de transformação
from sklearn.impute import SimpleImputer  # Para imputação de valores ausentes
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder  # Para codificação de variáveis categóricas
from sklearn.compose import ColumnTransformer  # Para transformação de colunas específicas
import category_encoders as ce         # Para codificação de variáveis categóricas

# Ignorando Avisos
import sys
import warnings

# Suprimindo avisos para melhorar a legibilidade do código
if not sys.warnoptions:
    warnings.simplefilter("ignore")