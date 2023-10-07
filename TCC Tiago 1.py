import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import random
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np
from sklearn.impute import KNNImputer
#from pycaret.regression import *


'''Neste projeto de pós-graduação em inteligência artificial, o contexto específico escolhido é a previsão de preços de ações em curto prazo utilizando técnicas de aprendizado de máquina. O mercado de ações é conhecido por sua volatilidade e complexidade, tornando a previsão de preços um desafio intrigante e significativo.

Problema a Ser Resolvido:

O problema a ser resolvido é a capacidade de prever com precisão os preços futuros das ações em um horizonte de curto prazo. O objetivo é desenvolver um modelo de aprendizado de máquina capaz de analisar padrões históricos de preços, volumes de negociação e outros indicadores relevantes, a fim de fornecer previsões que auxiliem os investidores na tomada de decisões informadas.

Mais especificamente, o problema consiste em:

    Coletar e preparar dados históricos de preços de ações e indicadores financeiros relevantes usando a biblioteca yfinance e outras fontes confiáveis.
    Realizar análises exploratórias dos dados para identificar tendências, sazonalidades e possíveis padrões.
    Desenvolver e treinar modelos de aprendizado de máquina, como regressão, séries temporais ou redes neurais, para prever os preços das ações em um horizonte de curto prazo.
    Avaliar o desempenho dos modelos usando métricas apropriadas, como o erro médio quadrático (MSE) ou o erro absoluto médio (MAE).
    Implementar um pipeline de previsão que permita a atualização regular dos modelos com novos dados.

Ao resolver esse problema, busca-se não apenas melhorar a compreensão dos padrões de preços de ações, mas também fornecer ferramentas úteis para investidores e profissionais financeiros que desejam tomar decisões de investimento mais informadas.'''

'''CA-1.1 Seleção da Base de Dados

A base de dados selecionada para o desenvolvimento deste projeto é o histórico de preços de ações obtidos por meio da biblioteca yfinance. Essa base de dados contém informações sobre os preços de fechamento diários de diversas ações, bem como indicadores financeiros relacionados. A base de dados é acessada diretamente dos serviços online de dados financeiros, oferecendo uma ampla gama de ativos, como ações de empresas, índices de mercado e commodities.

Detalhes da Base de Dados:

    Número de Instâncias: O número de instâncias na base de dados varia de acordo com a quantidade de dados históricos disponíveis para cada ativo.
    Atributos: Cada instância é caracterizada por uma série temporal de valores que inclui preços de fechamento, preços de abertura, volumes de negociação, entre outros indicadores financeiros relevantes. Essas séries temporais são organizadas em colunas, representando diferentes atributos para cada período de tempo.
    Tipos de Dados: Os tipos de dados incluem números de ponto flutuante para os preços e volumes, datas para os registros de tempo e possivelmente outros tipos de dados para indicadores específicos.'''

# Defina o símbolo de ticker da Apple
ticker_simbolo = "AAPL"

# Use a função 'download' do yfinance para obter os dados históricos
dados_apple = yf.download(ticker_simbolo, start="2013-01-01", end="2023-01-01")


'''Na seção CA-1.2 do seu trabalho, você deve descrever detalhadamente como realizou o tratamento da base de dados antes de usá-la para suas análises e previsões. Aqui está um exemplo de como você pode descrever essa seção:

CA-1.2 Descrição Detalhada do Tratamento da Base de Dados

Antes de prosseguir com as análises e previsões, realizamos um tratamento minucioso nos dados obtidos por meio da biblioteca yfinance. O objetivo desse tratamento foi garantir que os dados estivessem limpos, coerentes e preparados para análises e modelagem de aprendizado de máquina. A seguir, apresentamos em detalhes as etapas de pré-processamento realizadas:'''

'''1. Tratamento de Valores Ausentes:

    Identificamos quaisquer valores ausentes nos dados e avaliamos a relevância desses registros em relação à série temporal.
    Decidimos preencher os valores ausentes usando métodos apropriados, como a técnica de preenchimento com a média dos valores vizinhos. Essa abordagem minimiza a distorção dos dados.
    Não foi encontrado nenhum valor ausente.'''

# Identificar valores ausentes
# Aplicar a imputação baseada em modelos não lineares (KNeighborsRegressor)
knn_imputer = KNNImputer(n_neighbors=5, weights='uniform', missing_values=np.nan)
imputed_data = knn_imputer.fit_transform(dados_apple)

# Atualizar o DataFrame com os dados imputados
dados_apple = pd.DataFrame(imputed_data, columns=dados_apple.columns, index=dados_apple.index)

'''Optou-se manter os outliers, por conta das características do mercado financeiro, apesar de que valores extremos podem influenciar significativamente a inclinação da linha de regressão.
Argumentos para Manter Outliers:

    Informações Relevantes: Em finanças, outliers podem representar eventos significativos, como notícias importantes que afetaram o mercado. Essas informações podem ser relevantes para a previsão de preços.

    Comportamento Volátil: O mercado de ações é volátil e eventos imprevisíveis podem ocorrer. Manter outliers pode ajudar a capturar essa volatilidade.

    Validade Estatística: Remover outliers pode distorcer as estatísticas e propriedades dos dados, levando a resultados enviesados.'''


'''# Calcular os limites superior e inferior para identificar outliers
Q1 = dados_apple.quantile(0.25)
Q3 = dados_apple.quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Identificar e remover outliers
outliers = (dados_apple < limite_inferior) | (dados_apple > limite_superior)
dados_apple_sem_outliers = dados_apple[~outliers.any(axis=1)]

# Verificar a quantidade de registros removidos
print("Quantidade de outliers removidos:", len(dados_apple) - len(dados_apple_sem_outliers))

print('com outliers', dados_apple)
print('sem outliers', dados_apple_sem_outliers)'''


'''Identificação e remoção de Duplicatas:
Duplicatas são observações repetidas que aparecem mais de uma vez no conjunto de dados. Essas repetições podem distorcer análises e resultados, levando a conclusões equivocadas. 
Verificar e remover duplicatas é fundamental para garantir a qualidade, a imparcialidade e a confiabilidade das análises e dos modelos construídos a partir dos dados. Isso ajuda a evitar conclusões errôneas e a assegurar que as interpretações sejam baseadas em informações consistentes e representativas.
Não foi encontrado nenhuma duplicata'''

# Identificação de duplicatas
duplicatas = dados_apple.duplicated()
total_duplicatas = duplicatas.sum()


# Tratamento de duplicatas
dados_apple_sem_duplicatas = dados_apple.drop_duplicates()


'''Identificação e tratamento de datas anômolas:
Nesta abordagem, estamos reindexando os dados para incluir todas as datas no intervalo dos dados e, em seguida, removendo as datas correspondentes a fins de semana e feriados usando o método dropna(). 
Em um contexto de dados de preços de ações, você não deve preencher os dados faltantes de fins de semana e feriados com outros dados. Isso porque o mercado de ações não opera durante esses períodos, e qualquer preenchimento artificial poderia levar a análises enganosas e resultados distorcidos.

A abordagem correta é considerar apenas os dados disponíveis nos dias úteis em que o mercado está aberto e ignorar os fins de semana e feriados. A reindexação do DataFrame para incluir todas as datas e, em seguida, remover as datas correspondentes a fins de semana e feriados, como mencionado anteriormente, é uma forma apropriada de tratar esse cenário.

Ao lidar com dados de séries temporais como preços de ações, é importante manter a integridade dos dados e respeitar o contexto do mercado financeiro. '''

# Criar um índice de datas para incluir todas as datas no intervalo
todas_datas = pd.date_range(start=dados_apple.index.min(), end=dados_apple.index.max(), freq='D')

# Reindexar os dados com todas as datas para preencher lacunas nos fins de semana e feriados
dados_apple = dados_apple.reindex(todas_datas)

# Remover as datas que correspondem a fins de semana e feriados
dados_apple = dados_apple.dropna()

# Verificar o número de registros após o tratamento
print("Número de registros após o tratamento:", len(dados_apple))


'''CA-2.1
Análise, Exploração e Visualização dos Dados'''

'''Análises estatísticas dos dados:
Está sendo observada a contagem (número de observações), a média, o desvio padrão, o mínimo, 25% do percentil, 50% do percentil, 75% do percentil e o valor máximo '''
# Calcular estatísticas descritivas básicas para cada coluna
estatisticas_descritivas = dados_apple.describe()

# Imprimir as estatísticas descritivas
print(estatisticas_descritivas)



'''Gráfico de linha mostrando como os preços de fechamento da Apple variaram ao longo do tempo. Isso pode ajudar a identificar tendências, padrões sazonais ou movimentos significativos no mercado.'''
# Criar o gráfico de linha para os preços de fechamento
plt.figure(figsize=(10, 6))
plt.plot(dados_apple.index, dados_apple['Close'], color='blue')
plt.title('Variação dos Preços de Fechamento da Apple')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento')
plt.grid(True)
plt.show()

# Salvar o gráfico em um arquivo
plt.savefig('D:/Caleb/caleb/Python/TCC Tiago/grafico_preco_fechamento.png')

# Fechar a figura para liberar recursos
plt.close()


'''Histograma e gráfico de densidade para visualizar a distribuição das variáveis. Isso pode ajudar a entender se os preços seguem uma distribuição normal ou se há assimetrias.'''

# Análise Univariada
# Para cada coluna numérica, vamos gerar gráficos e estatísticas univariadas
colunas_numericas = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

for coluna in colunas_numericas:
    plt.figure(figsize=(10, 6))
    
    # Histograma
    plt.subplot(2, 1, 1)
    sns.histplot(dados_apple[coluna], bins=20, kde=False, color='blue')
    plt.title(f'Histograma de {coluna}')
    plt.xlabel(coluna)
    plt.ylabel('Frequência')
    
    # Gráfico de Densidade
    plt.subplot(2, 1, 2)
    sns.kdeplot(dados_apple[coluna], color='blue')
    plt.title(f'Gráfico de Densidade de {coluna}')
    plt.xlabel(coluna)
    plt.ylabel('Densidade')
    
    plt.tight_layout()
    plt.savefig(f'D:/Caleb/caleb/Python/TCC Tiago/univariada_{coluna}.png')
    plt.show()
    
    # Estatísticas Descritivas
    estatisticas_coluna = dados_apple[coluna].describe()
    print(f'Estatísticas Descritivas para {coluna}:\n{estatisticas_coluna}\n')

'''Análise Bivariada'''
# Gráfico de Dispersão (scatter plot) para analisar relações bivariadas
sns.pairplot(dados_apple[colunas_numericas])
plt.suptitle('Gráfico de Dispersão (Scatter Plot)')
plt.savefig('D:/Caleb/caleb/Python/TCC Tiago/analise_bivariada_scatter.png')
plt.show()


'''Análise Multivariada. Matriz de Correlação para analisar relações multivariadas. Calculo da matriz de correlação entre os preços de abertura, fechamento, máximo, mínimo, fechamento ajustado e volume. Represente essa matriz de correlação por meio de um heatmap para destacar as relações entre as variáveis.'''

# Calcular a matriz de correlação entre as variáveis
colunas_para_correlacao = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
correlation_matrix = dados_apple[colunas_para_correlacao].corr()
print(correlation_matrix)

# Criar um heatmap para visualizar as correlações
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".6f")
plt.title('Matriz de Correlação entre Variáveis')
plt.savefig('D:/Caleb/caleb/Python/TCC Tiago/matriz_correlacao.png')
plt.show()

'''Gráfico de linha para visualizar como o volume de negociação varia ao longo do tempo. Isso pode fornecer insights sobre o interesse dos investidores em diferentes períodos.'''
# Criar um gráfico de linha para visualizar o volume de negociação ao longo do tempo
plt.figure(figsize=(10, 6))
dados_apple['Volume'].plot(kind='line', color='blue')
plt.title('Volume de Negociação da Apple ao Longo do Tempo')
plt.xlabel('Data')
plt.ylabel('Volume de Negociação')
plt.grid(True)
plt.savefig('D:/Caleb/caleb/Python/TCC Tiago/volume_negociacao.png')
plt.show()


'''Gráfico boxplot para cada coluna numérica para visualizar a presença de outliers. Isso ajudará a identificar valores extremos que podem impactar sua análise.'''
# Criar gráficos boxplot para visualizar outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=dados_apple, orient='h')
plt.title('Boxplots para Visualização de Outliers')
plt.xlabel('Valores')
plt.savefig('D:/Caleb/caleb/Python/TCC Tiago/boxplots_outliers.png')
plt.show()


'''Técnicas de decomposição de séries temporais, como a decomposição sazonal, para identificar tendências de longo prazo e padrões sazonais nos dados.
    Série Temporal Observada:
    A série temporal observada é a representação original dos dados ao longo do tempo. É a sequência bruta dos preços de fechamento da Apple sem qualquer manipulação ou ajuste. Essa série fornece uma visão geral das flutuações e variações nos preços de fechamento ao longo do período analisado.

    Tendência de Longo Prazo:
    A tendência de longo prazo é uma componente da série temporal que captura a direção geral do movimento dos dados ao longo do tempo. Ela ajuda a identificar se os preços estão subindo, descendo ou permanecendo relativamente estáveis em períodos mais longos. Essa componente pode ser útil para identificar padrões de crescimento ou declínio ao longo dos anos.

    Padrões Sazonais:
    Os padrões sazonais são flutuações regulares e previsíveis que ocorrem em intervalos específicos de tempo, como estações do ano, trimestres ou meses. Essa componente revela repetições de padrões que ocorrem em intervalos consistentes, como aumentos sazonais ou quedas nos preços em determinados períodos. A identificação de padrões sazonais é importante para prever movimentos futuros nos preços.

    Resíduos:
    Os resíduos são a parte da série temporal que não pode ser explicada pelas componentes anteriores (tendência e padrões sazonais). Eles representam as variações aleatórias e imprevisíveis que podem ocorrer nos dados. Analisar os resíduos ajuda a identificar quaisquer padrões ou informações que ainda não foram capturados pelas outras componentes. Resíduos pequenos indicam que a decomposição conseguiu explicar a maior parte das variações nos dados, enquanto resíduos grandes podem sugerir a presença de informações não explicadas.

Essa decomposição sazonal ajuda a entender melhor a estrutura subjacente dos dados, permitindo a identificação de tendências de longo prazo, padrões sazonais e comportamentos aleatórios. Cada componente oferece insights valiosos para análises e previsões de séries temporais, incluindo previsões de preços de ações.'''
# Realizar a decomposição sazonal dos preços de fechamento
result = seasonal_decompose(dados_apple['Close'], model='multiplicative', period=252)  # Periodo de 1 ano (252 dias de negociação))

# Criar subplots para visualizar as componentes da decomposição
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(result.observed)
plt.title('Série Temporal Observada')

plt.subplot(4, 1, 2)
plt.plot(result.trend)
plt.title('Tendência de Longo Prazo')

plt.subplot(4, 1, 3)
plt.plot(result.seasonal)
plt.title('Padrões Sazonais')

plt.subplot(4, 1, 4)
plt.plot(result.resid)
plt.title('Resíduos')

plt.tight_layout()
plt.savefig('D:/Caleb/caleb/Python/TCC Tiago/decomposicao_sazonal.png')
plt.show()


'''Gráfico de candlestick para visualizar os preços de abertura, fechamento, máximo e mínimo em um único gráfico. Isso é comumente usado para análise técnica em finanças.'''
'''# Configurar o estilo do gráfico
estilo = mpf.make_mpf_style(base_mpf_style='yahoo', gridstyle=' ')

# Criar um gráfico de candlestick e salvar em um arquivo
mpf.plot(dados_apple, type='candle', title='Gráfico de Candlestick - Preços da Apple', style=estilo, savefig='candlestick.png')'''


# Pivot
# Vamos fazer um pivot para analisar como os preços de fechamento variam com base no dia da semana
dados_pivot = dados_apple.copy()
dados_pivot['DiaDaSemana'] = dados_pivot.index.day_name()  # Adicionar uma coluna com o dia da semana em português

# Mapear os nomes dos dias da semana para português
mapa_dias = {
    'Monday': 'Segunda-feira',
    'Tuesday': 'Terça-feira',
    'Wednesday': 'Quarta-feira',
    'Thursday': 'Quinta-feira',
    'Friday': 'Sexta-feira'
}
dados_pivot['DiaDaSemana'] = dados_pivot['DiaDaSemana'].map(mapa_dias)

# Calcular a média dos preços de fechamento para cada dia da semana
tabela_pivot = dados_pivot.pivot_table(values='Close', index='DiaDaSemana', aggfunc='mean')

# Ordenar os dias da semana
dias_ordem = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira']
tabela_pivot = tabela_pivot.reindex(dias_ordem)
print(tabela_pivot)

# Criar um gráfico de barras para visualizar as médias de preços de fechamento por dia da semana
plt.figure(figsize=(10, 6))
tabela_pivot.plot(kind='bar', color='blue')
plt.title('Média dos Preços de Fechamento por Dia da Semana')
plt.xlabel('Dia da Semana')
plt.ylabel('Média dos Preços de Fechamento')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('D:/Caleb/caleb/Python/TCC Tiago/pivot_media_preco_dia_semana.png')
plt.show()

# Pivot por mês
pivot_mes = dados_apple.pivot_table(values='Close', index=dados_apple.index.month, aggfunc='mean')

# Plotar o gráfico de linhas
pivot_mes.plot(kind='line', figsize=(10, 6))
plt.title('Média de Preço de Fechamento por Mês')
plt.xlabel('Mês')
plt.ylabel('Média de Preço de Fechamento')
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True)

# Salvar o gráfico em um arquivo
plt.savefig('D:/Caleb/caleb/Python/TCC Tiago/grafico_media_preco_mes.png')

plt.show()


'''Feature Engineering
 Feature Engineering envolve a criação ou transformação de variáveis (features) que podem melhorar o desempenho dos modelos de aprendizado de máquina.'''

''' Lag Features (Atraso Temporal): Criar novas colunas com os valores anteriores de algumas variáveis. Isso pode ajudar o modelo a capturar relações temporais.'''
# Criar lag features para o preço de fechamento
for lag in range(1, 6):  # Criar lags de 1 a 5 dias
    dados_apple[f'Close_Lag_{lag}'] = dados_apple['Close'].shift(lag)

# Exibir as primeiras linhas dos dados com as novas colunas
print(dados_apple.head())

''' Médias Móveis: Calcular médias móveis para diferentes janelas temporais. Isso pode ajudar a capturar tendências de curto e médio prazo.'''
# Calcular médias móveis de 5 e 10 dias para o preço de fechamento
dados_apple['Close_MA_4'] = dados_apple['Close'].rolling(window=4).mean()
dados_apple['Close_MA_16'] = dados_apple['Close'].rolling(window=16).mean()

# Exibir as primeiras linhas dos dados com as novas colunas
print(dados_apple.head())

'''Variáveis de Tempo: Extrair informações de data, como dia da semana, mês, trimestre, ano, etc.'''
# Criar colunas com informações de data
# Mapear os dias da semana para números inteiros
dia_da_semana_mapping = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5
}

# Aplicar o mapeamento aos dados
dados_apple['DiaDaSemana'] = dados_apple.index.day_name()
dados_apple['DiaDaSemana'] = dados_apple['DiaDaSemana'].map(dia_da_semana_mapping)
dados_apple['Mes'] = dados_apple.index.month
dados_apple['Trimestre'] = dados_apple.index.quarter
dados_apple['Ano'] = dados_apple.index.year

# Exibir as primeiras linhas dos dados com as novas colunas
print(dados_apple.head())

'''Indicadores Técnicos: Calcular indicadores técnicos comuns, como RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence)'''

'''O RSI, ou Relative Strength Index, é um indicador técnico utilizado na análise técnica de preços de ativos financeiros, como ações e moedas, para avaliar a força e a velocidade das mudanças de preço. Ele é calculado com base nas médias das variações positivas e negativas dos preços em um determinado período. O RSI é normalmente apresentado em uma escala de 0 a 100 e é frequentemente utilizado para identificar condições de sobrecompra e sobrevenda de um ativo.

O RSI compara o ganho médio (mudanças positivas) e a perda média (mudanças negativas) em um intervalo de tempo especificado, geralmente 14 períodos. Quanto maior o valor do RSI, mais força é indicada na alta dos preços. Por outro lado, um valor menor sugere uma força maior nas quedas dos preços.'''
# Calcular o Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

dados_apple['RSI'] = calculate_rsi(dados_apple['Close'])

# Exibir as primeiras linhas dos dados com a nova coluna
print(dados_apple.head())

'''O indicador MACD é calculado a partir das médias móveis exponenciais (EMAs) de preços. Ele é composto por três linhas: a linha MACD, a linha de sinal e o histograma. Essas linhas podem oferecer insights sobre tendências e reversões no mercado.'''
# Calcular o Moving Average Convergence Divergence (MACD)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

macd_line, signal_line, histogram = calculate_macd(dados_apple['Close'])
dados_apple['MACD_Line'] = macd_line
dados_apple['MACD_Signal_Line'] = signal_line
dados_apple['MACD_Histogram'] = histogram

# Exibir as primeiras linhas dos dados com as novas colunas
pd.set_option('display.max_columns', None)
dados_apple = dados_apple.fillna(method='bfill')
print(dados_apple)


'''CA-2.2
Preparação de Dados para o Modelo'''

'''Normalização dos dados
StandardScaler:

    Padroniza os dados para que tenham média zero e desvio padrão um.
    Não é tão sensível a outliers quanto o MinMaxScaler.
    Pode ser mais apropriado quando você não tem informações sobre a distribuição dos dados ou quer lidar melhor com outliers.'''
# Inicializar o StandardScaler
scaler = StandardScaler()

# Selecionar as colunas numéricas para normalização
colunas_numericas = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Aplicar o StandardScaler nas colunas selecionadas
dados_normalizados = dados_apple.copy()
dados_normalizados[colunas_numericas] = scaler.fit_transform(dados_normalizados[colunas_numericas])

# Exibir as primeiras linhas dos dados normalizados
print('dados_normalizados', dados_normalizados)



'''Separação em dados de treinamento e teste.
Separar os dados em conjuntos de treinamento e teste é uma prática fundamental no processo de desenvolvimento de modelos de aprendizado de máquina. Essa divisão desempenha um papel crucial na avaliação objetiva e no aprimoramento do desempenho do modelo. Aqui estão algumas razões pelas quais essa separação é importante:

    Avaliação Realista: A separação dos dados permite simular o desempenho do modelo em dados não vistos, ou seja, em situações reais. Isso ajuda a avaliar a capacidade do modelo de generalizar para novos dados, o que é essencial para evitar overfitting.

    Prevenção de Overfitting: Quando um modelo se ajusta excessivamente aos dados de treinamento, ele pode capturar ruídos e padrões aleatórios que não se aplicam a novos dados. A separação dos conjuntos de treinamento e teste ajuda a identificar se o modelo está superajustando os dados de treinamento.

    Seleção de Hiperparâmetros: A divisão dos dados permite ajustar os hiperparâmetros do modelo com base no desempenho em um conjunto independente de dados de teste. Isso ajuda a encontrar um equilíbrio entre a complexidade do modelo e sua capacidade de generalização.

    Avaliação de Desempenho: Ao avaliar o desempenho do modelo no conjunto de teste, é possível calcular métricas como acurácia, precisão, recall, F1-score, entre outras. Essas métricas fornecem uma medida objetiva do quão bem o modelo está se saindo em tarefas específicas.

    Tomada de Decisões Informadas: Uma vez que o modelo é treinado e avaliado no conjunto de teste, é possível tomar decisões informadas com base em seu desempenho. Isso inclui a seleção do melhor modelo, ajustes finos e a implantação em cenários do mundo real.'''

# Dividir os dados em conjunto de treinamento e teste (80% treinamento, 20% teste)
train_data, test_data = train_test_split(dados_normalizados, test_size=0.2, random_state=42, shuffle=True)
# Embaralhar apenas os dados de treinamento
train_data = train_data.sample(frac=1, random_state=42)

# Exibir o número de amostras em cada conjunto
print("Amostras de Treinamento:", train_data)
print("Amostras de Teste:", test_data)


# Funções de pontuação personalizadas para MAE, MSE e R²
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score, greater_is_better=True)

# Espaços de busca para os hiperparâmetros de cada modelo
param_space = {
    "Linear Regression": {
        "fit_intercept": [True, False],
        "positive": [True, False]
    },
    "Random Forest": {
        "n_estimators": Integer(8, 512),
        "max_depth": Integer(2, 64)
    },
    "Gradient Boosting": {
        "n_estimators": Integer(8, 512),
        "max_depth": Integer(2, 64),
        "learning_rate": Real(0.001, 0.3, prior='log-uniform')
    }
}

# Lista de modelos
models = [
    {"name": "Regressão Linear", "model": LinearRegression(), "params": param_space["Linear Regression"]},
    {"name": "Random Forest", "model": RandomForestRegressor(random_state=42), "params": param_space["Random Forest"]},
    {"name": "Gradient Boosting", "model": GradientBoostingRegressor(random_state=42), "params": param_space["Gradient Boosting"]}
]

# Dicionário para armazenar as métricas
metrics_dict = {}

# Array para armazenar previsões dos modelos
model_predictions = []

# Definir os pesos para cada métrica
weights = {"mse": -1, "mae": -1, "r2": 1}




# Loop para treinar modelos, fazer previsões e calcular métricas usando o BayesSearchCV
for model_info in models:
    model_name = model_info["name"]
    model = model_info["model"]
    model_params = model_info["params"]
    
    # Criar o BayesSearchCV para otimização
    search = BayesSearchCV(model, model_params, n_iter=100, cv=5, n_jobs=-1, random_state=42, scoring=mse_scorer)
    
    # Treinamento e otimização
    search.fit(train_data.drop('Close', axis=1), train_data['Close'])
    print('ASDF')
    # Melhores hiperparâmetros
    best_params = search.best_params_
    
    # Melhor modelo
    best_model = search.best_estimator_
    
    # Previsões
    predictions = best_model.predict(test_data.drop('Close', axis=1))
    model_predictions.append(predictions)  # Armazenar previsões
    
    # Cálculo das métricas MAE, MSE e R²
    mae = mean_absolute_error(test_data['Close'], predictions)
    mse = mean_squared_error(test_data['Close'], predictions)
    r2 = r2_score(test_data['Close'], predictions)
    
    # Calcular a métrica composta
    
    metrics_dict[model_name] = mse

    # Exibir resultados
    print(f"{model_name}:")
    print("Erro Médio Absoluto:", mae)
    print("Erro Médio Quadrático:", mse)
    print("R²:", r2)
    print("Melhores Hiperparâmetros:", best_params)
    print()

# Escolher o modelo com a maior pontuação ponderada
best_model = min(metrics_dict, key=metrics_dict.get)

print("Melhor modelo:", best_model)



# Ordenar conjuntos de treinamento e teste pelo índice (datas)
#train_data = train_data.sort_index()
#test_data = test_data.sort_index()
#print("test data:", test_data)
# Calcular os retornos diários
test_data['Daily_Return'] = test_data['Close'].pct_change()

# Calcular os retornos acumulados do "hold"
test_data['Hold_Return'] = (1 + test_data['Daily_Return']).cumprod()

# Calcular os retornos acumulados dos modelos
for i, model_info in enumerate(models):
    model_name = model_info["name"]
    model_predictions[i]  # Usar as previsões armazenadas
    test_data[f'{model_name}_Return'] = (1 + (model_predictions[i] - test_data['Close'].shift()) / test_data['Close'].shift()).cumprod()

# Criar um DataFrame para comparação
comparison_data = test_data[['Hold_Return'] + [f'{model_info["name"]}_Return' for model_info in models]]
print(comparison_data)


# Criar um gráfico de linha
plt.figure(figsize=(10, 6))
plt.plot(comparison_data.index, comparison_data)
plt.title('Comparação de Retornos: Hold vs. Modelos')
plt.xlabel('Data')
plt.ylabel('Retorno Acumulado')
plt.legend(['Hold'] + [model_info["name"] for model_info in models])
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Salvar o gráfico em um arquivo
plt.savefig('D:/Caleb/caleb/Python/TCC Tiago/comparacao_retornos.png')

# Exibir o gráfico
plt.show()


# Calcular a soma dos retornos acumulados
sum_returns = comparison_data[['Regressão Linear_Return', 'Random Forest_Return', 'Gradient Boosting_Return']].cumsum()
# Calcular a soma dos retornos acumulados do "hold"
sum_returns['Hold_Return'] = test_data['Hold_Return'].cumsum()


# Exibir a soma dos retornos acumulados
print("Soma dos Retornos Acumulados:")
print(sum_returns)


# Criar um gráfico de linha
plt.figure(figsize=(10, 6))
plt.plot(sum_returns.index, sum_returns)
plt.title('Comparação da Soma dos Retornos: Hold vs. Modelos')
plt.xlabel('Data')
plt.ylabel('Retorno Acumulado')
plt.legend(['Hold'] + [model_info["name"] for model_info in models])
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Salvar o gráfico em um arquivo
plt.savefig('D:/Caleb/caleb/Python/TCC Tiago/comparacao_soma_retornos.png')

# Exibir o gráfico
plt.show()


'''s = setup(data=dados_apple, target='Close', train_size=0.8, session_id=123)



# treina e ajusta vários modelos
best_model = tune_model(compare_models(), optimize='MAE')
# Combina o melhor modelo ajustado com o modelo original
ensemble = ensemble_model(best_model)

# Avalia o desempenho do modelo combinado
avaliação = evaluate_model(ensemble)
print(best_model)
# faz a previsão com o melhor modelo
predictions = predict_model(ensemble, dados_apple)
plt_model(best_model, plot='feature')
plt.savefig('pycaret.png')
plt.show()

# mostra as previsões
print(predictions)'''


'''# Exemplo de estratégia de investimento baseada em previsões
for i, model_info in enumerate(models):
    model_name = model_info["name"]
    
    # Usar as previsões armazenadas
    predictions = model_predictions[i]
    
    # Adicionar uma coluna com as previsões no DataFrame de teste
    test_data[f'{model_name}_Prediction'] = predictions
    
    # Criar uma coluna que indica quando comprar ou vender
    test_data[f'{model_name}_Action'] = 'Hold'  # Inicialmente, segure a posição
    
    for index, row in test_data.iterrows():
        if row[f'{model_name}_Prediction'] > row['Close']:  # Previsão indica aumento no preço
            test_data.at[index, f'{model_name}_Action'] = 'Buy'
        elif row[f'{model_name}_Prediction'] < row['Close']:  # Previsão indica queda no preço
            test_data.at[index, f'{model_name}_Action'] = 'Sell'
    
    # Calcular os retornos acumulados da estratégia
    test_data[f'{model_name}_Strategy_Return'] = 1  # Retorno inicial
    for index, row in test_data.iterrows():
        if row[f'{model_name}_Action'] == 'Buy':
            test_data.at[index, f'{model_name}_Strategy_Return'] = (1 + row['Daily_Return'])
        elif row[f'{model_name}_Action'] == 'Sell':
            test_data.at[index, f'{model_name}_Strategy_Return'] = (1 - row['Daily_Return'])
    
# Calcular a soma dos retornos acumulados da estratégia
strategy_returns = test_data[[f'{model_info["name"]}_Strategy_Return' for model_info in models]].cumsum()

# Exibir a soma dos retornos acumulados da estratégia
print("Soma dos Retornos Acumulados da Estratégia:")
print(strategy_returns)


# Calcular a soma dos retornos acumulados da estratégia
strategy_returns = test_data[[f'{model_info["name"]}_Strategy_Return' for model_info in models]].cumsum()

# Criar um gráfico de linha para comparar os retornos acumulados da estratégia com o "hold"
plt.figure(figsize=(10, 6))
plt.plot(strategy_returns.index, strategy_returns)
plt.plot(sum_returns.index, sum_returns['Hold_Return'], linestyle='dashed', color='black')
plt.title('Comparação de Retornos Acumulados: Estratégia vs. Hold')
plt.xlabel('Data')
plt.ylabel('Retorno Acumulado')
plt.legend([f'{model_info["name"]} Strategy' for model_info in models] + ['Hold'])
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Salvar o gráfico em um arquivo
plt.savefig('comparacao_retornos_estrategia.png')

# Exibir o gráfico
plt.show()'''