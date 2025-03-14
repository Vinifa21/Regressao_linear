import numpy as np #biblioteca que facilita a manipulação de dados
from sklearn.linear_model import LinearRegression #importa a classe que representa uma Regressão Linear
from sklearn.metrics import r2_score #importa o calculo de R²
import matplotlib.pyplot as plt # faz gráficos

# Dados:
#Densidade da corrente:
x = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5]).reshape(-1, 1)  # variavel independente # reshape, transforma o array para a formatação esperada
# Pressão do Hidrogênio:
y = np.array([86.1, 92.1, 64.7, 74.7, 223.6, 202.1, 132.9, 413.5, 231.5, 466.7, 365.3, 493.7, 382.3, 447.2, 563.8]) # variavel dependente

# Criar e ajustar o modelo de regressão linear
model = LinearRegression() # cria o objeto de regressão linear
model.fit(x, y) #coloca os dados na regressão linear

# Obter os coeficientes da regressão
intercepto = model.intercept_  # Intercepto (β₀)
inclinacao = model.coef_[0]        # Inclinação (β₁)

print(f"Intercepto (β₀): {intercepto}")
print(f"Inclinação (β₁): {inclinacao}")

# Fazer previsões
y_pred = model.predict(x) # para cada valor de X, mostra o valor de y, de acordo com a reta aproximada
print(f"Previsões: {y_pred}")

# Avaliar o modelo com R²
# quanto mais próximo de 1 é o R², mais confiável é a regressão linear
r2 = r2_score(y, y_pred)
print(f"Coeficiente de Determinação (R²): {r2}")

#imprime a estimativa da equação da reta
if inclinacao >= 0:
    print("Equação da reta: \n", f"y = {intercepto:.2f} + {inclinacao:.2f}x" )
else:
    print("Equação da reta: \n", f"y = {intercepto:.2f} {inclinacao:.2f}x")


#plt é o objeto (gráfico)
# adiciona os testes ao gráfico #
plt.scatter(x, y, color='blue', label='Testes') #gráfico de dispersão

# adiciona a reta de regressão ao gráfico
plt.plot(x, y_pred, color='red', label='Reta ajustada') #reta do gráfico

plt.grid(True)
plt.xlabel('Densidade da corrente (mA/cm²)') #etiqueta os eixos
plt.ylabel('Pressão do Hidrogênio (atm)') # faz legendas
plt.legend()
plt.show()  # mostra o gráfico

#só baixar as libs lá em cima que roda que é uma beleza


# ################################################
# #GRÁFICO QUE UTILIZA LOG PARA CONTROLAR OS RESÍDUOS
# #tudo é calculado de acordo com os vetores x_log e y_log
#
# import numpy as np #biblioteca que facilita a manipulação de dados
# from sklearn.linear_model import LinearRegression #importa a classe que representa uma Regressão Linear
# from sklearn.metrics import r2_score #importa o calculo de R²
# import matplotlib.pyplot as plt # faz gráficos
#
# # Dados:
# #Densidade da corrente:
# x = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5]).reshape(-1, 1)  # variavel independente # reshape, transforma o array para a formatação esperada
# # Pressão do Hidrogênio:
# y = np.array([86.1, 92.1, 64.7, 74.7, 223.6, 202.1, 132.9, 413.5, 231.5, 466.7, 365.3, 493.7, 382.3, 447.2, 563.8]) # variavel dependente
#
# # Aplicação de transformações logarítmicas
# # O logaritmo só pode ser aplicado a valores positivos
# x_log = np.log10(x) # Transformação logarítmica na variável independente
# y_log = np.log10(y) # Transformação logarítmica na variável dependente
#
# # Criar e ajustar o modelo de regressão linear no espaço transformado
# model = LinearRegression() # cria o objeto de regressão linear
# model.fit(x_log, y_log) #coloca os dados transformados na regressão linear
#
# # Obter os coeficientes da regressão no espaço transformado
# intercepto = model.intercept_  # Intercepto (β₀)
# inclinacao = model.coef_[0]        # Inclinação (β₁)
#
# print(f"Intercepto (β₀): {intercepto}")
# print(f"Inclinação (β₁): {inclinacao}")
#
# # Fazer previsões no espaço transformado
# y_log_pred = model.predict(x_log) # para cada valor de X_log, mostra o valor de y_log previsto pela reta ajustada
#
# print(y_log_pred)
#
# # Avaliar o modelo com R² no espaço transformado
# r2 = r2_score(y_log, y_log_pred)
# print(f"Coeficiente de Determinação (R²): {r2}")
#
# #imprime a estimativa da equação da reta no espaço transformado
# if inclinacao >= 0:
#     print("Equação da reta no espaço transformado: \n", f"log(y) = {intercepto:.2f} + {inclinacao:.2f}log(x)" )
# else:
#     print("Equação da reta no espaço transformado: \n", f"log(y) = {intercepto:.2f} {inclinacao:.2f}log(x)")
#
# #plt é o objeto (gráfico)
# # adiciona os testes ao gráfico #
# plt.scatter(x_log, y_log, color='blue', label='Testes (log-transformados)') #gráfico de dispersão no espaço transformado
#
# # adiciona a reta de regressão ao gráfico
# plt.plot(x_log, y_log_pred, color='red', label='Reta ajustada (log-transformada)') #reta do gráfico no espaço transformado
#
# plt.grid(True)
# plt.xlabel('Log(Densidade da corrente) (log(mA/cm²))') #etiqueta os eixos no espaço transformado
# plt.ylabel('Log(Pressão do Hidrogênio) (log(atm))') # faz legendas no espaço transformado
# plt.legend()
# plt.show()  # mostra o gráfico
#
#
#
# Os resíduos do primeiro gráfico estavam variando conforme x aumentava, ou seja, para diferentes valores de x, o teste tinha confiabilidades diferentes
#
# para tentar ajustar ( a amplitude do residuo ser constante para todo x), aplicamos log em todos os valores x e y, a fim de uniformizar.
#
# pq log? pois o log pouco modifica números pequenos  e altera bem numeros grandes(onde a dispersão de resíduos é maior(ex: log 10 = 1 e log 10000 = 4)