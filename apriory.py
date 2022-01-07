# Reglas de asociacion
# Apriori
#
# La gente que compro/vio/hizo esto, tambien compro/vio/hizo ...
# el algoritmo analiza analiza toda una serie de reglas que intentan explicar el comportamiento emparejado, el comportamiento en parejas de acciones por parte de los usuarios
# Son sucesos que aislados no cobran mucho sentido pero que los relacionamos y resulta que ocurre una magia, ocurre una serie de relaciones muy interesantes
#
# Ejemplo:
# sistema de recomendacion de peliculas ol algoritmo de Netflix
#
#                                   USER ID        |                   Peliculas que le han gustado (datos estadisticos)
#                               -------------------|------------------------------------------------------------
#                                    46578         |              Pelicula1, Pelicula2, Pelicula3, Pelicula4                                 Se infieren reglas de asociacion que el algoritmo se
#                                    98989         |              Pelicula1, Pelicula2                                                       encarga de crear, algunas reglas pueden salir de este
#                                    71527         |              pelicula1, Pelicula2, Pelicula4                                            ejemplo, las cuales pueden ser por ejemplo:
#                                    78981         |              Pelicula1, Pelicula2                                                       Si nosotros  miramos los datos sin haber explicado de que
#                                    89192         |              Pelicula2, Pelicula4                                                       va (que hace) el algoritmo uno puede decir
#                                    61557         |              Pelicula1, Pelicula3                                        En general la gente que ve la pelicula1 tambien tiende a mirar la pelicula2
#__________________________________________________________________________________________________________________________   (esto ocurre no con todo el mundo, con el 100% de los usuarios), pero es
#                                                           Pelicula1 --->   Pelicula2                                        robable que a la gente que le gusta la pelicula numero 1 tambien le guste
#                  Reglas Significativas:                   Pelicula2 --->   Pelicula4                                        la pelicula numero2, y del mismo modo, la genmte que ve la pelicula numero2
#                                                           Pelicula1 --->   Pelicula3                                        en varios caso tambien ve la pelicula numero4, por tanto de ahi se inferiria
#                                                                                                                             otra regla de asociacion, o de los que ven la pelicula numero1 tambien hay un
#                                                                                                                             conjunto que ve la pelicula numero3
#
# Parece ser que se establece una serie de relaciones entre las peliculas, evidentemente podemos ver que en estos datos hay una informacion que es interesante explotar, eso es la idea que hay detras de las
#   reglas de asociacion o del algoritmo APRIORI, para aplicar este tipo de algoritmos no se vale tener pocos usuarios o pocos datos, se necesita tener muchos datos para proponer reglas vastantes solidads
#   que ocurran con bastante frecuencia
#
# Pasos para desarrollar el algoritmo:
#
# Paso 1.- Decidir un porcentaje de "soporte" y un porcentaje de "confianza" minimo (pudiera ser del 30% o 20%)
#
# Paso 2.- Elegir todos los subconjuntos de transacciones con soporte superior a el minimo elegido
#
# Paso 3.- Elegir todas las reglas de estos subconjuntos con nivel de confianza superior al minimo elegido
#
# Paso 4.- Ordenar todas las reglas anteriores por lift descendiente

# Importing the libraries ============================================== Optimizar las ventas de un centro comercial =====================================================
# Aplicar reglas de asociacion con el algoritmo de APYORI
# Dentro de la carpeta "apyori.py" ya hay un scrip de pytrhon que vamos a utlizar, no tomaremos un paquete directamente de la libreria de python, no ocuparemos algun "import" del sistema de
#    python, lo que haremos es incorporar el archivo "apyori.py" que es una implementacion del modelo de "Apriori" basada en el software basico de python, este archivo contiene alguna de las clases
#    que utlizaremos para construir las reglas de "Apriori" para nuestro algoritmo, y en general esto significa que no vamos ha tener que depender de ninguna libreria, porque esta implementacion
#    funcionara de lujo
#
# El modelo "Apriori" no necesita dividir el conjunto de datos en datos de entrenamiento y datos de test, ni tampoco separar la matriz de caracteristicas de la variable a predecir (Y)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('D:/TRABAJO ARIHUS IA/PROGRAMAS CURSO IA/P14-Part5-Association-Rule-Learning/Section 28 - Apriori/Python/Market_Basket_Optimisation.csv', header = None)
# "header = None" nos ayuda nos ayuda a modificar el dataset, no habra titulos, no habra una fila de titulos, no habra encabezado en las columnas y dejara el estandar de python
print("dataset".center(150, "-"))
print(dataset)
# El modelo "Apriori" se va a entrenar para el ejemplo de una tienda del sur francia, el gerente de la tienda quiere obtimizar la hubicacion de los productos para conseguir mas ventas
#    queremos conocer las reglas de asociacion de los diferentes productos de la tienda para ver como los clientes cuando van por un producto tambien se llevan otro adicional en su sesta
#    de compra, esta tienda es un lugar muy concurrido
# Son en total 7501 clientes, y cada uno puede llevar un maximo de 20 productos

transactions = []
for i in range(0, 7501):   # Rango de 0 a 7501 elementos que son la cantidad de clientes que tenemos en el dataset, se recorreran todos los items
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)]) # Se toma la lista "transactions" que se inicializo al inicio y colocamos ".append(" para añadir a la lista completa de las
                                                                          #    transacciones la nueva transaccion que estamos procesando, y para ello solo tenemos que identificar de la propia
                                                                          #    transaccion que elementos son los que deben de entrar a formar parte de la misma
# Lo que tendremos al final sera una lista de listas, donde tendremos una gran lista que contenga las listas de cada uno de los productos que compro cada cliente
# Esta es la idea que necesitamos para poder entrenar nuestro algoritmo de Apriori



# Training the Apriori model on the dataset o entrenamiento del algoritmo de Apriori para crear lñas reglas de asociacion
from apyori import apriori    # Añadimos manualmente la libreria local apyori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
# Declaramos una variable llamada reglas de asociacion "rules" que se encarga de tomar las transacciones como entrada y nos devuelva las reglas de asociacion que nos ayuden a entender el
#      comportamiento de los usuarios como salida, esas reglas de asociacion se crearan con la funcion "apriori", la funcion "apriori" toma un unico argumento principal, que son las transacciones,
#      la informacion de las transacciones que en nuestro caso es el dataset de "transactions", esas 7501 listas donde cada una de las cuales contiene los elementos de una transaccion
# "min_support" o soporte minimo y "min_confidence" o confianza minima son argumentos que van a depender ne nuestro problema de negocios, dependera del numero de datos que tengamos en nuestro dataset
# "min_support" o soporte minimo o minimo porsentaje que tienen que aparecer los productos en las sestas de la compra, no es lo mismo si hay 1000 transacciones que si hay 100,000 trnsacciones, que
#      si hay 1,000,000. Con "min_support" queremos establecer cual tiene que ser el soporte minimo de los items, que porcentaje de las cestas de la compra tiene que aparecer el item para que se tenga
#      en cuenta, Tambien se puede decir como el numero de transacciones o la proporcion de transacciones que contienen un determinado item y dividido por el numero total de transacciones, es un
#      argumento que nos indica con que precensia minima debe estar un item para estar considerado como el objetivo de una regla de asociacion, queremos items que se hayan comprado minimo 3 veces al dia
#      7X3=21 (7 dias de la semana por 3 items comprados minimo), con esto buscamos la precencia de items que salgan al menos en 21 cestas de la compra sobre el total del dataframe, para hacer esto
#      tenemos que dividir la frecuencia total de los items con respecto al total del dataset, entonces seria (7X3)/7501=0.00279 donde lo podemos redondear a 0.003
#      ------------Con esto buscamos items cuya frecuencia relativa de compra sea igual o superior al 0.003 porciento-----------------
#
# "min_confidence" o confianza minima para cuando tengamos el soporte establecido en que porcentaje de las cestas de la compra que tenemos que exista esa asociacion. "min_confidence" o confianza minima
#      sirve para establecer en que porcentaje de las cestas de la compra tienen que aparecer los items en conjunto para tenerse en cuenta
# Sabiendo que ya los items se compran un minimo de 21 veces por semana, lo que queremos es saber en que porcentaje de sestas de la compra aparecen esos items juntos
# -----------------El nivel de confianza se deja en un 20%, esto significa que queremos estar seguro de que la regla se cumpla al menos en un 20% de las veces en las que se compra el primer item----------
#
# "min_lift" o minimo valor de la variable lift por si tambien queremos eliminar los que tengan un lift muy bajo.
#
# Con "min_length" o longitud minima sirve para cuando queramos que existan un minimo de dos productos juntos en la cesta de la compra que deriben de la compra del uno del otro, sirve para evitar reglas
#      de asociacion de un solo elemento, cosa que no tiene mucho sentido (ejemplo: la gente solo compra aseitunas, "no", una longitud minima de dos (2) elementos tiene mucho mas sentido),
# --------Se colocan al menos 2 items en la cesta de la compra---------------




# Visualising the results
print("resultados".center(150, "-"))
results = list(rules) # Se convierte "rules" en formato de lista para tener una mejor visualizacion
print(results)

print("resultados en la posicion 0".center(150, "-"))
print(results[0])    # support=0.004532728969470737 o 0.45% de las sestas de la compras incluyeron  el pollo
                     # confidence=0.29059829059829057 en el 29% de las cestas de la compra aparecio el pollo
print("resultados en la posicion 1".center(150, "-"))
print(results[1])
print("resultados en la posicion 2".center(150, "-"))
print(results[2])
print("resultados en la posicion 3".center(150, "-"))
print(results[3])
print("resultados en la posicion 1".center(150, "-"))
print(results[4])