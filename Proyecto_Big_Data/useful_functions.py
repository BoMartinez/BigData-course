# Conjunto de funciones personalizadas para facilitar el desarrollo del proyecto
# y la implementación de los algoritomos de los Sistemas de Recomendación

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
import dask


# Histograma
# %matplotlib inline
def histogram(ratings, title, xlab, n_bins = 100, density = False, range = [0,100]):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    n, bins, patches = ax.hist(ratings, bins = n_bins, density = density, range = range)
    plt.xlabel(xlab)
    if density:
      plt.ylabel('Densidad')
    else:
      plt.ylabel('Número de casos')
    plt.title(title)


# Datos comunes por columna
def only_common_data(df1, df2, column):
  '''
  Retorna dos conjuntos de datos que comparten todos los registros de una columna en común
  column:       string con el nombre de la columna en común
  '''
  data1 = df1.copy()
  data2 = df2.copy()
  a = data1[column].unique()
  b = data2[column].unique()
  # union de las series
  union = pd.Series(np.union1d(a, b))
  # interseccion de las 
  intersect = pd.Series(np.intersect1d(a, b))
  # elementos que no son comunes
  notcommonseries = union[~union.isin(intersect)]
  # ajustando los conjuntos de datos
  data1 = data1[~data1[column].isin(notcommonseries)]
  data2 = data2[~data2[column].isin(notcommonseries)]
  return data1, data2


# Nivel de escasez (sparsity)
def sparsity(data, user_label, movie_label):
  '''
  Mide que tan escaso es un conjunto de datos de puntuaciones, en términos,
  de la densidad de puntuaciones por usuarios y por película
  data:         Dataset de puntuaciones
  user_label:   string con el label del id del usuario    
  movie_label:  string con el label del id de la pelicula
  '''
  n_users = data[user_label].nunique()
  n_movies = data[movie_label].nunique()
  return 1.0 - len(data) / float(n_users*n_movies)
  

# Reducción dimensionalidad
def dim_reduction(data, r_p_user, r_p_movie, user_label, movie_label):
  '''
  Función para reducir la dimensionalidad del conjunto de datos de puntuaciones.
  data:         Dataset de puntuaciones
  r_p_user:     número mínimo de puntuaciones por usuario
  r_p_movie:    número mínimo de puntuaciones por película
  user_label:   string con el label del id del usuario    
  movie_label:  string con el label del id de la pelicula
  '''
  # puntuaciones por usuario y por película
  df1 = data.groupby(user_label).size().reset_index(name = 'count_user')
  df2 = data.groupby(movie_label).size().reset_index(name = 'count_movie')
  df3 = data.merge(df1, 'left', on = user_label).merge(df2, 'left', on = movie_label)
  # filtro inicial
  data_reduced = df3[(df3['count_movie'] > r_p_movie) & (df3['count_user'] > r_p_user)]
  data_reduced = data_reduced.drop(['count_user', 'count_movie'], axis = 1)
  # filtra los usuarios con pocas puntuaciones
  tmp1 = data_reduced.groupby(user_label).size()
  data_reduced = data_reduced[data_reduced[user_label].isin(tmp1.index[tmp1 > 20])]
  # filtra las películas con pocas puntuaciones
  tmp2 = data_reduced.groupby(movie_label).size()
  data_reduced = data_reduced[data_reduced[movie_label].isin(tmp2.index[tmp2 > 20])]

  return data_reduced


# Matriz usuario-item
def user_item_matrix(ratings, user_label, movie_label, mode='train_test', test_size=0.2, random_state=0):
  '''
  Construye la matriz usuario-pelicula. Las filas corresponden a cada usuario y las columnas a cada película
  del conjunto de datos de puntuaciones.
  ratings:      Dataset de puntuaciones
  user_label:   string con el label del id del usuario    
  movie_label:  string con el label del id de la pelicula
  mode:         'train_test' (por defecto) devuelve las matrices usuario-item de entrenamiento y evaluación a partir de un conjunto de datos. 'trainset' devuelve la matriz usuario-item de un conjunto de datos
  '''
  data = ratings.copy()
  n_usuarios = data[user_label].nunique()
  n_items = data[movie_label].nunique()
  # un indice nuevo para los usuarios
  i_usuario = {user:i for i,user in enumerate(data[user_label].unique())}
  data['i_usuario'] = data[user_label].map(i_usuario)
  # un indice nuevo para las películas
  i_pelicula = {peli:i for i,peli in enumerate(data[movie_label].unique())}
  data['i_pelicula'] = data[movie_label].map(i_pelicula)
  
  if mode == 'train_test':
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, shuffle=True)
    # matriz de entrenamiento usuario-item
    train_matrix = np.zeros((n_usuarios, n_items))
    for fila in train.itertuples(index=False):
      train_matrix[fila[3], fila[4]] = fila[2]
    # matriz de evaluación usuario-item
    test_matrix = np.zeros((n_usuarios, n_items))
    for fila in test.itertuples(index=False):
      test_matrix[fila[3], fila[4]] = fila[2]
    return train_matrix, test_matrix

  elif mode == 'trainset':
  # matriz user-item
    matrix_user_item = np.zeros((n_usuarios, n_items))
    for fila in data.itertuples(index=False):
      matrix_user_item[fila[3], fila[4]] = fila[2]
    return matrix_user_item



# Conversion de identificadores de usuarios y películas
def get_id(id, ratings_userId, ratings_movieId, kind='user', mode='raw_to_mat'):
  '''
  Función que devuelve el id en el dataset ratings (matriz usuario-item) de un usuario/pelicula a partir del id en la matriz usuario-item (dataset ratings). Retorna -999 si el id no se encuentra en las bases de datos.
  ratings_userId:   Serie con los id de los usuarios del dataset original de puntuaciones
  ratings_movieId:  Serie con los id de las películas del dataset original de puntuaciones
  id:               identificador que se quiere convertir
  kind:             'user' (por defecto): id de los usuarios, 'item': id de las películas
  mode:             'raw_to_mat' (por defecto): retorna el id en la matriz usuario-item a partir del id original. 'mat_to_raw': retorna el id original a partir del id en la matriz usuario-item.
  '''
  if kind=='user':
    mat_id = {user:i for i,user in enumerate(ratings_userId.unique())}
    raw_id = {i:user for i,user in enumerate(ratings_userId.unique())}
  
    if mode=='raw_to_mat':
      return [mat_id[i] if i in mat_id.keys() else -999 for i in id]
    if mode=='mat_to_raw':
      return [raw_id[i] if i in raw_id.values() else -999 for i in id]
  
  if kind=='item':
    mat_id = {peli:i for i,peli in enumerate(ratings_movieId.unique())}
    raw_id = {i:peli for i,peli in enumerate(ratings_movieId.unique())}
  
    if mode=='raw_to_mat':
      return [mat_id[i] if i in mat_id.keys() else -999 for i in id]
    if mode=='mat_to_raw':
      return [raw_id[i] if i in raw_id.values() else -999 for i in id]
  

# Gestor de recomendaciones de películas
def get_recommendation(search, n, ratings_userId, ratings_movieId, meta_movie, movie_similarity, user_item_matrix=None, prediction=None, mode='by_movie'):
  '''
  Retorna la recomendación de n películas por usuario o película consultada.
  mode:         'by_movie' (por defecto), recomendación de películas por título. 'by_user', recomendación de películas por usuario
  search:       lista con títulos de películas si la recomendación es por película, lista con id de usuarios si la recomendación es por usuario
  n:            número de recomendaciones
  ratings_movieId:      Serie con los id de las películas del dataset original de puntuaciones
  ratings_userId:       Serie con los id de los usuarios del dataset original de puntuaciones
  meta_movie:           Dataset de películas, columnas: id y título
  movie_similarity:     matriz de similaridad basada en items
  user_item_matrix:     matriz usuario-item
  prediction:           matriz usuario-item de predicciones
  '''
  data = meta_movie.copy()
  data['mat_id'] = get_id(id = data.iloc[:, 0],
                          ratings_userId = ratings_userId,
                          ratings_movieId = ratings_movieId,
                          kind = 'item',
                          mode = 'raw_to_mat')
  
  if mode=='by_movie':
    # indice en la matriz de similaridad a partir de la consulta del usuario
    mat_id = []
    for title in search:
      filtro = data.iloc[:,1].str.contains(title, case=False, na=False)
      movie_id = data['mat_id'][filtro].values[0]
      mat_id.append(movie_id)
    # indices con las peliculas más similares para cada titulo consultado
    recommendation = [list(movie_similarity[id, ].argsort()[-n-1:-1][::-1]) for id in mat_id]
    # recomendaciones de películas
    print('Recomendación por Filtrado Colaborativo basado en items')
    print('\n')
    for t in range(len(search)):
      query = data[data['mat_id']==mat_id[t]].values[0][1]
      filtro = data['mat_id'].isin(recommendation[t])
      output = meta_movie.iloc[:,1][filtro]
      print(f'Película consultada: {query}')
      print('*'*30)
      print('Películas recomendadas:')
      for i,r in enumerate(output):
        print(f'{i+1}. {r}')
      print('\n')
  
  if mode=='by_user':
    user = get_id(id=search,
                  ratings_userId=ratings_userId,
                  ratings_movieId = ratings_movieId,
                  kind='user',
                  mode='raw_to_mat')

    pred = prediction[user, ].copy()

    # solo predicciones de peliculas no puntuadas
    mask = user_item_matrix[user, ] != 0
    pred[mask] = 0
    # id de las peliculas con mayor prediccion
    recommendation = pred.argsort()[:, -n:][:, ::-1]
    get_title = lambda x: [data[data['mat_id']==id].iloc[:,1].values[0] for id in x]
    # recomendacion de películas
    titles = [get_title(re) for re in recommendation]

    print('Recomendación por Filtrado Colaborativo basado en items')
    print('\n')
    for i,user in enumerate(search):
      print(f'User: {user}')
      print('*'*30)
      print('Películas recomendadas:')
      for r,title in enumerate(titles[i]):
        print(f'{r+1}. {title}')
      print('\n')


# Medida de similaridad
@dask.delayed
def calc_similarity(user_item_matrix, kind='user', metric='correlation'):
  '''
  Función retrasada con el dedorador de Dask para realizar los cálculos de forma paralela.
  user_item_matrix:   matriz usuario-item
  metric:           medida de similaridad, 'correlation' (por defecto) Coeficiente de correlación de Pearson. 'cosine' Similaridad Coseno ().
  kind:             'user'(por defecto): basada en usarios o usuario-usuario, 'item': basada en items o item-item
  '''
  if kind == 'user':
    similarity = 1 - pairwise_distances(user_item_matrix, metric=metric, n_jobs=-1)
  elif kind == 'item':
    similarity = 1 - pairwise_distances(user_item_matrix.T, metric=metric, n_jobs=-1)
  return similarity


# Predicción de puntuaciones
def predict_CF(user_item_matrix, similarity, kind='user'):
  '''
  Estima por el método de Filtrado Colaborativo la puntuación de cada usuario para cada película
  a partir de las similaridades (usuaro-usuario y pelicula-pelicula)
  user_item_matrix:   matriz usuario-item
  similarity:       matriz de similaridades
  kind:             'user': para FC basado en usuarios, 'item': para FC basado en usuarios
  '''
  n, p = user_item_matrix.shape
  if kind == 'user':
    mean_user_rating = user_item_matrix.mean(axis=1)  
    user_item_matrix_diff = (user_item_matrix - mean_user_rating[:, np.newaxis])
    pred = mean_user_rating[:, np.newaxis] + similarity.dot(user_item_matrix_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
  
  elif kind == 'item':
    pred = user_item_matrix.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
  
  return pred.reshape((n, p))
 

# Función para calcular el RMSE
def rmse(pred, actual):
  '''
  La función compara directamente la prediccion de la puntuacion con las puntuaciones del conjunto de evaluación, esto es para cada usuario
  y para cada película del conjunto de evaluación. Es decir, omite las puntuaciones cero (peliculas por fuera del conjunto de evaluación)
  pred:     matriz usuario-item de predicciones
  actual:   matriz usuario-item de evaluación
  '''
  pred = pred[actual.nonzero()].flatten()
  actual = actual[actual.nonzero()].flatten()
  return (mean_squared_error(pred, actual))**.5