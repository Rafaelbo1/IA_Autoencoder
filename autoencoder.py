import functions as f
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split

rcParams['figure.figsize'] = 12, 6
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

path = "json_filtrados\justica_eleitoral.json"
encoded_seqs = f.load_data(path)
np.random.shuffle(encoded_seqs)
X_train, X_test= train_test_split(encoded_seqs, test_size=0.30, random_state=1, shuffle=True)
#X_train, X_test = f.prepare_inputs(X_train, X_test)
print(encoded_seqs.shape)

# Preparando os dados

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
scaled_seqs = np.vstack((X_train,X_test))

# Construindo um Autoencouder

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

input_dim = X_train.shape[1]  # features
encoding_dim = input_dim

nb_epoch = 1000
batch_size = 128
input_layer = Input(shape=(input_dim,))

encoder = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-50))(input_layer)
encoder = Dense(14, activation="relu")(encoder)
encoder = Dense(14, activation='relu')(encoder)
decoder = Dense(encoding_dim, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model_seqs2.h5",
                               verbose=0,
                               save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit(X_train, X_train,
                          epochs=nb_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          verbose=1,
                          callbacks=[checkpointer, tensorboard]).history

autoencoder = load_model('model_seqs2.h5')
print(f'Min Loss:{np.min(history["loss"])}')
print(f'Max Accuracy:{np.max(history["acc"])}')

# ### Calculate the Error Term
# MSE em termo de erro
predictions = autoencoder.predict(scaled_seqs)
mse = np.mean(np.power(scaled_seqs - predictions, 2), axis=1)
print('MSE:', np.quantile(mse, 0.8))

# ### Carregando novamente a base de dados com as anomalias
# codificar toda base de dados
encoded_seqs = scaled_seqs
# Padronizar
#scaled_data = MinMaxScaler().fit_transform(encoded_seqs)
# fazer a predição com base na rede treinada
predicted = autoencoder.predict(encoded_seqs)
# MSE em termo de erro
mse = np.mean(np.power(encoded_seqs - predicted, 2), axis=1)
#mse = mse.reshape((2490, 1))
seqs_ds = pd.DataFrame(encoded_seqs)
seqs_ds['MSE'] = mse
# Adicionar o MSE ao data frame
#encoded_seqs= np.hstack((encoded_seqs, mse))
# Detectando os erros no data frame
mse_threshold = np.quantile(seqs_ds['MSE'], 0.80)
print(f'MSE 0.99980005 threshhold:{mse_threshold}')

seqs_ds['MSE_Outlier'] = 0
seqs_ds.loc[seqs_ds['MSE'] > mse_threshold, 'MSE_Outlier'] = 1
print(f"Num of MSE outlier:{seqs_ds['MSE_Outlier'].sum()}")


plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['train loss', 'test loss', 'train acc', 'test acc'], loc='upper right')

plt.show()