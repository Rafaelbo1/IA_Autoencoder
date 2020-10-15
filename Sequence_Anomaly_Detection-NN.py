import random
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

rcParams['figure.figsize'] = 12, 6
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

# ### Gerando dados
first_letters =  'ABCDEF'
second_numbers = '120'
last_letters = 'QWOPZXML'

# retornando uma string com o formato: [4 letras =  A-F][1 digito = 0-2][3 letras QWOPZXML]
def get_random_string():
    str1 = ''.join(random.choice(first_letters) for i in range(4))
    str2 = random.choice(second_numbers)
    str3 = ''.join(random.choice(last_letters) for i in range(3))
    return str1+str2+str3
    
print(get_random_string())

#  25,000 vetores no formato especificado
random_sequences = [get_random_string() for i in range(25000)]



#Contruindo o char index que usaremos para codificar as strings em numeros
char_index = 'abcdefghijklmnopqrstuvwxyz'
char_index +='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_index += '0123456789'
char_index += '().,-/+=&$?@#!*:;_[]|%⸏{}\"\'' + ' ' +'\\'

char_to_int = dict((c, i) for i, c in enumerate(char_index))
int_to_char = dict((i, c) for i, c in enumerate(char_index))


#Função de conversão string em numeros (codificador)

def encode_sequence_list(seqs, feat_n=0):
    encoded_seqs = []
    for seq in seqs:
        encoded_seq = [char_to_int[c] for c in seq]
        encoded_seqs.append(encoded_seq)
    if(feat_n > 0):
        encoded_seqs.append(np.zeros(feat_n))
    return pad_sequences(encoded_seqs, padding='post')

#Decodificador
def decode_sequence_list(seqs):
    decoded_seqs = []
    for seq in seqs:
        decoded_seq = [int_to_char[i] for i in seq]
        decoded_seqs.append(decoded_seq)
    return decoded_seqs


# adicionando anomalias na list de dados
random_sequences.extend(['XYDC2DCA', 'TXSX1ABC','RNIU4XRE','AABDXUEI','SDRAC5RF'])
#Salva tudo em um Data Frame
seqs_ds = pd.DataFrame(random_sequences)
# codificando cada string em um array = ex: [[1],[5],[67]], [[45],[76],[7]
encoded_seqs = encode_sequence_list(random_sequences)
# shuffle nos dados
np.random.shuffle(encoded_seqs)
print(random_sequences[10])
print(encoded_seqs[10])

print(encoded_seqs.shape)


#Preparando os dados

scaler = MinMaxScaler()
scaled_seqs = scaler.fit_transform(encoded_seqs)
X_train = scaled_seqs[:20000]
X_test = scaled_seqs[20000:]

#Construindo um Autoencouder

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

input_dim = X_train.shape[1] #features
print(input_dim)
encoding_dim = 8
hidden_dim = int(encoding_dim / 2)

nb_epoch = 30
batch_size = 128

input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-30))(input_layer)
encoder = Dense(hidden_dim, activation="relu")(encoder)
#decoder = Dense(hidden_dim, activation='relu')(encoder)
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
#MSE em termo de erro
predictions = autoencoder.predict(scaled_seqs)
mse = np.mean(np.power(scaled_seqs - predictions, 2), axis=1)
print('MSE:', np.quantile(mse, 0.9999))

# ### Carregando novamente a base de dados com as anomalias
#codificar toda base de dados
encoded_seqs = encode_sequence_list(seqs_ds.iloc[:,0])
#Padronizar
scaled_data = MinMaxScaler().fit_transform(encoded_seqs)
#fazer a predição com base na rede treinada
predicted = autoencoder.predict(scaled_data)
#MSE em termo de erro
mse = np.mean(np.power(scaled_data - predicted, 2), axis=1)
#Adicionar o MSE ao data frame
seqs_ds['MSE'] = mse
#Detectando os erros no data frame
mse_threshold = np.quantile(seqs_ds['MSE'], 0.99980005)
print(f'MSE 0.99980005 threshhold:{mse_threshold}')
seqs_ds['MSE_Outlier'] = 0
seqs_ds.loc[seqs_ds['MSE'] > mse_threshold, 'MSE_Outlier'] = 1
print(f"Num of MSE outlier:{seqs_ds['MSE_Outlier'].sum()}")

Anomalis= seqs_ds.iloc[25000:]
print(Anomalis.to_string())

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