import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def load_data(path):
    df = pd.read_json(path, lines=True)
    df = df.drop(columns=['_id'])
    dadosBasicos = np.zeros((1, 7))[:-1]
    s = 0
    for i in range(10000):
        one_dbs = np.zeros((1, 7))
        i = i + s
        print(i)
        one_db = df.iloc[i, 0]
        a = one_db['assunto'][0]
        for k in a.keys():
            if k == 'principal' and a[k]=='true':
                one_dbs[0,0]= 1
            if k == 'codigoNacional':
                one_dbs[0,1] = a[k]
            if k == 'assuntoLocal':
                b = a[k]
                for key in b.keys():
                    if key == 'codigoAssunto':
                        one_dbs[0,2] =b[key]
                    if key == 'codigoPaiNacional':
                        one_dbs[0,3] =b[key]
        for j in one_db.keys():
            if j == 'numero':
                digits = [int(x) for x in str(one_db[j])]
                numeroJ = digits [13]
                numero= str(digits[14]) + str(digits [15])
                one_dbs[0,4] = one_db[j]
            if j == 'classeProcessual':
                one_dbs[0,5] = one_db[j]
            if j == 'competencia':
                one_dbs[0,6] = one_db[j]
        dadosBasicos = np.vstack((dadosBasicos, one_dbs))
    return dadosBasicos

def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(np.concatenate([X_train, X_test], axis=0))
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc