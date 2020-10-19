import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def dadosBasicos (path):
    df = pd.read_json(path, lines=True)
    df = df.drop(columns=['_id'])
    dadosBasicos = np.zeros((1, 16))[:-1]
    s = 0
    for i in range(10000):
        i = i+s
        print (i)
        dadoBasico = df.iloc[i, 3]
        dadoBasico1 = df.iloc[i, 0]
        a = dadoBasico['assunto'][0]
        if tuple(a.items())[1][0] == 'assuntoLocal' or len(dadoBasico) != 12:
            continue
        dadoB = []
        for k in dadoBasico.keys():
            dadoB.append(dadoBasico[k])

        cell = dadoB[0]
        assunto = cell[0]
        ass = []
        for j in assunto.keys():
            ass.append(assunto[j])
        if len(dadoB) > 12:
            continue
        nomeOrgao = dadoB[9]
        nOrgao = []
        for n in nomeOrgao.keys():
            nOrgao.append(nomeOrgao[n])

        listDados = np.array(())
        for m in range(12):
            if m == 0:
                for a in ass:
                    listDados = np.append(listDados, a)
            if m == 9:
                for f in nOrgao:
                    listDados = np.append(listDados, f)
            if m > 0 and m != 9:
                listDados = np.append(listDados, dadoB[m])
        dadosBasicos = np.vstack((dadosBasicos, listDados))
        print(i)
    dadosBasicos = np.delete(dadosBasicos, [0], 1)
    return dadosBasicos

def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(np.concatenate([X_train, X_test], axis=0))
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc
