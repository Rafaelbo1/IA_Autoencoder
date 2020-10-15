import numpy as np
import pandas as pd

path = "justica_eleitoral\processos-tre-ac\processos-tre-ac_1.json"

df = pd.read_json(path)

a = df.iloc[0,0]
x = df.iloc[0,3]
b=[]
for i in a.keys():
  b.append(a[i])

removeB= b[0]
assunto = removeB[0]
ass = []
for j in assunto.keys():
  ass.append(assunto[j])


nomeOrgao = b[9]
nOrgao = []
for n in nomeOrgao.keys():
  nOrgao.append(nomeOrgao[n])

listDados = ass


for m in range(12):

    if m ==9:
        for f in nOrgao:
            listDados.append(f)

    if m > 0 and m !=9:
        listDados.append(b[m])
print(listDados)

