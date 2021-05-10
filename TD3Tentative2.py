# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:00:07 2021

@author: petil
"""

import math
from collections import Counter
import pandas as pd
import copy
import time as t
import random

#Variable
nbRandom = 0.75;
pd.set_option('min_rows', 20);
pathApprentissage = 'data.csv';
pathTest = 'preTest.csv';
pathExecutable = 'finalTest.csv';
pathWrite = 'result.txt'
compteurVarConnu = 0;
afficherMatriceConfusion = True;
kMax = 5;

#Return le dataframe du csv permetant le calcul de k
def importDataWithDataFrame(path,fileApprentissage):
    global compteurVarConnu;
    df=pd.read_csv(path)
    if(df.axes[1][0] != '0'):
        df = pd.read_csv(path,header=None)
        df.columns = df.columns.map(str)
    
    if fileApprentissage:
        df['prediction'] = df[str(len(df.columns) - 1)]
    else:
        df[str(len(df.columns))] = "None"
        df['prediction'] = "None"
    compteurVarConnu = compteurVarConnu + len(df);
    return df
    
def createNoneValue(data):
    global compteurVarConnu;
    for i in data.index:
        if(random.random() > nbRandom):
            data.loc[i, 'prediction'] = "None";
            compteurVarConnu = compteurVarConnu - 1;
      
def calculEuclidien(i,j,data):
    res = 0;
    for k in range(0,(len(data.columns) - 2)):
        res = res + (data[str(k)][i] - data[str(k)][j])**2
        
    return [math.sqrt(res),data['prediction'][j]];

def knn(data,i):
    distancePointTest = []
    for j in data.loc[data['prediction'] != "None"].index:
            distancePointTest.append(calculEuclidien(i,j,data));
    
    #on trie
    distancePointTest = sorted(distancePointTest);

    return distancePointTest;
    

def remplissagePoint(data,k,lastRemplissage):
    newdata = copy.deepcopy(data);
    #pour tout les points qui n'ont pas de prédiction
    tabPrediction = []
    for i in newdata.loc[newdata['prediction'] == "None"].index:
            #On calcul distance euclidienne
            knnTab = knn(data,i);
            #On short au k choisi
            FinalChoixTab = [];
            for j in range(0,k):
                FinalChoixTab.append(knnTab[j][1]);
            #On determine la classe
            FinalChoixTab = Counter(FinalChoixTab);
            #on l'ajoute a la tab data
            tabPrediction.append([list(FinalChoixTab.most_common(1)[0])[0],i]);
    for j in tabPrediction:
        newdata.loc[j[1],'prediction'] = j[0]
    if(lastRemplissage):
        print("Remplissage du DataFrame final : OK ... ");
        fichier = open(pathWrite, "w")
        for j in tabPrediction:
            print(j);
            fichier.write(j[0] + "\n")
        fichier.close()
    return newdata;

def MatriceConfusion(data):
    #on affiche le header
    y_actu = pd.Series(data[str(len(data.columns) - 2)], name='Actual')
    y_pred = pd.Series(data['prediction'], name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)

def Prediction(data):
    global compteurVarConnu;
    nombrelementIncorrect = 0;
    #On fait les calcul
    for i in data.index:
        if not (data.loc[i, 'prediction'] == data.loc[i, str(len(data.columns) - 2)]):
            nombrelementIncorrect = nombrelementIncorrect + 1;
    print('nombrelementIncorrect : ',nombrelementIncorrect , "compteurVarConnu",compteurVarConnu);
    
    #On affiche
    if(afficherMatriceConfusion):
        MatriceConfusion(data)
        
    #On renvoie le pourcentage de réussite.
    print("Ratio :" , 100 - ((nombrelementIncorrect * 100)/(len(data) - compteurVarConnu)))
    return 100 - ((nombrelementIncorrect * 100)/(len(data) - compteurVarConnu));

def Bestk(data):
    #On Cherche le meillieur k
    tabBestK = []
    for k in range(1,kMax + 1):
        print("Debut Analyse du k = ",k ," : OK")
        #On rempli notre tableau avec les prévisions de l'algo
        newdata = copy.deepcopy(data)
        newdata = remplissagePoint(newdata,k,False);
                
        #prediction & matrice
        print("Debut Calcul ratioPrediction k = ",k ," : OK")
        ratioPrediction = Prediction(copy.deepcopy(newdata));
        tabBestK.append([k,ratioPrediction,newdata])
        print("Fin Analyse du k = ",k ," : OK")
    
    #on le choisi
    bestkval = tabBestK[0];
    for i in tabBestK[1:]:
        if(bestkval[1] < i[1]):
            bestkval = i;
    return bestkval;

def checklabel():
    allLabels = ['classA','classB','classC','classD','classE']
    #ce fichier s'attend à lire 3000 prédictions, une par ligne
    #réduisez nbLines en période de test.
    nbLines = 3000
    fd =open(pathWrite,'r')
    lines = fd.readlines()
    
    
    count=0
    for label in lines:
    	if label.strip() in allLabels:
    		count+=1
    	else:
    		if count<nbLines:
    			print("Wrong label line:"+str(count+1))
    			break
    if count<nbLines:
    	print("Labels Check : fail!")
    else:
    	print("Labels Check : Successfull!")
                
if __name__ == "__main__":
    #METHODE DATAFRAME
    début=t.process_time()
    print("Lancement de l'IA : OK ... ");
    #on recuperere les data , on remplie les colums d'apprentissage et on trouve le meilleur k
    data = importDataWithDataFrame(pathApprentissage,True).append(importDataWithDataFrame(pathTest,True), ignore_index=True)
    
    createNoneValue(data);
    print("Generation du dataFrame : OK ... ");
    bestApprentissage = Bestk(data)
    print("Valeur de k (",bestApprentissage,") optimal trouvé : OK ... ");
    #on remet les bonne valeur de preTest pour enlever les erreurs
    data['prediction'] = data[str(len(data.columns) - 2)]
        
    #on prédis les valeurs sans outpout
    data = data.append(importDataWithDataFrame(pathExecutable,False),ignore_index=True);
    print("Creation du DataFrame final : OK ... ");
    remplissagePoint(data,bestApprentissage[0],True)
    print("Stockage dans un txt du DataFrame final : OK ... ");
    checklabel()   
    print("Verification du DataFrame final : OK ... ");
    
    fin=t.process_time()
    temps=fin-début
    print("temps d'execution du programme : ",temps, "s")
    print("Fin du programme");