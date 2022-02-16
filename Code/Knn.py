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
import random as rd

#Variable
pd.set_option('min_rows', 20)
pathApprentissage = 'data.csv';
pathTest = 'preTest.csv';
pathExecutable = 'finalTest.csv';
pathWrite = 'result.txt'
compteurVarConnu = 0;
afficherMatriceConfusion = True;
kMax = 10;
ratioValApprentissage = 0.75;

#Return le dataframe du csv fournit dans le path
def importDataWithDataFrame(path):
    df=pd.read_csv(path)
    if(df.axes[1][0] != '0'):
        df = pd.read_csv(path,header=None)
        df.columns = df.columns.map(str)
    
    df["prediction"] = "None";
    return df

def choixValApprentissage(data):
    global compteurVarConnu
    newdata = copy.deepcopy(data);
    newdata["prediction"] = "None";
    for i in newdata.index:
        if(rd.random() < ratioValApprentissage):
            newdata.loc[i,"prediction"] = newdata.loc[i,str(len(newdata.columns) - 2)]
            compteurVarConnu = compteurVarConnu + 1
    return newdata;       
      
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
    

def remplissagePoint(data,k):
    newdata = copy.deepcopy(data);
    #pour tout les points qui n'ont pas de prédiction
    for i in newdata.loc[newdata['prediction'] == "None"].index:
            #On calcul distance euclidienne
            knnTab = knn(data,i);
            #On short au k choisi
            FinalChoixTab = [];  
            for j in range(0,k+1):
                FinalChoixTab.append(knnTab[j][1]);
            #On determine la classe
            FinalChoixTab = Counter(FinalChoixTab);
            #on l'ajoute a la tab data
            newdata.loc[i, 'prediction'] = list(FinalChoixTab.most_common(1)[0])[0];
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
        newdata = remplissagePoint(newdata,k);
                
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
    fd =open("result.txt",'r')
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
        
def predictionFinal(bestApprentissage):
    newdata = importDataWithDataFrame(pathExecutable);
    newdata2 = bestApprentissage[2];
    newdata2 = newdata2.append(newdata, ignore_index=True);
    fichier = open(pathWrite, "w")
    for i in newdata.index:
        print(i)
        #On calcul distance euclidienne
        knnTab = knn(newdata2,i);
        #On short au k choisi
        FinalChoixTab = [];
        for j in range(0,bestApprentissage[0] + 1):
                FinalChoixTab.append(knnTab[j][1]);   
        #On determine la classe
        FinalChoixTab = Counter(FinalChoixTab);
        print(Counter(FinalChoixTab));
        #on l'ajoute a la tab data
        newdata2.loc[i, 'prediction'] = list(FinalChoixTab.most_common(1)[0])[0];
        fichier.write(list(FinalChoixTab.most_common(1)[0])[0] + "\n")


    fichier.close()

    checklabel()
        
if __name__ == "__main__":
    #METHODE DATAFRAME
    début=t.process_time()
    print("Lancement de l'IA : OK ... ");
    #on recuperere les data , on remplie les colums d'apprentissage et on trouve le meilleur k
    data = choixValApprentissage(importDataWithDataFrame(pathApprentissage).append(importDataWithDataFrame(pathTest), ignore_index=True))
    

    bestApprentissage = Bestk(data)
    print("Valeur de k optimal trouvé : OK ... ");
    
    #on remet les bonne valeur de preTest pour enlever les erreurs
    bestApprentissage[2]['prediction'] = data[str(len(data.columns) - 2)]
    
    #on prédis les valeurs sans outpout
    predictionFinal(bestApprentissage)   
    
    fin=t.process_time()
    temps=fin-début
    print("temps d'execution du programme : ",temps, "s")
    
    print("Fin du programme");