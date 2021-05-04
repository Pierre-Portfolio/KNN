import sys

#code permettant de tester si un fichier de prédictions est au bon format.
#il prend en paramètre un fichier de labels prédits
#exemple> python checkLabels.py mesPredictions.txt

allLabels = ['classA','classB','classC','classD','classE']
#ce fichier s'attend à lire 3000 prédictions, une par ligne
#réduisez nbLines en période de test.
nbLines = 3000
fd =open(sys.argv[1],'r')
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


