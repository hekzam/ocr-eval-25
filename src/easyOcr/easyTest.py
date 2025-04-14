import easyocr
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import TestResultat.setup as su

reader = easyocr.Reader({'en'}, gpu=False, model_storage_directory = 'src\easyOcr\stockageModele')

"""
Utilisation de l'algo de base de easyocr, si on veut utiliser easy ocr il faudra le modifier car il n'est pas fait pour
 le manuscrit et prend en compte l'alphabet (ce qui n'est pas ce que l'on cherche)
pour personnaliser :
    en utilisant sa propre structure avec un .predict(): 
    reader = easyocr.Reader({'en'}, user_network_directory = CustomModel)
ou
    en utilisant un entrainement de la structure personnalisé, doit être entrainé pour easyOcr obligé:
    reader = easyocr.Reader({'en'}, user_network_directory='/path/to/user_network_directory')
"""

for num in su.exemplesMistDict:
    for img in su.exemplesMistDict[num]:
        rslt = reader.readtext(img)
        if rslt != []:
            for i in range(len(rslt)):
                print(num, ": " ,su.comparaisonRslt(img, int(rslt[i][1])), " // rslt obt: ", rslt[i][1], " prob: ", rslt[i][2])
        else:
            print(num, " Resultat non détecté")


