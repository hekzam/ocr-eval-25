#Ici le choix est dans le but d'être propre pour un haut taux de réussite.
exemplesMistDict = {
    0 : ["resources\mnist\image_1.png", "resources\mnist\image_1040.png", "resources\mnist\image_1084.png"],
    1 : ["resources\mnist\image_1031.png", "resources\mnist\image_1044.png", "resources\mnist\image_484.png"],
    2 : ["resources\mnist\image_650.png", "resources\mnist\image_1301.png", "resources\mnist\image_1584.png"],
    3 : ["resources\mnist\image_1559.png", "resources\mnist\image_74.png", "resources\mnist\image_341.png"],
    4 : ["resources\mnist\image_1050.png", "resources\mnist\image_9.png", "resources\mnist\image_64.png"],
    5 : ["resources\mnist\image_537.png", "resources\mnist\image_879.png", "resources\mnist\image_964.png"],
    6 : ["resources\mnist\image_816.png", "resources\mnist\image_630.png", "resources\mnist\image_430.png"],
    7 : ["resources\mnist\image_84.png", "resources\mnist\image_71.png", "resources\mnist\image_1422.png"],
    8 : ["resources\mnist\image_144.png", "resources\mnist\image_202.png", "resources\mnist\image_146.png"],
    9 : ["resources\mnist\image_621.png", "resources\mnist\image_707.png", "resources\mnist\image_772.png"]
}

def comparaisonRslt(img, numObtenu):
    return exemplesMistDict[numObtenu].count(img) != 0

#TODO: une fonction pour tester chaque moteurs/algo