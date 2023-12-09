import cv2
import numpy as np

# Pour les bricks (1: On continue a crée la brick, 0: On arrête de créer la brique car la balle l'a touchée)
b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15 = 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
# Niveau de difficulté (difficulte = 1 : Facile, difficulte = 2 : Moyen, difficulte = 3 : Difficile)
difficulte = 1
# Pour le score (Le nombre de brique touché)
score = 0
# Pour démarer la ball au debut de jeux (si s=1 la balle ne bouge pas si s=0 la ball bouge)
s = 1
# On crée une image vide
img = np.zeros((480,640,3),dtype='uint8')
# RGB de La couleur bleu clair
lightBlue = tuple(reversed((135,206,250)))
# On ajoutes une couleur de fond bleu clair à l'image
img[:] = lightBlue
# dx et dy représentent le pas de déplacement de la balle en pixel
dx,dy = 4,4
# La position du centre de la balle dans l'image
x,y = 300,200
# La position de la raquette (Un rectangle) : 
# (raquetteX,y1): position du point en haut à gauche du réctangle
# (raquetteX+100,y2): position du point en bas à droite du réctangle
raquetteX = 250
y1,y2 = 420,430
# Fonction pour changer la direction de la balle en cas de 
# Collision avec une brique en bas ou en haut ou à gauche ou à droite de la brique
# (x1,y1) : La position du point en haut à gauche du réctangle (La brique)
# (x2,y2) : La position du point en bas à droite du réctangle (La brique)
# x,y : La position du centre de la balle (Le cercle)
def changerDirectionBrick(x,y,x1,y1,x2,y2):
    global dx
    global dy
    global score
    b = False
    # En cas de Collision en bas avec la brique on change la direction de la ball dans l'axe y
    # On met b a true (La brique est touché)
    # On augmente le score
    if (y == y2+20) and  (x >=x1-20) and  (x <= x2+20):
        dy *= -1
        b = True 
        score = score + 1 
    # En cas de Collision en haut avec la brique on change la direction de la ball dans l'axe y
    # On met b a true (La brique est touché)
    # On augmente le score
    elif (y == y2-40) and (x >=x1-20) and (x <= x2+20):
        dy *= -1
        b = True
        score = score + 1 
    # En cas de Collision a gauche avec la brique on change la direction de la ball dans l'axe y
    # On met b a true (La brique est touché)
    # On augmente le score
    elif (x == x1-20) and  (y>=y1-20) and  (y <= y2+20):
        dy *= -1
        b = True
        score = score + 1 
    # En cas de Collision a droit avec la brique on change la direction de la ball dans l'axe x
    # On met b a true (La brique est touché)
    # On augmente le score
    elif (x == x1+60) and  (y>=y1-20) and  (y <= y2+20):
        dx *= -1
        b = True
        score = score + 1 
    # Si la direction de la balle est changée en frappant la brique, on retourne 0 Pour que la brique soit supprimée
    if(b == True):
        return(0)
    else:
        return(1)
    
# Menu de démarrage du jeu pour choisir une difficulté
# En appuyant sur la touche 1 on choisit la difficulté "facile"
# En appuyant sur la touche 2 on choisit la difficulté "Moyen" 
# En appuyant sur la touche 3 on choisit la difficulté "Difficile" et on augmente un peu la vitesse de la balle (dx,dy=5,5)
while True:
    cv2.putText(img=img, text='Jeux brick breaker', org=(135, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255),thickness=3)
    cv2.putText(img=img, text='Choisissez le niveau de difficulte :', org=(15, 120), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 255),thickness=2)
    cv2.putText(img=img, text='"1" pour Facile', org=(160, 200), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255),thickness=2)
    cv2.putText(img=img, text='"2" pour Moyen', org=(160, 300), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255),thickness=2)
    cv2.putText(img=img, text='"3" pour Difficile', org=(160, 400), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255),thickness=2)
    cv2.imshow("Brick Breaker",img)
    q = cv2.waitKey(0) & 0xFF
    if ord('1') == q:
        difficulte = 1
        break
    if ord('2') == q:
        difficulte = 2
        break
    if ord('3') == q:
        difficulte = 3
        dx,dy = 5,5
        break

# Track bar pour configurer la couleur à détecter manuellement (1ere amélioration)
# Par défaut la couleur est bleu : lower_blue = (90,60,0),  upper_blue = (121,255,255)
def nothing(x):
    pass
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 92, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 101, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 156, 255, nothing)
cv2.createTrackbar("US", "Tracking", 130, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

# Capturez la vidéo via la webcam
cap = cv2.VideoCapture(0)
# Boucle infinie du jeux et pour détecter la couleur bleu
while True:
    #####################################################################################################
    ############################### Code de détection de la couleur par la camera #######################
    #####################################################################################################
    # Capturez la vidéo frame par frame
    _, frame = cap.read()
    # Convertir la Frame de BGR en HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Prendre Lower H de la trackbar
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    # Prendre Lower S de la trackbar
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    # Prendre Lower V de la trackbar
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    # Prendre upper H de la trackbar
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    # Prendre upper S de la trackbar
    u_s = cv2.getTrackbarPos("US", "Tracking")
    # Prendre upper V de la trackbar
    u_v = cv2.getTrackbarPos("UV", "Tracking")
    # Lower range de la couleur a détecter (bleu par défaut)
    lower_color = np.array([l_h,l_s,l_v])
    # Upper range de la couleur a détecter (bleu par défaut)
    upper_color = np.array([u_h, u_s, u_v])
    # définir le masque
    mask = cv2.inRange(hsv, lower_color,upper_color)
    # Transformation morphologique : Dilatation, pour supprimer les bruits de l'image.
    kernal = np.ones((5, 5), "uint8")
    # Bitwise_and entre le frame et le masque est effectué pour détecter 
    # Spécifiquement cette couleur particulière et en écarter les autres.
    mask = cv2.dilate(mask, kernal)
    res_blue = cv2.bitwise_and(frame, frame, mask = mask)
    # Création du contour pour suivre la couleur bleu
    cnst, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # Créer un rectangle approximatif autour de la couleur bleu
    if(len(cnst)):
        for c in cnst:
            if cv2.contourArea(c) > 500:
                # La position de réctangle dans la frame
                raquetteX, raquetteY, w, h =  cv2.boundingRect(c)
                # Dessinez un rectangle autour de la couleur
                cv2.rectangle(frame, (raquetteX,raquetteY), (raquetteX+w,raquetteY+h), (0,0,255),3)
    #####################################################################################################
    ########################################## Code du jeux #############################################
    #####################################################################################################
    # Afficher la frame
    cv2.imshow("Frame",frame)
    # Afficher l'image
    cv2.imshow("Brick Breaker",img)
    k = cv2.waitKey(10) & 0xFF
    if k == ord('e'):
        break
    # Créer une image vide
    img = np.zeros((480,640,3),dtype='uint8') 
    color = tuple(reversed((135,206,250)))
    # On ajoutes une couleur de fond bleu clair à l'image
    img[:] = color
    # Afficher le score
    cv2.putText(img=img, text='Score : '+str(score), org=(10,460), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0, 0, 0),thickness=2)
    # Pour chaque difficulté
    # Gagnez si toutes les briques sont cassées
    if(difficulte == 1):
        if(score >= 5):
            cv2.putText(img=img, text='Tu as gagne', org=(100, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 0, 255),thickness=3)
            cv2.putText(img=img, text='Appuyez sur "E" pour quitter', org=(60, 350), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0),thickness=2)
            cv2.imshow("Brick Breaker",img)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('e'):
                break
    if(difficulte == 2):
        if(score >= 10):
            cv2.putText(img=img, text='Tu as gagne', org=(100, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 0, 255),thickness=3)
            cv2.putText(img=img, text='Appuyez sur "E" pour quitter', org=(60, 350), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0),thickness=2)
            cv2.imshow("Brick Breaker",img)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('e'):
                break
    if(difficulte == 3):
        if(score >= 15):
            cv2.putText(img=img, text='Tu as gagne', org=(100, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 0, 255),thickness=3)
            cv2.putText(img=img, text='Appuyez sur "E" pour quitter', org=(60, 350), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0),thickness=2)
            cv2.imshow("Brick Breaker",img)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('e'):
                break
    # Incrémenter la position de la balle 
    # On ajout a chaque itération un dx a x et dy a y au centre du cercle
    x = x+dx 
    y = y+dy
    # On crée un cercle (la balle) et on ajout la position x,y
    cv2.circle(img,(x,y),20,(0,0,255),-1)
    # On crée La raquette (un rectangle) 
    # On ajout la position raquetteX qui est la position du point en haut à gauche du réctangle de la couleur détecter dans l'axe x
    cv2.rectangle(img,(raquetteX,y1),(raquetteX+100,y2),(255,255,255),-1)
    # On crée les briques (des rectangles) pour chaque difficulté (2eme amélioration)
    # difficulte == 1 on crée 5 briques
    # difficulte == 2 on crée 10 briques
    # difficulte == 3 on crée 15 briques
    # Si b = 1 cela signifie que la balle n'a pas encore touché la brique donc elle continuera donc d'exister
    # Si b = 0 cela signifie que la balle a touché la brique donc on enlève la brick b
    if(difficulte == 1):
        if(b1 == 1):
            cv2.rectangle(img,(20,130),(90,160),(0,0,0),-1)
        if(b2 == 1):
            cv2.rectangle(img,(150,130),(220,160),(0,0,0),-1)
        if(b3 == 1):
            cv2.rectangle(img,(280,130),(350,160),(0,0,0),-1) 
        if(b4 == 1):
            cv2.rectangle(img,(410,130),(480,160),(0,0,0),-1)
        if(b5 == 1):
            cv2.rectangle(img,(540,130),(610,160),(0,0,0),-1)
    elif(difficulte == 2):
        if(b1 == 1):
            cv2.rectangle(img,(20,130),(90,160),(0,0,0),-1)
        if(b2 == 1):
            cv2.rectangle(img,(150,130),(220,160),(0,0,0),-1)
        if(b3 == 1):
            cv2.rectangle(img,(280,130),(350,160),(0,0,0),-1) 
        if(b4 == 1):
            cv2.rectangle(img,(410,130),(480,160),(0,0,0),-1)
        if(b5 == 1):
            cv2.rectangle(img,(540,130),(610,160),(0,0,0),-1)
        if(b6 == 1):
            cv2.rectangle(img,(20,70),(90,100),(0,0,0),-1)
        if(b7 == 1):
            cv2.rectangle(img,(150,70),(220,100),(0,0,0),-1)
        if(b8 == 1):
            cv2.rectangle(img,(280,70),(350,100),(0,0,0),-1) 
        if(b9 == 1):
            cv2.rectangle(img,(410,70),(480,100),(0,0,0),-1)
        if(b10 == 1):
            cv2.rectangle(img,(540,70),(610,100),(0,0,0),-1)
    elif(difficulte == 3):
        if(b1 == 1):
            cv2.rectangle(img,(20,130),(90,160),(0,0,0),-1)
        if(b2 == 1):
            cv2.rectangle(img,(150,130),(220,160),(0,0,0),-1)
        if(b3 == 1):
            cv2.rectangle(img,(280,130),(350,160),(0,0,0),-1) 
        if(b4 == 1):
            cv2.rectangle(img,(410,130),(480,160),(0,0,0),-1)
        if(b5 == 1):
            cv2.rectangle(img,(540,130),(610,160),(0,0,0),-1)
        if(b6 == 1):
            cv2.rectangle(img,(20,70),(90,100),(0,0,0),-1)
        if(b7 == 1):
            cv2.rectangle(img,(150,70),(220,100),(0,0,0),-1)
        if(b8 == 1):
            cv2.rectangle(img,(280,70),(350,100),(0,0,0),-1) 
        if(b9 == 1):
            cv2.rectangle(img,(410,70),(480,100),(0,0,0),-1)
        if(b10 == 1):
            cv2.rectangle(img,(540,70),(610,100),(0,0,0),-1)
        if(b11 == 1):
            cv2.rectangle(img,(20,10),(90,40),(0,0,0),-1)
        if(b12 == 1):
            cv2.rectangle(img,(150,10),(220,40),(0,0,0),-1)
        if(b13 == 1):
            cv2.rectangle(img,(280,10),(350,40),(0,0,0),-1) 
        if(b14 == 1):
            cv2.rectangle(img,(410,10),(480,40),(0,0,0),-1)
        if(b15 == 1):
            cv2.rectangle(img,(540,10),(610,40),(0,0,0),-1)

    # Changer le signe de l'incrément en cas de collision avec la limite ou la raquette
    # Si il y'a une collision avec la raquette on change la direction de la ball dans l'axe y
    # Sinon si le cercle atteint la limite supérieure du l'image on change la direction de la ball dans l'axe y
    if (y == y1-20) & (x >=raquetteX) & (x <= raquetteX+100):
        dy *= -1
    elif y<=0:
        dy *= -1
    # Si le cercle atteint la limite droit du l'image on change la direction de la ball dans l'axe x
    # Sinon si le cercle atteint la limite gauche du l'image on change la direction de la ball dans l'axe y
    if x >=640:
        dx *= -1
    elif x<=0:
        dx *= -1
    # Si le cercle atteint la limite inférieure du l'image on perd ("Game Over")
    if y>=480:
        cv2.putText(img=img, text='Game Over', org=(130, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 0, 255),thickness=3)
        cv2.putText(img=img, text='Appuyez sur "E" pour quitter', org=(60, 350), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0),thickness=2)
    # Changer la direction de la balle en cas de collision avec la brick pour chaque difficulté
    """
    Si la fonction changerDirectionBrick est true sela signifie que : 
    - La direction de la balle a changé on touchant la brique donc b = 0 et on peut enlever la brique
    Si la fonction changerDirectionBrick est false sela signifie que : 
    - La direction de la balle n'a pas changé en touchant la brique donc b restera 1 et on peut pas enlever la brique
    """
    if(difficulte == 1):        
        if(b1==1):
            b1 = changerDirectionBrick(x,y,20,130,90,160)
        if(b2==1):
            b2 = changerDirectionBrick(x,y,150,130,220,160)
        if(b3==1):
            b3 = changerDirectionBrick(x,y,280,130,350,160)
        if(b4==1):
            b4 = changerDirectionBrick(x,y,410,130,480,160)
        if(b5==1):
            b5 = changerDirectionBrick(x,y,540,130,610,160)
    elif(difficulte == 2):
        if(b1==1):
            b1 = changerDirectionBrick(x,y,20,130,90,160)
        if(b2==1):
            b2 = changerDirectionBrick(x,y,150,130,220,160)
        if(b3==1):
            b3 = changerDirectionBrick(x,y,280,130,350,160)
        if(b4==1):
            b4 = changerDirectionBrick(x,y,410,130,480,160)
        if(b5==1):
            b5 = changerDirectionBrick(x,y,540,70,610,160)
        if(b6==1):
            b6 = changerDirectionBrick(x,y,20,70,90,100)
        if(b7==1):
            b7 = changerDirectionBrick(x,y,150,70,220,100)
        if(b8==1):
            b8 = changerDirectionBrick(x,y,280,70,350,100)
        if(b9==1):
            b9 = changerDirectionBrick(x,y,410,70,480,100)
        if(b10==1):
            b10 = changerDirectionBrick(x,y,540,70,610,100)
    elif(difficulte == 3):
        if(b1==1):
            b1 = changerDirectionBrick(x,y,20,130,90,160)
        if(b2==1):
            b2 = changerDirectionBrick(x,y,150,130,220,160)
        if(b3==1):
            b3 = changerDirectionBrick(x,y,280,130,350,160)
        if(b4==1):
            b4 = changerDirectionBrick(x,y,410,130,480,160)
        if(b5==1):
            b5 = changerDirectionBrick(x,y,540,70,610,160)
        if(b6==1):
            b6 = changerDirectionBrick(x,y,20,70,90,100)
        if(b7==1):
            b7 = changerDirectionBrick(x,y,150,70,220,100)
        if(b8==1):
            b8 = changerDirectionBrick(x,y,280,70,350,100)
        if(b9==1):
            b9 = changerDirectionBrick(x,y,410,70,480,100)
        if(b10==1):
            b10 = changerDirectionBrick(x,y,540,70,610,100)
        if(b11==1):
            b11 = changerDirectionBrick(x,y,20,10,90,40)
        if(b12==1):
            b12 = changerDirectionBrick(x,y,150,10,220,40)
        if(b13==1):
            b13 = changerDirectionBrick(x,y,280,10,350,40)
        if(b14==1):
            b14 = changerDirectionBrick(x,y,410,10,480,40)
        if(b15==1):
            b15 = changerDirectionBrick(x,y,540,10,610,40)

    # Pour stopper la balle au début du jeu et la lancer avec un input
    if(s==1):
        s = 0
        while True:
            cv2.putText(img=img, text='Appuyez sur "s" pour que la balle bouge', org=(60,300), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0, 0, 0),thickness=1)
            # Afficher l'image de début de jeux
            cv2.imshow("Brick Breaker",img)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('s'):
                break
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
cap.release()
cv2.destroyAllWindows()