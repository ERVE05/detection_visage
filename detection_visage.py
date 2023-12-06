
import cv2
import streamlit as st

# Chargement du classificateur de visages Haar Cascade
face_cascade = cv2.CascadeClassifier('C:/Users/USER/Downloads/haarcascade_frontalface_default.xml')
# Dictionnaire pour stocker les noms et les images des individus connus
known_faces = {}

def detect_faces(min_neighbors, scale_factor, rectangle_color):
    # Initialisation de la webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Lecture des frames de la webcam
        ret, frame = cap.read()
        # Conversion des frames en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Détection des visages en utilisant le classificateur de visages Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        # Dessin des rectangles autour des visages détectés
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
        # Affichage des frames
        cv2.imshow("Détection de visages avec l'algorithme de Viola-Jones", frame)
        # Sortie de la boucle lorsque 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Libération de la webcam et fermeture de toutes les fenêtres
    cap.release()
    cv2.destroyAllWindows()

def capture_and_label():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("La webcam n'est pas disponible. Assurez-vous que la webcam est correctement connectée.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Impossible de capturer une image depuis la webcam.")
            break

        cv2.imshow('Capture et labélisation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # Demander le nom de l'individu
    name = st.text_input("HARVER")

    # Enregistrer l'image capturée avec le nom de l'individu dans le dictionnaire
    known_faces[name] = frame

def app():
    st.title("Détection de visages avec l'algorithme de Viola-Jones")

    # Ajout des instructions à l'interface
    st.write("Appuyez sur le bouton ci-dessous pour commencer la détection de visages depuis votre webcam.")
    st.write("Ajustez les paramètres ci-dessous pour personnaliser la détection de visages.")

    # Ajout de fonctionnalités pour choisir la couleur des rectangles
    rectangle_color = st.color_picker("Choisissez la couleur du rectangle", "#00ff00")
    rectangle_color = tuple(int(rectangle_color[i:i+2], 16) for i in (1, 3, 5))


    # Ajout de fonctionnalités pour ajuster le paramètre minNeighbors
    min_neighbors = st.slider("Ajuster minNeighbors", 1, 10, 5)

    # Ajout de fonctionnalités pour ajuster le paramètre scaleFactor
    scale_factor = st.slider("Ajuster scaleFactor", 1.1, 2.0, 1.3, step=0.1)

    # Ajout du bouton pour démarrer la détection de visages
    if st.button("Détecter les visages"):
        # Appel de la fonction detect_faces avec les paramètres ajustés
        detect_faces(min_neighbors, scale_factor, rectangle_color)

    # Ajout de fonctionnalités pour enregistrer les images avec les visages détectés
    if st.button("Capturer et labéliser"):
        # Ajout d'instructions pour guider l'utilisateur sur la capture et l'étiquetage
        st.write("Appuyez sur le bouton 'Capturer et labéliser' pour capturer une image et attribuer un nom.")
        capture_and_label()

if __name__ == "__main__":
    app()
