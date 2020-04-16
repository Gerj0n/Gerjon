# importeer de benodigde bestanden
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# creër de opdracht ontleder en ontleed de opdrachten
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
    help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
    help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# pak de paden naar de invoer afbeeldingen in onze database
print("[INFO] gezichten instellen...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialiseer de lijst met bekende coderingen en bekende namen
knownEncodings = []
knownNames = []

# loop over de afbeelding paden
for (i, imagePath) in enumerate(imagePaths):
    # haal de naam van de persoon uit het afbeeldingspad
    print("[INFO] afbeelding verwerken {}/{}".format(i + 1,
        len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # laad de invoer afbeelding en zet hem om van RGB (OpenCV ordening)
    # naar dlib ordening (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detecteer de (x, y)-coordinaten van het gezichtskader
    # overeenkomend met elk gezicht in de invoer afbeelding
    boxes = face_recognition.face_locations(rgb,
        model=args["detection_method"])

    # bereken het kader voor het gezicht
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over de coderingen
    for encoding in encodings:
        # voeg elke codering + naam toe aan onze reeks van bekende namen
        # coderingen
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump de gezichts coderingen + namen naar het bestand
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
