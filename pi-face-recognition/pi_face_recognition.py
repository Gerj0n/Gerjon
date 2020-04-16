# importeer de benodigde bestanden
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# creër de argumenten ontleder en ontleed de argumenten
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
    help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# laad de bekende gezichten en de gezichtsomvang samen met OpenCV's Haar
# cascade voor de gezichts detectie
print("[INFO] coderingen laden + gezichts detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

# de video stream voorbereiden en de camerasensor laten opwarmen
print("[INFO] de videostream starten...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start de FPS teller
fps = FPS().start()

# loop over frames van de video stream
while True:
    # pak het frame van de video stream en verklein het
    # tot 500px (om het process te versnellen)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    
    # zet het invoer frame om van (1) BGR naar grijswaarden (voor gezichts
    # detectie) en (2) van BGR naar RGB (voor gezichtsherkenning)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detecteer gezichten in de grijswaarden
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # OpenCV stuurt coordinaten van het gezichtskader in (x, y, w, h) volgorde terug
    # maar ze moeten in (top, right, bottom, left) volgorde
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # bereken de gezichtsomvang voor elk gezichtskader
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over de gezichtskaders
    for encoding in encodings:
        # probeer elk gezicht in de invoer te matchen met een bekend gezicht
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        name = "Unknown"

        # controleer of we een match hebben gevonden
        if True in matches:
            # zoek de indexen van alle overeenkomende gezichten en houdt 
            # van elk gezicht bij hoevaak ze overeen kwamen
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over de overeenkomende indexen en houd een telling bij voor
            # elk herkend gezichtsvlak
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # bepaal het herkende gezicht met het hoogste getal
            # (opmerking: als er twee gezichten zijn met een evenhoog getal 
            # zal Python het eerste gezicht selecteren)
            name = max(counts, key=counts.get)
        
        # update de lijst met namen
        names.append(name)

    # loop over de herkende gezichten
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # zet de voorspelde gezichtsnaam op het gezicht
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)

    # toon de het frame op ons scherm 
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # als de 'q' toets ingedrukt wordt, stopt de loop
    if key == ord("q"):
        break

    # update de FPS teller
    fps.update()

# stop de timer en geef de FPS informatie weer
fps.stop()
print("[INFO] verlopen tijd: {:.2f}".format(fps.elapsed()))
print("[INFO] ca. FPS: {:.2f}".format(fps.fps()))

# om alles schoon te houden
cv2.destroyAllWindows()
vs.stop()
