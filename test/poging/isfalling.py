import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)


def is_fallen(pose_landmarks):
    """
    Beoordeel of de houding van een persoon wijst op een val.
    Dit wordt gedaan door de relatieve posities van belangrijke punten zoals het hoofd, de schouders
    en de heupen te analyseren.
    """
    # Haal de belangrijke punten op
    nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    # Bereken de geschatte afstand van het hoofd tot de grond, en de relatieve hoogte van de schouders en heupen
    head_y = nose.y
    shoulders_avg_y = (left_shoulder.y + right_shoulder.y) / 2
    hips_avg_y = (left_hip.y + right_hip.y) / 2

    # Als het hoofd dicht bij de grond is en de schouders en heupen op ongeveer hetzelfde niveau liggen,
    # wordt dit als een val beschouwd
    if head_y > shoulders_avg_y and abs(shoulders_avg_y - hips_avg_y) < 0.1:
        return True
    return False


def process_image(image_path):
    print("Laad afbeelding:", image_path)
    image = cv2.imread(image_path)
    if image is None:
        print("Fout: Afbeelding niet gevonden.")
        return

    print("Afbeelding succesvol geladen, verwerken...")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        print("Houdingslandmarks gedetecteerd, tekenen...")

        # Teken houdingslandmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Beoordeel of de persoon is gevallen
        if is_fallen(results.pose_landmarks):
            print("De persoon is gevallen.")
        else:
            print("De persoon is niet gevallen.")

        cv2.imshow('Pose Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Geen houdingslandmarks gedetecteerd.")


process_image('move.jpg')
