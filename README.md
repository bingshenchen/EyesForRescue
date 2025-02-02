# AI-Applications

## Project Informatie
**AJ: 2024-2025**   
**Docent: Jan D'Espallier**  
**Versie: 1.6**   
**Laatste update: 08-12-2024**   
**Groep: Groep13**   
**Groepsleden: Bingshen Chen, Marthe D'Hooghe, Jorrit Ruelens, Jarne Theys**

## Probleemomschrijving
Wanneer iemand een noodgeval heeft op straat, is de responstijd waarin hulpdiensten kunnen reageren onderhevig aan een aantal stappen waardoor men heen moet om hulp te krijgen. Eerst moet de noodcentrale gebeld worden door voorbijgangers. Die moeten proberen om de belangrijke informatie zo goed mogelijk door te geven, maar doordat het in dit geval meestal gaat over mensen zonder medische achtergrond die vaak ook al onder de indruk zijn van wat er gebeurt is, kan deze stap al gemakkelijk 3 minuten duren. Daarna moeten de hulpdiensten ook nog effectief ter plaatse komen, wat opnieuw een 10 minuten kan duren. En gezien in noodsituaties elke minuut telt, willen wij deze responstijd minimaliseren.    
Met ons project willen we de responstijd verkleinen door die eerste stap te automatiseren. Dat doen we door via camerabeelden op straat te detecteren of mensen hulp nodig hebben. Zo kunnen de hulpdiensten onmiddelijk verwittigd worden en krijgen ze een exacte locatie en een beeld waarin mensen met medische achtergrond ook zelf kunnen inschatten wat de situatie is.

## General Architecture
Gezien op high level, onze applicatie werkt met 3 stappen: eerst detecteren we mensen die op straat rondlopen, daarna kijken we of deze mensen vallen of niet. Van de mensen waarbij we een val gedetecteerd hebben en die blijven liggen, kijken we vervolgens of ze effectief in nood zijn en bijvoorbeeld geen boek aan het lezen zijn. Met deze stappen kunnen we een alert aanmaken waarna we in een later stadium de hulpdiensten automatisch verwittigen.

### Detectie mensen op straat
We gebruiken YOLO om mensen op straat te detecteren. 

### Val-detectie
Wanneer de bounding box die het YOLO-model teruggeeft, plots van proporties verandert en van een staande rechthoek naar een liggende gaat, kunnen we uitgaan van een val. Daarna blijven we deze persoon tracken om te kijken of hij/zij onbeweeglijk blijft liggen. Wanneer dat het geval is voor een aantal seconden, kunnen we verdergaan naar de volgende stap.

### Classificatie noodgevallen
Wanneer we effectief een val hebben gedetecteerd, gaan we nu een classifier op de beelden loslaten om te kijken of het hier gaat over een noodgeval zoals een hartaanval of een ongevaarlijk voorval waarin iemand bijvoorbeeld een boek leest.

## Resultaten
Onze huidige beste resultaten zijn opgedeeld in de performantie van de valdetectie en de classifier.

### Valdetectie
Precision: 100%    
Recall:    73,08%     
Accuracy:  86,67%     
![image info](./docs/performance%20report/Performance.png)
### Classifier
Precision: 95%    
Recall:    95%    
F1-score:  95%    
![image info](./docs/performance%20report/RF.png)
## Scripts Runnen
### Installatie
1. **Clone de repository**:
    ```bash
    git clone https://github.com/JorritR/AI-Applications.git
    cd AI-Applications
    ```
2. **Installeer de benodigde Python-pakketten**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Configureren van .env bestand**:
   Maak een `.env` bestand gebaseerd op `.env.example` en vul de juiste paden in voor de YOLO modellen, video’s en datasets.
   
### Runnen
#### Runnen met GUI
Om de GUI te starten voor een interactieve ervaring, gebruik:
   ```bash
   python src/app.py
   ```
Het `app.py` script dient als de hoofdtoepassing met een grafische gebruikersinterface (GUI) voor het uitvoeren van verschillende functionaliteiten van het project. Dit script maakt gebruik van het `tkinter`-framework om een gebruiksvriendelijke interface te bieden voor het beheren van de volgende acties:

- **Frames extraheren**: Stelt de gebruiker in staat om frames uit videobestanden te extraheren voor verdere analyse.
- **Labels genereren**: Maakt gebruik van een YOLO-model om automatisch voorspellingen te genereren die de gebruiker handmatig kan bevestigen en corrigeren. Dit proces versnelt het labelen van datasets voor verdere modeltraining. Na het genereren van labels kunnen gebruikers de optie selecteren om afbeeldingen te croppen op basis van de YOLO-labels.
- **Model trainen**: Start de trainingsfase van het YOLO-model direct vanuit de GUI.
- **Rapport analyseren**: Toont een GUI-interface voor het analyseren van het prestatieverslag van het model.
- **Video openen voor valdetectie**: Analyseert een geselecteerd videobestand om valincidenten te detecteren.
- **Camera starten voor real-time analyse**: Maakt het mogelijk om de camera in real-time te gebruiken voor valdetectie.
- **Alert Test**: Voert een test uit om alarmfunctionaliteit en detectieprestaties te valideren.

**Interface Functies**:

- **Menubalkopties**:
    - **Detection**: Kies voor 'Fall Detection (Video)' om een videobestand te analyseren, of 'Fall Detection (Camera)' om een real-time camerafeed te gebruiken.
    - **Data Preparation**: Gebruik 'Extract Frames' om frames uit een video te extraheren, of 'Auto Generate Labels' om labels te genereren voor datasets.
    - **Training**: Gebruik 'Model Training' om het YOLO-model te trainen met de gespecificeerde parameters en datasets.
    - **Analysis**: Selecteer 'Model Performance Report' om prestatieverslagen te analyseren of 'Alert Test' om het waarschuwingssysteem te testen.

#### Runnen zonder GUI
Om de applicatie zonder de grafische interface te starten, voer het volgende commando uit:
   ```bash
   python src/main.py
   ```
Het `main.py` script is het belangrijkste bestand voor het uitvoeren van de trainings- en testprocedures voor het model. Het script voert verschillende taken uit, waaronder het:
- **Splitsen van dataset**: Verdeelt de gegevens in trainings-, validatie- en testsets.
- **Training van het model**: Start de trainingsfase van het YOLO-model indien ingesteld in de omgevingsvariabelen.
- **Evaluatie van submappen**: Voert een evaluatie uit voor elke submap in de testset en genereert prestatiemetrics, zoals precisie, recall, mAP50 en mAP50-95.
- **Rapportage van resultaten**: Slaat de resultaten op in een Excel-bestand om gedetailleerde metingen per onderwerp en submap weer te geven.

## Projectstructuur

<details>
<summary>Hier is een overzicht van de projectstructuur:</summary>

```plaintext
AI-Applications/
    .env
    .env.example
    README.md
    requirements.txt
    analysis/
        componentDiagram.png
        erdDiagram.png
        flowchart.png
        sequenceDiagram.png
        stateDiagram.png
        userCaseDiagram.png
    docs/
        performance report/
            BestModel1.3/
                Performance report.md
                Results/
                    confusion_matrix_normalized.png
                    F1_curve.png
                    labels.jpg
                    PR_curve.png
                    P_curve.png
                    results.png
                    R_curve.png
                    train_batch0.jpg
                    val_batch0_pred.jpg
        report/
            WS1.md
            WS2.md
            WS3.md
            WS4.md
    src/
        app.py
        main.py
        minio-test.py
        minio_data.py
        tracked_fall_d.mp4
        detection/
            fall_detection.py
            fall_detection_visualization.py
        gui/
            gui_analyze.py
            gui_extract_frame.py
            gui_generate_labels.py
            gui_menubar.py
            gui_open_video_file.py
            gui_start_camera.py
            gui_train_model.py
        models/
            best1.2.pt
            best1.3.pt
            train_model.py
            train_model_met_minio.py
            yolov11n.pt
        utils/
            calculate_danger.py
            evaluate_model_on_video.py
            extract_frames.py
            generate_labels_b.py
            generate_labels_g.py
            generate_report.py
            generate_video_raport.py
            image_detection.py
            video_processing.py
```
</details> 

## Open issues

[GitHub Issues](https://github.com/JorritR/AI-Applications/issues)
