# **AI Applications**
**Groep 13**  
**October 6, 2024**  
**Author: Marthe**

## **1. Opdracht 1**

Voor de eerste opdracht werd er van ons verwacht dat we nadenken over mogelijke strategieën om het project van AI Applications aan te pakken. Ook hebben we de GitHub repository opgezet waarin we het komende semester zullen werken: [GitHub Repository](https://github.com/JorritR/AI-Applications).

### **1.1 Probleemomschrijving**

Het probleem dat we willen aanpakken, is de responstijd van hulpdiensten. Momenteel moet een omstander een noodsituatie inschatten en vervolgens de 112 bellen, wat vaak tijd kost omdat de omstanders meestal niet getraind zijn in noodsituaties. Dit kan de reactie vertragen.

In deze opdracht kijken we naar manieren waarop AI gebruikt kan worden om aan de hand van camerabeelden automatisch noodsituaties op straat te detecteren. Dit systeem zou direct de hulpdiensten kunnen verwittigen wanneer er een probleem is, waardoor de langere reactietijd van omstanders wordt omzeild. Bovendien kan de exacte locatie direct worden doorgegeven.

Dit systeem fungeert als een back-up voor de hulp van omstanders, omdat snel handelen belangrijk blijft. Het systeem moet echter ook door een 112-operator worden gecontroleerd om te beoordelen of het daadwerkelijk om een noodsituatie gaat (bijvoorbeeld bij een pop die valt en onterecht als noodgeval wordt gezien). Om de 112-operatoren niet te overbelasten, proberen we het aantal false positives te beperken.

### **1.2 Strategie 1: Human Pose Estimation**

Onze eerste strategie is het gebruik van Human Pose Estimation om bewegingen van mensen op straat te analyseren. We definiëren een aantal scenario’s, bijvoorbeeld een persoon die voor een bepaalde tijd horizontaal ligt, wat kan wijzen op een val. Omdat Human Pose Estimation maar één persoon tegelijk kan volgen, onderzoeken we ook Multi-Person Pose Estimation, aangezien meerdere personen in beeld kunnen zijn.

Voordelen:
- Eenvoudige logica om te implementeren.
- Kan specifiekere situaties detecteren door zelfgekozen regels.

Nadelen:
- Vereist vooraf vastgestelde patronen, wat sommige problemen kan missen.

### **1.3 Strategie 2: Anomalie Detectie**

Onze tweede strategie richt zich op het detecteren van afwijkingen van het ‘normale’ gedrag op straat. Dit systeem is gevoeliger en kan situaties zoals gevechten detecteren. Het nadeel hiervan is dat ook onschuldige acties zoals het oppakken van een telefoon als anomalie worden gemarkeerd.

Voordelen:
- Kan bredere soorten incidenten detecteren, zoals gevechten.

Nadelen:
- Hogere kans op false positives bij normale activiteiten.

### **1.4 Strategie 3: Gecombineerde Strategie**

In de derde strategie combineren we beide benaderingen. Eerst detecteren we anomalieën (zoals in Strategie 2) en vervolgens controleren we de gemarkeerde incidenten met behulp van posities (zoals in Strategie 1). Dit systeem biedt dubbele controle, maar vergt veel rekenkracht en kan traag zijn doordat meerdere AI-processen achter elkaar moeten draaien.

Voordelen:
- Nauwkeuriger door dubbele controle.

Nadelen:
- Hoge rekenkosten, mogelijk verlies van flexibiliteit door strikte regels.

### **1.5 Literatuur en Bestaande Systemen**

Voor ons project werken we met Human Pose Estimation en Multi-Person Pose Estimation. Hieronder enkele bronnen die ons hierbij ondersteunen:
- [GeeksForGeeks - Pose Estimation](https://www.geeksforgeeks.org/python-opencv-pose-estimation/)
- [HackersRealm - Realtime Human Pose Estimation](https://www.hackersrealm.net/post/realtime-human-pose-estimation-using-python)
- [Cloudzilla - Multi-Person Pose Estimation](https://www.cloudzilla.ai/dev-education/multi-person-pose-estimator-with-python/)
- [Medium - Multi-Person Pose Estimation](https://shawntng.medium.com/multi-person-pose-estimation-with-mediapipe-52e6a60839dd)

Een belangrijke toepassing van pose estimation is binnen beveiligingssystemen, die grote groepen mensen monitoren en afwijkend gedrag detecteren. Deze toepassingen liggen dicht bij wat wij willen doen. Documentatie hierover kan gevonden worden in de paper: *Where are we with Human Pose Estimation in Real-World Surveillance?*

### **1.6 Evaluatie**

We zullen een deel van onze data aan de kant houden voor de evaluatie van ons model. Vanwege mogelijke problemen met GDPR kunnen we geen willekeurige datasets gebruiken en zullen we alle data zelf moeten genereren. Dit betekent dat we zelf veel data moeten produceren.

Een uitdaging bij Strategie 3 is dat het systeem tijdens productie blijft leren, wat kan betekenen dat de testdata snel ontoereikend wordt. Dit kan deels worden opgelost door feedback van 112-operatoren te verzamelen en periodiek te gebruiken om het systeem verder te trainen.

Daarnaast moeten we rekening houden met wat wel en niet mogelijk is volgens de GDPR-regels.

