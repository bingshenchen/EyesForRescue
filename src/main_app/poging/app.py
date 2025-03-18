from src.main_app.poging import YOLODisplay

display = YOLODisplay()
display.process_video("C:/Users/Bingshen/Videos/AI Train/testVideos/fall.mp4")

display = YOLODisplay(enable_tracking=True, enable_classification=False)
display.process_video("C:/Users/Bingshen/Videos/AI Train/testVideos/fall.mp4")

display = YOLODisplay(enable_tracking=False, enable_classification=True)
display.process_video("C:/Users/Bingshen/Videos/AI Train/testVideos/fall.mp4")

display = YOLODisplay()
display.process_video(0)