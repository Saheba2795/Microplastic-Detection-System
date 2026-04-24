from ultralytics import YOLO


model = YOLO("D:/microplastic_project/microplastic_project/runs/detect/microplastic_train3/weights/best.pt")


results = model.predict(source="D:/microplastic_project/microplastic_project/test_image.jpg", conf=0.25)
print("Executed")

results[0].show()   
print("Executed Show")