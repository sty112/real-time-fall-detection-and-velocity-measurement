from ultralytics import YOLO

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    from ultralytics import YOLO

    model = YOLO('yolov8n.yaml')
    model = YOLO('yolov8n.pt')
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')
    model = YOLO(
        'D:\computer_vision_project\_runs\detect\_train3\weights\last.pt')

    results = model.train(data='D:\computer_vision_project\Fall Detection.v4-resized640_aug3x-accurate.yolov8\data.yaml',
                          epochs=200, imgsz=640, batch=-1, plots=True)
    # results = model.train(resume=True)
