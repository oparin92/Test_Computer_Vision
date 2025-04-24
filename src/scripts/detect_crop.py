from src.scripts.utils import create_video
import cv2
import numpy as np
from ultralytics import YOLO
import os

def process_video(input_path: str):
    """Обрабатывает видео."""
  
    frames = 'output/frames' # Путь к папке с кадрами
    margin = 20 # Отступ
    frame_num = 100 # Счетчик для наименования кадров

    model = YOLO("yolo11x.pt")  # загрузка предобученной модели

    cap = cv2.VideoCapture('input/input_video.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int((cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int((cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while cap.isOpened():
        succes, frame = cap.read()

        if not succes:
            break

        results = model(frame, conf=0.2, imgsz=1024, max_det=1)
        boxes = results[0].boxes.xywh.numpy()
        for box in boxes:
            x, y, w, h = box
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

            # Добавление отступа
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(width, x2 + margin)
            y2 = min(height, y2 + margin)

            # Обрезка кадра
            cropped = frame[y1:y2, x1:x2]


        if not os.path.exists(frames):  # Проверка существования директории
            os.makedirs(frames)  # Создание директории, если она не существует
        cv2.imwrite(os.path.join(frames, f"{frame_num}.jpg"), cropped)  # Сохранение кадра
        frame_num +=1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    # Получение списка файлов в папке
    files = sorted(os.listdir(frames))

    # Чтение первого кадра для определения размера
    first_frame_path = os.path.join(frames, files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    # Создание объекта VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/processed_video.mp4', fourcc, fps, (width, height))

    # Проход по всем кадрам и добавление их в видео
    for file in files:
        frame_path = os.path.join(frames, file)
        frame = cv2.imread(frame_path)
         
        # Изменение размера кадра с сохранением пропорций
        frame_height, frame_width, _ = frame.shape
        aspect_ratio = frame_width / frame_height
        new_width = width
        new_height = int(new_width / aspect_ratio)
        if new_height > height:
            new_height = height
            new_width = int(new_height * aspect_ratio)
        frame = cv2.resize(frame, (new_width, new_height))
         
        # Добавление черных полос для заполнения оставшегося пространства
        result_frame = np.zeros((height, width, 3), dtype=np.uint8)
        result_frame[(height - new_height) // 2:(height - new_height) // 2 + new_height,
                        (width - new_width) // 2:(width - new_width) // 2 + new_width] = frame
         
        out.write(result_frame) 


    cap.release()
    out.release()

    output_video = "output/processed_video.mp4"
    create_video(input_path, output_video)
    print("✅ Обработка завершена.")
