from ultralytics import YOLO

# Load the teacher model
teacher_model = YOLO(
    "../YOLO-Training/trained/tr_yolov11/tr_n/yolov11n/weights/best.pt")

student_model = YOLO("../YOLO-Training/raw_nnet/yolov11/yolo11n.pt")

student_model.train(
    data="../YOLO-Training/datasets/ds_yolov11/ds_851005/data.yaml",
    teacher=teacher_model.model,  # None if you don't wanna use knowledge distillation
    distillation_loss="cwd",
    epochs=100,
    batch=16,
    workers=0,
    exist_ok=True,
)
