import cv2
import torch
import numpy as np
import tensorflow as tf

# Kiểm tra xem GPU có khả dụng hay không
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Đường dẫn tới tệp .pt của mô hình YOLOv5
yolo_model_path = 'best.pt'

# Đường dẫn tới tệp .h5 của mô hình segmentation
segmentation_model_path = 'brick_crack_model.h5'

# Tải mô hình YOLOv5 từ tệp .pt và đặt chế độ sử dụng GPU
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path).to(device).eval()

# Tải mô hình segmentation từ tệp .h5
segmentation_model = tf.keras.models.load_model(segmentation_model_path)

# Đường dẫn tới ảnh
image_path = 'test_1.png'

# Đọc ảnh từ đường dẫn
image = cv2.imread(image_path)

# Chuyển đổi frame sang định dạng RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Phát hiện đối tượng bằng YOLOv5
results = yolo_model(image_rgb)

# Lấy kết quả phát hiện
detections = results.pandas().xyxy[0]

# Vẽ hình bao quanh đối tượng và hiển thị tên của chúng
for _, detection in detections.iterrows():
    xmin, ymin, xmax, ymax = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
    label = detection['name']
    confidence = detection['confidence']

    # Cắt phần của frame chứa vật được phát hiện
    object_image = image[int(ymin):int(ymax), int(xmin):int(xmax)]
    object_image_rgb = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)

    # Resize vật thể về kích thước mà mô hình segmentation yêu cầu
    input_width, input_height = 256, 256
    resized_object_image = cv2.resize(object_image_rgb, (input_width, input_height))

    # Chuẩn hóa và mở rộng kích thước của ảnh để phù hợp với đầu vào của mô hình segmentation
    segmentation_image = resized_object_image / 255.0
    segmentation_image = np.expand_dims(segmentation_image, axis=0)

    # Thực hiện dự đoán bằng mô hình segmentation
    prediction_mask = segmentation_model.predict(segmentation_image)

    # Chuyển đổi ảnh mask thành đen trắng
    prediction_mask_thresholded = (prediction_mask > 0.5).astype(np.float32)

    # Vẽ hình bao quanh đối tượng
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    # Hiển thị tên đối tượng và độ tin cậy
    label_text = f'{label}: {confidence:.2f}'
    cv2.putText(image, label_text, (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị vật thể được detect
    cv2.imshow('Detected Object', cv2.resize(object_image, (640, 480)))

    # Hiển thị kết quả segmentation nhị phân
    cv2.imshow(f'{label} Segmentation (Binary)', cv2.resize(prediction_mask_thresholded[0, :, :, 0], (640, 480)))

    cv2.waitKey(0)

# Hiển thị frame đã được đánh dấu
cv2.imshow('Annotated Image', cv2.resize(image, (640, 480)))
cv2.waitKey(0)
cv2.destroyAllWindows()
