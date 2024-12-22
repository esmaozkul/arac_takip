import cv2
import numpy as np
import tensorflow as tf

# Modeli yükle
model = tf.keras.models.load_model("car_detection_model_custom.keras")  # .h5 veya .keras olabilir

# Video dosyasını aç
video_path = "C:/Users/Esma/Desktop/arac_takip/arac.mp4"
cap = cv2.VideoCapture(video_path)

# İlk kareyi oku
ret, frame = cap.read()
if not ret:
    print("Video yüklenemedi!")
    cap.release()
    exit()

# Görüntüyü yeniden boyutlandır
resized_frame = cv2.resize(frame, (128, 128))

# Görüntüyü normalize et ve 4D tensor'a dönüştür
img_input = np.expand_dims(resized_frame, axis=0) / 255.0

# Model ile tahmin yap
prediction = model.predict(img_input)

# Araç varsa (prediction > 0.5) kırmızı renk ile işaretle
if prediction > 0.5:
    label = "Car "
    color = (0, 0, 255)  # Kırmızı
else:
    label = "No Car"
    color = (0, 255, 0)  # Yeşil

# Kullanıcıya araç seçme bölgesi göster
print("Araç tespit edildi. Araç takibi için başlangıç bölgesini seçin.")

# Araç bölgesini seçmek için ROI (Region of Interest) seçimi
roi = cv2.selectROI("Select Car", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Car")

# Eğer araç bölgesi seçildiyse, takip başlat
if roi != (0, 0, 0, 0):
    # Takip için CSRT algoritmasını kullan
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, roi)

    # Video çıktısı için yazıcı
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec
    out = cv2.VideoWriter("output_video_tracking.avi", fourcc, 30, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Takipçi algoritmasını bir adım ilerlet
        success, tracked_box = tracker.update(frame)

        # Takip başarıyla yapıldıysa
        if success:
            p1 = (int(tracked_box[0]), int(tracked_box[1]))
            p2 = (int(tracked_box[0] + tracked_box[2]), int(tracked_box[1] + tracked_box[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)  # Yeşil kutu çiz
            cv2.putText(frame, "Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Sonuçları ekrana yazdır
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # İşaretlenmiş kareyi ekranda göster
        cv2.imshow('Video', frame)
        
        # Aynı kareyi çıktıya yaz
        out.write(frame)

        # 'q' tuşuna basıldığında çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Takip tamamlandığında nesneleri serbest bırak
    cap.release()
    out.release()
    cv2.destroyAllWindows()
