import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Veri yollarını belirle
base_path = "C:/Users/Esma/Desktop/arac_takip/img1/"
arac_var_path = os.path.join(base_path, "arac_var")
arac_yok_path = os.path.join(base_path, "arac_yok")

# Görselleri ve etiketleri yükle
X, y = [], []

# Araç resimlerini ekle
for img_name in os.listdir(arac_var_path):
    img_path = os.path.join(arac_var_path, img_name)
    img = cv2.imread(img_path)
    if img is not None:  # Dosya gerçekten bir resim mi?
        img = cv2.resize(img, (128, 128))  # Yeniden boyutlandır
        X.append(img)
        y.append(1)  # Araç var

# Araç olmayan resimleri ekle
for img_name in os.listdir(arac_yok_path):
    img_path = os.path.join(arac_yok_path, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (128, 128))
        X.append(img)
        y.append(0)  # Araç yok

# Listeyi numpy dizisine çevir ve normalizasyon yap
X = np.array(X, dtype=np.float32) / 255.0  # Görselleri normalleştir (0-1 aralığı)
y = np.array(y, dtype=np.float32)  # Etiketleri numpy dizisine çevir

# Veriyi eğitim ve doğrulama setlerine böl
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli tanımla
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# Modeli derle
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Erken durdurma tanımla
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Modeli eğit
model.fit(X_train, y_train, 
          epochs=20, 
          batch_size=32, 
          validation_data=(X_val, y_val), 
          callbacks=[early_stopping])

# Modeli kaydet
model.save("car_detection_model_custom.keras")
