import sys
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QSpinBox, QDoubleSpinBox,
    QFileDialog, QTableWidget, QTableWidgetItem
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikasi Kematangan Tomat 🍅")
        self.setGeometry(100, 100, 1200, 700)
        self.initUI()

        self.original_image = None
        self.rgb_image = None
        self.gray_image = None
        self.binary_image = None

    def initUI(self):
        self.setStyleSheet("""
            QWidget { background-color: #f0fdf4; font-family: Arial; }
            QLabel { font-size: 14px; color: #333; }
            QPushButton {
                background-color: #ef4444; color: white; padding: 6px 12px;
                border-radius: 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #dc2626; }
            QSlider::handle:horizontal {
                background: #16a34a; border: 1px solid #999999;
                width: 10px; margin: -2px 0; border-radius: 5px;
            }
        """)

        titleLabel = QLabel("🍅 APLIKASI PERBAIKAN KUALITAS CITRA\nKLASIFIKASI TINGKAT KEMATANGAN TOMAT 🍅")
        titleLabel.setAlignment(Qt.AlignCenter)
        titleLabel.setStyleSheet("font-size: 20px; font-weight: bold; color: #166534;")

        controlPanel = QVBoxLayout()

        for label, attr in [("Brightness", "sliderBrightness"), ("Contrast", "sliderContrast"), ("Saturation", "sliderSaturation")]:
            lbl = QLabel(label)
            controlPanel.addWidget(lbl)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.valueChanged.connect(self.updateImage)
            setattr(self, attr, slider)
            controlPanel.addWidget(slider)

        self.spinRotation = QSpinBox()
        self.spinRotation.setRange(0, 360)
        self.spinRotation.valueChanged.connect(self.updateImage)
        controlPanel.addWidget(QLabel("Rotation"))
        controlPanel.addWidget(self.spinRotation)

        self.spinScale = QDoubleSpinBox()
        self.spinScale.setRange(0.1, 5.0)
        self.spinScale.setSingleStep(0.1)
        self.spinScale.setValue(1.0)
        self.spinScale.valueChanged.connect(self.updateImage)
        controlPanel.addWidget(QLabel("Scale"))
        controlPanel.addWidget(self.spinScale)

        self.btnLoad = QPushButton("📂 Load Image")
        self.btnLoad.clicked.connect(self.loadImage)
        controlPanel.addWidget(self.btnLoad)

        self.btnReset = QPushButton("🔄 Reset")
        self.btnReset.clicked.connect(self.resetControls)
        controlPanel.addWidget(self.btnReset)

        self.btnExport = QPushButton("📤 Export Excel")
        self.btnExport.clicked.connect(self.exportToExcel)
        controlPanel.addWidget(self.btnExport)

        self.btnInputExcel = QPushButton("📋 Input to Table Excel")
        self.btnInputExcel.clicked.connect(self.inputToTable)
        controlPanel.addWidget(self.btnInputExcel)

        self.btnEkstraksi = QPushButton("🔍 Ekstraksi Ciri")
        self.btnEkstraksi.clicked.connect(self.ekstraksiCiri)
        controlPanel.addWidget(self.btnEkstraksi)

        self.btnKlasifikasi = QPushButton("🧪 Klasifikasi Kematangan")
        self.btnKlasifikasi.clicked.connect(self.klasifikasiKematangan)
        controlPanel.addWidget(self.btnKlasifikasi)

        self.btnSegmentasi = QPushButton("🧩 Segmentasi Citra")
        self.btnSegmentasi.clicked.connect(self.segmentasiCitra)
        controlPanel.addWidget(self.btnSegmentasi)

        self.btnDeteksi = QPushButton("🎯 Deteksi Objek")
        self.btnDeteksi.clicked.connect(self.deteksiObjek)
        controlPanel.addWidget(self.btnDeteksi)

        def createImageColumn(label_text):
            view = QLabel()
            view.setMinimumSize(200, 200)
            view.setFrameShape(QLabel.Box)
            view.setStyleSheet("background-color: white; border: 2px solid #22c55e;")
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignCenter)
            return view, label

        self.viewOriginal, self.labelAvgOriginal = createImageColumn("Original")
        self.viewRGB, self.labelAvgRGB = createImageColumn("R: -, G: -, B: -")
        self.viewGrayscale, self.labelAvgGrayscale = createImageColumn("Grayscale: -")
        self.viewBiner, self.labelAvgBiner = createImageColumn("Biner: -")

        imageLayout = QHBoxLayout()
        for view, label in [
            (self.viewOriginal, self.labelAvgOriginal),
            (self.viewRGB, self.labelAvgRGB),
            (self.viewGrayscale, self.labelAvgGrayscale),
            (self.viewBiner, self.labelAvgBiner),
        ]:
            col = QVBoxLayout()
            col.addWidget(view)
            col.addWidget(label)
            imageLayout.addLayout(col)

        self.tableRGB = QTableWidget()
        self.tableRGB.setColumnCount(5)
        self.tableRGB.setHorizontalHeaderLabels(["R", "G", "B", "Grayscale", "Biner"])
        self.tableRGB.setStyleSheet("background-color: #fff;")

        imageArea = QVBoxLayout()
        imageArea.addLayout(imageLayout)
        imageArea.addWidget(self.tableRGB)

        topLayout = QHBoxLayout()
        topLayout.addLayout(controlPanel, 1)
        topLayout.addLayout(imageArea, 4)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(titleLabel)
        mainLayout.addLayout(topLayout)
        self.setLayout(mainLayout)

    def adjust_image(self, image):
        brightness = self.sliderBrightness.value()
        contrast = self.sliderContrast.value()
        saturation = self.sliderSaturation.value()
        rotation = self.spinRotation.value()
        scale = self.spinScale.value()

        new_img = image.astype(np.float32)
        new_img = new_img * (1 + contrast / 100.0) + brightness
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)

        if new_img.ndim == 3 and saturation != 0:
            hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= (1 + saturation / 100.0)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            new_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        h, w = new_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation, scale)
        rotated_scaled = cv2.warpAffine(new_img, M, (w, h))

        return rotated_scaled

    def updateImage(self):
        if self.original_image is None:
            return
        self.showOriginal()
        img = self.adjust_image(self.original_image.copy())
        self.rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, self.binary_image = cv2.threshold(self.gray_image, 127, 255, cv2.THRESH_BINARY)
        self.showRGB()
        self.showGrayscale()
        self.showBiner()

    def loadImage(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if filename:
            self.original_image = cv2.imread(filename)
            self.updateImage()
            self.deteksiObjek()

    def showOriginal(self):
        if self.original_image is not None:
            bgr = self.original_image
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.displayImage(self.viewOriginal, rgb)

    def showRGB(self):
        if self.rgb_image is not None:
            self.displayImage(self.viewRGB, self.rgb_image)
            avg = cv2.mean(self.rgb_image)
            self.labelAvgRGB.setText(f"R: {int(avg[0])}, G: {int(avg[1])}, B: {int(avg[2])}")

    def showGrayscale(self):
        if self.gray_image is not None:
            self.displayImage(self.viewGrayscale, self.gray_image, is_gray=True)
            self.labelAvgGrayscale.setText(f"Grayscale: {int(np.mean(self.gray_image))}")

    def showBiner(self):
        if self.binary_image is not None:
            self.displayImage(self.viewBiner, self.binary_image, is_gray=True)
            self.labelAvgBiner.setText(f"Biner: {int(np.mean(self.binary_image))}")

    def displayImage(self, label, image, is_gray=False):
        qformat = QImage.Format_Grayscale8 if is_gray else QImage.Format_RGB888
        height, width = image.shape[:2]
        bytesPerLine = 1 * width if is_gray else 3 * width
        qimg = QImage(image.data, width, height, bytesPerLine, qformat)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resetControls(self):
        self.viewOriginal.clear()
        self.viewRGB.clear()
        self.viewGrayscale.clear()
        self.viewBiner.clear()
        self.labelAvgRGB.setText("R: -, G: -, B: -")
        self.labelAvgGrayscale.setText("Grayscale: -")
        self.labelAvgBiner.setText("Biner: -")
        self.tableRGB.setRowCount(0)
        self.sliderBrightness.setValue(0)
        self.sliderContrast.setValue(0)
        self.sliderSaturation.setValue(0)
        self.spinRotation.setValue(0)
        self.spinScale.setValue(1.0)
        self.original_image = None

    def inputToTable(self):
        if self.rgb_image is not None and self.gray_image is not None and self.binary_image is not None:
            h, w, _ = self.rgb_image.shape
            mid_h = h // 2
            mid_w = w // 2
            r, g, b = self.rgb_image[mid_h, mid_w]
            gray = self.gray_image[mid_h, mid_w]
            biner = self.binary_image[mid_h, mid_w]
            self.tableRGB.setRowCount(0)
            self.tableRGB.insertRow(0)
            self.tableRGB.setItem(0, 0, QTableWidgetItem(str(r)))
            self.tableRGB.setItem(0, 1, QTableWidgetItem(str(g)))
            self.tableRGB.setItem(0, 2, QTableWidgetItem(str(b)))
            self.tableRGB.setItem(0, 3, QTableWidgetItem(str(gray)))
            self.tableRGB.setItem(0, 4, QTableWidgetItem(str(biner)))

    def exportToExcel(self):
        rows = self.tableRGB.rowCount()
        cols = self.tableRGB.columnCount()
        data = []
        headers = [self.tableRGB.horizontalHeaderItem(i).text() for i in range(cols)]
        for row in range(rows):
            row_data = [self.tableRGB.item(row, col).text() if self.tableRGB.item(row, col) else "" for col in range(cols)]
            data.append(row_data)
        df = pd.DataFrame(data, columns=headers)
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Excel Files (*.xlsx)")
        if filename:
            if not filename.endswith(".xlsx"):
                filename += ".xlsx"
            df.to_excel(filename, index=False)

    def ekstraksiCiri(self):
     if self.rgb_image is not None:
        mean_r, mean_g, mean_b = cv2.mean(self.rgb_image)[:3]
        print(f"[Ekstraksi Ciri] Mean R: {mean_r:.2f}, G: {mean_g:.2f}, B: {mean_b:.2f}")
        
        # Logika klasifikasi kematangan tomat
        if mean_g > mean_r and mean_g > 120:
            warna = "Mentah"
        elif mean_r > 150 and mean_r > mean_g + 30:
            warna = "Matang"
        elif mean_r > 120 and mean_g > 80 and abs(mean_r - mean_g) <= 40:
            warna = "Setengah Matang"
        else:
            warna = "Tidak dikenali"
        
        print(f"[Analisis Ciri] Tingkat kematangan tomat: {warna} (R: {int(mean_r)}, G: {int(mean_g)}, B: {int(mean_b)})")

    def klasifikasiKematangan(self):
     if self.rgb_image is not None:
        mean_r, mean_g, mean_b = cv2.mean(self.rgb_image)[:3]
        
        # Hitung rasio warna untuk klasifikasi yang lebih akurat
        total_rgb = mean_r + mean_g + mean_b
        ratio_r = mean_r / total_rgb if total_rgb > 0 else 0
        ratio_g = mean_g / total_rgb if total_rgb > 0 else 0
        
        # Logika klasifikasi berdasarkan karakteristik tomat
        # MENTAH: Hijau dominan dengan nilai G tinggi
        if (mean_g > mean_r and mean_g > mean_b and 
            mean_g > 100 and ratio_g > 0.35):
            if mean_g > 140:
                kategori = "Mentah (Hijau Tua)"
            else:
                kategori = "Mentah (Hijau Muda)"
        
        # SANGAT MATANG: Merah sangat dominan dan pekat
        elif (mean_r > 180 and mean_r > mean_g + 50 and 
              mean_r > mean_b + 50 and ratio_r > 0.45):
            kategori = "Sangat Matang"
        
        # MATANG: Merah dominan
        elif (mean_r > 150 and mean_r > mean_g + 30 and 
              mean_r > mean_b + 30 and ratio_r > 0.40):
            kategori = "Matang"
        
        # SETENGAH MATANG: Campuran merah-oren/kuning
        elif (mean_r > 120 and mean_g > 80 and 
              abs(mean_r - mean_g) <= 50 and 
              mean_r >= mean_g and ratio_r > 0.32):
            if mean_r > mean_g + 20:
                kategori = "Setengah Matang (Oren Kemerahan)"
            else:
                kategori = "Setengah Matang (Oren Kuning)"
        
        # Default jika tidak masuk kategori manapun
        else:
            kategori = "Tidak dikenali"
        
        print(f"[Klasifikasi] Rata-rata Warna - R: {int(mean_r)}, G: {int(mean_g)}, B: {int(mean_b)}")
        print(f"[Klasifikasi] Rasio - R: {ratio_r:.3f}, G: {ratio_g:.3f}, B: {(mean_b/total_rgb):.3f}")
        print(f"[Klasifikasi] Tomat termasuk kategori: {kategori}")
        
        return kategori
     else:
        print("[Error] Tidak ada gambar RGB yang tersedia")
        return "Error"



    def segmentasiCitra(self):
        if self.gray_image is not None:
            _, segmented = cv2.threshold(self.gray_image, 100, 255, cv2.THRESH_BINARY)
            cv2.imshow("Segmentasi Citra", segmented)
            unique, counts = np.unique(segmented, return_counts=True)
            pixel_info = dict(zip(unique, counts))
            obj_pixels = pixel_info.get(255, 0)
            total_pixels = segmented.size
            print(f"[Segmentasi] Jumlah piksel objek: {obj_pixels} dari total {total_pixels} piksel.")

def deteksi_objek_tomat_dari_file(path_gambar):
    rgb_image = cv2.imread(path_gambar)
    if rgb_image is None:
        print("Gagal membaca gambar.")
        return

    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        circularity = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2 + 1e-5)

        if not (0.7 <= aspect_ratio <= 1.3 and circularity > 0.5):
            continue

        roi = rgb_image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        std_dev = np.std(gray_roi)

        if std_dev < 10:
            continue

        detected = True
        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(rgb_image, "TOMAT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    if detected:
        print("[Deteksi] Objek TOMAT terdeteksi.")
    else:
        print("[Deteksi] Bukan tomat – objek tidak dikenali.")

    hasil = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hasil Deteksi", hasil)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
