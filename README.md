Aplikasi Klasifikasi Kematangan Tomat dan Deteksi Objek Berbasis PyQt5
📌 Tujuan Proyek
Proyek ini bertujuan untuk membangun sebuah aplikasi GUI berbasis PyQt5 yang mampu melakukan pengolahan citra digital untuk mendukung klasifikasi tingkat kematangan tomat dan mendeteksi objek tomat secara otomatis, baik dari gambar statis maupun kamera secara real-time. Aplikasi ini mencakup fitur lengkap seperti peningkatan kualitas citra, konversi ke beberapa format (RGB, Grayscale, Binary), ekstraksi ciri mean warna, klasifikasi tingkat kematangan, deteksi objek visual tomat, dan ekspor hasil ke file Excel.

✅ Manfaat Proyek
1. Otomatisasi proses klasifikasi kematangan tomat, yang sebelumnya dilakukan secara manual dan subjektif.
2. Mendeteksi objek tomat secara otomatis menggunakan kombinasi warna, bentuk, dan tekstur, sehingga mampu membedakan tomat dari objek lain yang serupa (misalnya bola merah).
3. Peningkatan efisiensi dan akurasi dalam proses sortasi hasil panen.
4. Media pembelajaran dalam bidang pengolahan citra digital dan penerapan machine learning berbasis fitur sederhana.
5. User-friendly, karena dilengkapi dengan antarmuka GUI interaktif menggunakan PyQt5.

FITUR - FITUR UTAMA
1. Load Image dan Tampilan RGB, Grayscale, Binary
   Memuat gambar tomat dan menampilkan hasil dalam tiga mode tampilan warna untuk keperluan analisis visual.
2. Brightness, Contrast, Saturation, Rotation, Scale
   Mengatur pencahayaan, kontras, kejenuhan warna, rotasi, dan skala gambar untuk meningkatkan kualitas visual citra.
3. Ekstraksi Mean Warna RGB
   Menghitung rata-rata nilai warna merah, hijau, dan biru dari citra tomat untuk digunakan sebagai fitur klasifikasi.
4. Klasifikasi Tingkat Kematangan Tomat Berdasarkan Mean RGB
   Menentukan apakah tomat mentah, setengah matang, atau matang berdasarkan nilai rata-rata warna RGB.
5. Deteksi Objek Tomat dari Citra dan Kamera
   Mendeteksi keberadaan tomat secara otomatis dalam gambar atau video menggunakan segmentasi warna dan analisis bentuk.
6. Ekspor Data Hasil Klasifikasi ke Excel
   Menyimpan hasil klasifikasi tomat ke dalam file Excel untuk dokumentasi atau analisis lebih lanjut.
7. Tombol Reset dan Kontrol Kamera (Start/Stop)
   Mengatur ulang semua pengaturan dan memberi kontrol untuk memulai atau menghentikan deteksi kamera secara real-time.

DEPENDENCIES
1. Python 3.x
2. OpenCV (cv2)
3. NumPy
4. PyQt5
5. pandas

BY : 152023180 - Ai Resti S
     152023183 - Rafina Az Zahra
     152023210 - Jihan Khairunnisa
     152023219 - Merry Gabriella A
