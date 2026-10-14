# Structural Damage Classification

Proyek ini memprediksi `damage_grade` bangunan setelah gempa. Target terdiri
dari lima kelas, yaitu Grade 1 sampai Grade 5. Evaluasi menggunakan macro-F1,
sehingga F1 setiap kelas memiliki bobot yang sama.

Script training membentuk ensemble dari CatBoost multiclass dan empat
LightGBM ordinal. Model LightGBM memprediksi empat batas kelas: `y > 1`,
`y > 2`, `y > 3`, dan `y > 4`. Probabilitas CatBoost dan LightGBM kemudian
digabungkan menggunakan bobot yang dipilih dari validation set.

## Struktur repository

```text
nama-kompetisi-kaggle/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── metadata.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_submission.ipynb
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── models/
│   ├── pipeline_model_ensemble.pkl
│   └── metadata.json
├── submissions/
│   └── submission.csv
└── reports/
    └── figures/
        ├── eda_target_distribution.png
        ├── eda_building_age_distribution.png
        ├── eda_foundation_damage_grade.png
        ├── eda_numeric_features_by_grade.png
        ├── eda_age_group_damage_grade.png
        ├── eda_superstructure_damage_grade.png
        ├── eda_district_average_grade.png
        ├── eda_train_test_numeric_comparison.png
        ├── eda_numeric_correlation.png
        └── feature_importance.png
```

File di dalam `models`, `submissions`, dan `reports/figures` dibuat saat
notebook, training, atau prediksi dijalankan. Data kompetisi, model, dan submission
dikecualikan dari Git melalui `.gitignore`.

## Workflow analisis

1. Periksa struktur data, missing value, duplikasi, dan distribusi target.
2. Hitung cardinality kategori dan periksa kategori test yang tidak terdapat
   pada train.
3. Bandingkan fitur numerik, umur, material, fondasi, dan lokasi terhadap
   `damage_grade`.
4. Bandingkan distribusi fitur antara train dan test.
5. Buat fitur umur, ukuran bangunan, dan prefix lokasi.
6. Pisahkan preprocessing fitur kategorikal dan numerik dengan
   `ColumnTransformer`.
7. Latih CatBoost multiclass.
8. Buat target encoding secara out-of-fold untuk data LightGBM.
9. Latih empat LightGBM ordinal.
10. Pilih bobot probabilitas dan bias kelas berdasarkan macro-F1 validation.
11. Latih model final menggunakan seluruh data train.
12. Prediksi data test dan periksa format submission.

## Hasil EDA

Data training berisi 533.465 baris. Distribusi targetnya adalah:

| Target | Jumlah | Persentase |
|---|---:|---:|
| Grade 1 | 55.170 | 10,34% |
| Grade 2 | 61.080 | 11,45% |
| Grade 3 | 95.488 | 17,90% |
| Grade 4 | 128.691 | 24,12% |
| Grade 5 | 193.036 | 36,19% |

Grade 5 mencakup 36,19% data, sedangkan Grade 1 mencakup 10,34%. CatBoost
menggunakan class weight untuk mengurangi pengaruh perbedaan jumlah tersebut.
Macro-F1 digunakan agar hasil pada kelas kecil tetap diperhitungkan.

Sebanyak 2.198 baris, atau 0,41% data, memiliki nilai umur `999`. Dataset
menggunakan nilai tersebut untuk umur yang tidak diketahui. Feature engineering
membuat kolom `age_is_999` dan membatasi `age_clean` pada rentang 0 sampai 100.

Median umur menunjukkan pola bertingkat. Median umur Grade 1 adalah 7 tahun,
Grade 2 adalah 12 tahun, Grade 3 adalah 16 tahun, sedangkan Grade 4 dan Grade 5
masing-masing 20 tahun. Median luas dasar turun dari 400 kaki persegi pada
Grade 1 menjadi 342 kaki persegi pada Grade 5. Ringkasan ini bersifat deskriptif
dan tidak menunjukkan hubungan sebab akibat.

EDA memeriksa 11 fitur kategorikal, termasuk district, ward, foundation,
roof, position, dan superstructure. Seluruh nilai kategori pada test juga
terdapat pada train. Model tetap menangani kategori yang tidak dikenal agar
pipeline dapat digunakan pada data baru.

Rata-rata fitur numerik train dan test serupa. Perbedaannya adalah -0,03% untuk
jumlah lantai, -0,29% untuk umur, 0,12% untuk luas dasar, dan 0,04% untuk tinggi
bangunan. Perbandingan ini tidak menemukan pergeseran besar pada empat fitur
numerik tersebut.

## Feature engineering

`DamageFeatureEngineer` membuat fitur berikut:

- prefix `building_id` dengan panjang 3 sampai 8 karakter;
- penanda umur bernilai 999;
- tinggi rata-rata per lantai;
- luas rata-rata per lantai;
- perkiraan volume bangunan.

`building_id` lengkap tidak dimasukkan ke model karena nilainya unik untuk
setiap bangunan. Prefix ID digunakan sebagai kategori lokasi dengan tingkat
detail yang berbeda.

LightGBM juga menggunakan target encoding untuk distrik, municipality, ward,
prefix bangunan, serta kombinasi lokasi dan material. Mapping dibuat pada fold
training dan diterapkan pada fold validation. Target pada fold validation tidak
digunakan untuk membuat mapping fold tersebut.

## Pipeline dan ColumnTransformer

Cabang CatBoost menggunakan pipeline berikut:

```text
DamageFeatureEngineer
        ↓
ColumnTransformer
        ↓
CatBoostClassifier
```

`ColumnTransformer` memisahkan fitur kategorikal dan numerik. Missing value
kategorikal diisi dengan nilai yang paling sering muncul, sedangkan missing
value numerik diisi dengan median. CatBoost menerima kategori secara langsung,
sehingga cabang ini tidak menggunakan one-hot encoding atau standardisasi.

## Interpretasi data

Jenis fondasi memiliki distribusi grade yang berbeda. Pada fondasi
`Mud mortar-Stone/Brick`, 41,7% observasi berada pada Grade 5. Pada fondasi
`RC`, 59,5% observasi berada pada Grade 1 dan 1,6% berada pada Grade 5.

Kategori `has_superstructure_mud_mortar_stone` muncul pada 415.327 baris dan
merupakan kategori superstructure yang paling sering ditemukan. Model memakai
`foundation_type` dan `superstructure` sebagai fitur kategorikal. Target
encoding juga memakai kombinasi kedua fitur tersebut.

Proporsi Grade 5 berubah menurut kelompok umur. Pada kelompok 0 sampai 10 tahun,
27,7% observasi berada pada Grade 5. Proporsinya menjadi 37,8% pada kelompok
11 sampai 20 tahun dan 42,3% pada kelompok 21 sampai 40 tahun. Kelompok 41
sampai 60 tahun mempunyai proporsi Grade 5 sebesar 42,6%.

Rata-rata grade juga berbeda antar-distrik. District 31 mempunyai rata-rata
grade terendah, yaitu 2,349, sedangkan District 23 mempunyai rata-rata tertinggi,
yaitu 4,680. Perbedaan ini mendukung penggunaan district dan prefix lokasi
sebagai fitur. Nilai tersebut merupakan ringkasan data, bukan perkiraan dampak
lokasi secara terpisah dari material, umur, atau karakteristik bangunan lain.

Notebook modeling menyimpan grafik feature importance dari cabang CatBoost ke
`reports/figures/feature_importance.png`. Grafik itu menjelaskan cabang
CatBoost saja. Kontribusi LightGBM masuk melalui probabilitas hasil blending.

## Instalasi

Proyek menggunakan Python 3.10 atau versi yang lebih baru. Jalankan perintah
berikut dari folder proyek pada Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Daftarkan virtual environment sebagai kernel Jupyter:

```powershell
python -m ipykernel install --user --name damage-classification --display-name "Python (Damage Classification)"
```

## Menjalankan notebook

Jalankan notebook dalam urutan berikut:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_modeling.ipynb`
3. `notebooks/03_submission.ipynb`

Pada `02_modeling.ipynb`, atur `RUN_FULL_TRAINING = True` untuk menjalankan
training. Proses tersebut menyimpan model ke
`models/pipeline_model_ensemble.pkl` dan metrik validation ke
`models/metadata.json`.

`03_submission.ipynb` memuat model tersebut, memprediksi test, dan menyimpan
hasil ke `submissions/submission.csv`. Notebook juga memeriksa jumlah baris,
urutan `building_id`, label, dan missing value. Kolom `damage_grade` pada file
submission menggunakan angka 1 sampai 5. Model tetap menggunakan label
`Grade 1` sampai `Grade 5` selama training.

## Menjalankan dari terminal

Training:

```powershell
python -m scripts.train
```

Prediksi:

```powershell
python -m scripts.predict
```

Training penuh menggunakan 533.465 baris dan dapat berjalan selama beberapa
jam, tergantung CPU dan jumlah thread yang tersedia.

## Output training

`models/metadata.json` mencatat konfigurasi, jumlah baris training, waktu
training, bobot LightGBM, macro-F1 sebelum kalibrasi, dan macro-F1 setelah
kalibrasi. Nilai tersebut berasal dari validation set yang digunakan untuk
memilih bobot ensemble dan bias kelas.

Artifact model memuat pipeline CatBoost, empat model LightGBM, mapping kategori,
mapping target encoding, bobot blend, dan bias kelas. Folder `scripts` harus
tersedia ketika artifact dimuat karena pipeline menggunakan class yang
didefinisikan di dalam proyek.

## Catatan penggunaan

- Seed training ditetapkan ke 42.
- `sample_submission.csv` digunakan untuk memeriksa urutan ID dan nama kolom.
- Data kompetisi tidak disimpan di Git.
- Periksa ketentuan kompetisi sebelum membagikan data atau hasil prediksi.
