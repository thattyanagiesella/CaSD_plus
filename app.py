import os
import io
import base64
import traceback
import json
from datetime import datetime
from functools import wraps
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import networkx as nx
import seaborn as sns

import numpy as np
import pandas as pd

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, session, jsonify
)


import mysql.connector
from mysql.connector import errorcode

app = Flask(__name__, template_folder="templates", static_folder="public")
app.secret_key = os.environ.get("FLASK_SECRET", "super-secret-key")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/generated", exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASS", "acel"),
    "database": os.environ.get("DB_NAME", "cashless_db"),
    "raise_on_warnings": True
}
DB_TABLE = "responden"

APP_TITLE = "CaSD+ (Cashless Society Detector)"

WILAYAH_MAP = {
    "1": "Surabaya",
    "2": "Malang",
    "3": "Pasuruan",
    "4": "Jember",
    "5": "Sampang"
}

JENIS_PEKERJAAN_OPTIONS = [
    "ASN/TNI/Polri/Pegawai BUMN",
    "karyawan swasta/ Profesional non PNS (dokter, notaris, pengacara, konsultan, dosen, arsitek)",
    "Pengusaha/Pedagang/Kontraktor",
    "Buruh harian/Ojek Online/Petani/Nelayan/Peternak/Freelancer",
    "IRT/Pelajar/Mahasiswa/Belum (Tidak) Bekerja"
]

LIKERT_SCALE = {
    "SANGAT TIDAK SETUJU": 1,
    "TIDAK SETUJU": 2,
    "NETRAL": 3,
    "SETUJU": 4,
    "SANGAT SETUJU": 5
}
LIKERT_OPTIONS = ["Sangat Tidak Setuju", "Tidak Setuju", "Netral", "Setuju", "Sangat Setuju"]

LIKERT_ALIASES = {
    "STS": "SANGAT TIDAK SETUJU",
    "TS": "TIDAK SETUJU",
    "R": "NETRAL",
    "N": "NETRAL",
    "RAGU-RAGU": "NETRAL",
    "S": "SETUJU",
    "SS": "SANGAT SETUJU",
}

X_VAR_NAME = {
    "X1": "Faktor Psikologi",
    "X2": "Faktor Sosial Budaya",
    "X3": "Faktor Pribadi",
    "X4": "Product",
    "X5": "Price",
    "X6": "Promotion",
    "X7": "Place",
    "X8": "People",
    "X9": "Process",
    "X10": "Physical Evidence"
}

PATH_VAR_NAME = {
    "Y1": "Digitalisasi Uang Elektronik",
    "Y2": "Perkembangan Cashless Society",
    "Y3": "Pemulihan Ekonomi",
    "M": "Financial Technology",
    "Y1_M": "Digitalisasi × Financial Technology",
    "Y2_M": "Cashless Society × Financial Technology",
}

SCORE_COLUMNS_X = [f"X{i}" for i in range(1, 11)] + ["M"]
SCORE_COLUMNS_Y_SURVEY = {"Y1": "Y1_survey", "Y2": "Y2_survey", "Y3": "Y3_survey"}
ALL_SCORE_COLUMNS = SCORE_COLUMNS_X + list(SCORE_COLUMNS_Y_SURVEY.values())

QUESTIONS_DETAILED = {
    "X1": {
        "name": "Faktor Psikologi",
        "items": {
            # Motivasi (X₁.₁)
            "X1.1": {"text": "Mobile banking digunakan karena dapat mempercepat proses transaksi"},
            # Persepsi (X₁.₂)
            "X1.2": {"text": "Mobile banking digunakan karena penggunaannya dianggap mudah"},
            # Pengetahuan (X₁.₃)
            "X1.3": {"text": "Adaptasi dirasa perlu dilakukan untuk menggunakan mobile banking dalam transaksi sehari-hari"},
            # Sikap (X₁.₄)
            "X1.4": {"text": "Aplikasi mobile banking yang digunakan dirasa merupakan aplikasi yang handal"},
        }
    },
    "X2": {
        "name": "Faktor Sosial Budaya",
        "items": {
            # Kelompok Acuan (X₂.₁)
            "X2.1": {"text": "Mobile banking digunakan karena rekomendasi dari teman"},
            # Keluarga (X₂.₂)
            "X2.2": {"text": "Mobile banking digunakan karena kebiasaan dari keluarga"},
            # Peran dan Status (X₂.₃)
            "X2.3": {"text": "Mobile banking digunakan karena mencerminkan peran yang dijalani di lingkungan kerja/sekolah"},
            # Budaya (X₂.₄)
            "X2.4": {"text": "Mobile banking digunakan karena masyarakat sekitar sudah terbiasa menggunakannya"},
            # Sub Budaya (X₂.₅)
            "X2.5": {"text": "Mobile banking digunakan karena lebih mudah digunakan di wilayah tempat tinggal"},
            # Kelas Sosial (X₂.₆)
            "X2.6": {"text": "Mobile banking digunakan karena mengikuti tren masa kini"},
        }
    },
    "X3": {
        "name": "Faktor Pribadi",
        "items": {
            # Usia dan Tahapan Siklus Hidup (X₃.₁)
            "X3.1": {"text": "Mobile banking digunakan karena sesuai dengan kebutuhan yang dirasakan pada masa sekarang"},
            # Gaya Hidup (X₃.₂)
            "X3.2": {"text": "Mobile banking digunakan karena dapat memudahkan aktivitas sehari-hari"},
            # Kepribadian dan Konsep Diri (X₃.₃)
            "X3.3": {"text": "Mobile banking digunakan karena sesuai dengan karakter yang menyukai hal-hal praktis"},
        }
    },
    "X4": {
        "name": "Product",
        "items": {
            # Ketersediaan Macam (X₄.₁)
            "X4.1": {"text": "Mobile banking disukai untuk digunakan karena tersedia berbagai macam fitur"},
            # Pembaruan (X₄.₂)
            "X4.2": {"text": "Mobile banking yang digunakan selalu diperbaharui fitur-fiturnya secara berkala"},
            # Keandalan Sistem (X4.3)
            "X4.3": {"text": "Mobile banking yang digunakan dapat diandalkan untuk memproses transaksi dengan lancar dan akurat"},
        }
    },
    "X5": {
        "name": "Price",
        "items": {
            # Kesesuaian Harga (X₅.₁)
            "X5.1": {"text": "Biaya layanan mobile banking dirasa sudah sesuai dengan kemudahan transaksi yang dirasakan"},
            # Perbandingan Harga (X₅.₂)
            "X5.2": {"text": "Mobile banking yang digunakan dirasa menawarkan biaya yang lebih murah daripada mobile banking dari kompetitor lain"},
        }
    },
    "X6": {
        "name": "Promotion",
        "items": {
            # Sponsorship (X₆.₁)
            "X6.1": {"text": "Mobile banking yang digunakan secara rutin melakukan sponsorship pada event tertentu"},
            # Media (X₆.₂)
            "X6.2": {"text": "Mobile banking yang digunakan dirasa hampir selalu menggunakan media sosial untuk melakukan promosi"},
            # Tawaran yang Diberikan (X₆.₃)
            "X6.3": {"text": "Mobile banking yang digunakan dirasa hampir selalu memberikan reward pada nasabah"},
        }
    },
    "X7": {
        "name": "Place",
        "items": {
            # Letak (X₇.₁)
            "X7.1": {"text": "Lokasi kantor untuk melakukan aktivasi mobile banking dirasa mudah diakses"},
            # Suasana (X₇.₂)
            "X7.2": {"text": "Kenyamanan dirasakan saat melakukan transaksi menggunakan mobile banking"},
        }
    },
    "X8": {
        "name": "People",
        "items": {
            # Kinerja (X₈.₁)
            "X8.1": {"text": "Keluhan pada aplikasi ditanggapi dengan cepat oleh customer service"},
            # Sikap (X₈.₂)
            "X8.2": {"text": "Kepedulian tinggi dirasa dimiliki oleh customer service saat melayani pengguna mobile banking"},
        }
    },
    "X9": {
        "name": "Process",
        "items": {
            # Arus Kegiatan (X₉.₁)
            "X9.1": {"text": "Prosedur pembukaan akun/aktivasi mobile banking dirasa mudah"},
            # Langkah Proses (X₉.₂)
            "X9.2": {"text": "Proses layanan transaksi melalui mobile banking dirasa lebih efektif"},
            # Penyelesaian Proses (X9.3)
            "X9.3": {"text": "Setelah melakukan transaksi melalui mobile banking, terdapat konfirmasi atau bukti transaksi yang jelas"},
        }
    },
    "X10": {
        "name": "Physical Evidence",
        "items": {
            # Tampilan (X10.1)
            "X10.1": {"text": "Tampilan dari aplikasi mobile banking dinilai menarik dan mudah untuk digunakan"},
            # Fasilitas (X10.2)
            "X10.2": {"text": "Fitur yang tersedia di mobile banking dirasa membuat nasabah nyaman dan aman dalam melakukan transaksi"},
        }
    },
    "Y1": {
        "name": "Digitalisasi Uang Elektronik",
        "items": {
            # Convenience (Y1.1)
            "Y1.1": {"text": "Penggunaan mobile banking dirasa mudah dipelajari"},
            # Complementarity (Y1.2)
            "Y1.2": {"text": "Mobile banking dinilai sudah sesuai dengan tingkat pertumbuhan ekonomi saat ini"},
            # Network Effects (Y1.3)
            "Y1.3": {"text": "Mobile banking diinginkan untuk digunakan ketika ada pilihan untuk melakukan transaksi tunai atau nontunai"},
            # Security (Y1.4)
            "Y1.4": {"text": "Responden merasa aman ketika bertransaksi dengan menggunakan aplikasi mobile banking"},
            # Regulation (Y1.5)
            "Y1.5": {"text": "Responden melihat bahwa dengan berbelanja menggunakan mobile banking dapat melindungi responden"},
            # Trust (Y1.6)
            "Y1.6": {"text": "Responden lebih percaya ketika berbelanja dengan menggunakan mobile banking"},
        }
    },
    "Y2": {
        "name": "Perkembangan Cashless Society",
        "items": {
            # Akselerasi Pembayaran Digital (Y₂.₁)
            "Y2.1": {"text": "Penggunaan mobile banking dirasa sudah diterapkan di berbagai transaksi"},
            # Kemudahan Transaksi (Y₂.₂)
            "Y2.2": {"text": "Sistem transaksi yang dilakukan dengan mobile banking dirasa jelas dan mudah dimengerti"},
        }
    },
    "M": {
        "name": "Financial Technology",
        "items": {
            # Pembayaran (M₁)
            "M1": {"text": "Perkembangan sistem pembayaran digital pada semua bank dirasa berkembang cepat"},
            # Layanan Konsultasi (M₂)
            "M2": {"text": "Layanan konsultasi aktivasi mobile banking hampir selalu didengar dilakukan secara online"},
            # Kepatuhan (M₃)
            "M3": {"text": "Gaya hidup yang lebih hemat dirasa dimiliki karena adanya kepatuhan penerapan budget pada aplikasi"},
            # Kepuasan (M₄)
            "M4": {"text": "Responden merasa puas dalam membayar dengan menggunakan pembayaran digital"},
        }
    },
    "Y3": {
        "name": "Pemulihan Ekonomi",
        "items": {
            # Kolaborasi Bisnis dan Teknologi (Y3.1)
            "Y3.1": {"text": "Pengguna terbiasa dengan pemanfaatan teknologi dalam bisnis"},
            # Pemberdayaan pelaku UMKM (Y3.2)
            "Y3.2": {"text": "Pelaku UMKM diarahkan untuk mengikuti perkembangan ekonomi digital untuk mendukung pemberdayaan UMKM"},
            # Pertumbuhan Ekonomi (Y3.3)
            "Y3.3": {"text": "Pertumbuhan ekonomi meningkat ketika mulai banyak masyarakat melakukan transaksi dengan sistem cashless"},
        }
    },
}

SEM_COEFFS = {
    "1": {  # Wilayah A
        "Y1": {"X1": 0.328, "X2": 0.123, "X3": 0.198, "X4": 0.324, "X5": 0.264, "X6": 0.394, "X7": 0.579, "X8": 0.554, "X9": 0.345, "X10": 0.588},
        "Y2": {"X1": 0.482, "X2": -0.04, "X3": 0.573, "X4": 0.254, "X5": 0.341, "X6": 0.421, "X7": 0.347, "X8": 0.476, "X9": 0.159, "X10": 0.3, "Y1": 0.222},
        "Y3": {"Y1": 0.343, "Y2": 0.219, "M": 0.256, "Y1_M": 0.129, "Y2_M": 0.281},
    },
    "2": {  # Wilayah B
        "Y1": {"X1": 0.194, "X2": 0.189, "X3": 0.319, "X4": 0.21, "X5": 0.539, "X6": 0.41, "X7": 0.355, "X8": 0.461, "X9": 0.473, "X10": 0.574},
        "Y2": {"X1": 0.199, "X2": 0.144, "X3": 0.386, "X4": 0.14, "X5": 0.459, "X6": 0.33, "X7": 0.437, "X8": -0.04, "X9": 0.25, "X10": 0.208, "Y1": 0.566},
        "Y3": {"Y1": 0.441, "Y2": 0.193, "M": 0.34, "Y1_M": 0.094, "Y2_M": 0.28},
    },
    "3": {  # Wilayah C
        "Y1": {"X1": 0.284, "X2": 0.42, "X3": 0.279, "X4": 0.261, "X5": 0.457, "X6": 0.389, "X7": 0.905, "X8": 0.385, "X9": 0.596, "X10": 0.705},
        "Y2": {"X1": 0.274, "X2": 0.496, "X3": 0.643, "X4": 0.224, "X5": 0.547, "X6": 0.349, "X7": 0.569, "X8": 0.319, "X9": 0.344, "X10": 0.39, "Y1": 0.16},
        "Y3": {"Y1": 0.192, "Y2": 0.385, "M": 0.168, "Y1_M": 0.239, "Y2_M": 0.17},
    },
    "4": {  # Wilayah D
        "Y1": {"X1": 0.194, "X2": 0.248, "X3": 0.194, "X4": 0.321, "X5": 0.329, "X6": 0.425, "X7": 0.335, "X8": 0.564, "X9": 0.454, "X10": 0.518},
        "Y2": {"X1": 0.419, "X2": 0.361, "X3": 0.319, "X4": 0.323, "X5": 0.282, "X6": 0.586, "X7": 0.038, "X8": 0.431, "X9": 0.138, "X10": 0.426, "Y1": 0.176},
        "Y3": {"Y1": 0.281, "Y2": 0.251, "M": 0.176, "Y1_M": 0.141, "Y2_M": 0.303},
    },
    "5": {  # Wilayah E
        "Y1": {"X1": 0.159, "X2": 0.208, "X3": 0.216, "X4": 0.051, "X5": 0.429, "X6": 0.403, "X7": 0.481, "X8": 0.674, "X9": 0.503, "X10": 0.728},
        "Y2": {"X1": 0.141, "X2": 0.588, "X3": 0.157, "X4": 0.441, "X5": 0.327, "X6": 0.126, "X7": 0.104, "X8": 0.241, "X9": 0.024, "X10": 0.337, "Y1": 0.475},
        "Y3": {"Y1": 0.338, "Y2": 0.321, "M": 0.229, "Y1_M": 0.174, "Y2_M": 0.225},
    },
}

ITEM_LOADING = {
    "1": {
        "X1.1": 0.2579, "X1.2": 0.2426, "X1.3": 0.2541, "X1.4": 0.2454, "X2.1": 0.1758,
        "X2.2": 0.1698, "X2.3": 0.1661, "X2.4": 0.1704, "X2.5": 0.163, "X2.6": 0.1549,
        "X3.1": 0.333, "X3.2": 0.3485, "X3.3": 0.3185, "X4.1": 0.3459, "X4.2": 0.3418,
        "X4.3": 0.3123, "X5.1": 0.5276, "X5.2": 0.4724, "X6.1": 0.3359, "X6.2": 0.3544,
        "X6.3": 0.3097, "X7.1": 0.4401, "X7.2": 0.5599, "X8.1": 0.4814, "X8.2": 0.5186,
        "X9.1": 0.3084, "X9.2": 0.3595, "X9.3": 0.3321, "X10.1": 0.5302, "X10.2": 0.4698,
        "Y1.1": 0.1675, "Y1.2": 0.1614, "Y1.3": 0.1724, "Y1.4": 0.1562, "Y1.5": 0.1755,
        "Y1.6": 0.1669, "Y2.1": 0.5061, "Y2.2": 0.4939, "Y3.1": 0.3231, "Y3.2": 0.3304,
        "Y3.3": 0.3465, "M1": 0.2267, "M2": 0.2596, "M3": 0.2565, "M4": 0.2572,
    },
    "2": {
        "X1.1": 0.2549, "X1.2": 0.2379, "X1.3": 0.2473, "X1.4": 0.2599, "X2.1": 0.178,
        "X2.2": 0.1764, "X2.3": 0.1665, "X2.4": 0.1761, "X2.5": 0.1659, "X2.6": 0.137,
        "X3.1": 0.3239, "X3.2": 0.3642, "X3.3": 0.3119, "X4.1": 0.3548, "X4.2": 0.342,
        "X4.3": 0.3031, "X5.1": 0.4716, "X5.2": 0.5284, "X6.1": 0.3167, "X6.2": 0.3622,
        "X6.3": 0.321, "X7.1": 0.4615, "X7.2": 0.5385, "X8.1": 0.3891, "X8.2": 0.6109,
        "X9.1": 0.3249, "X9.2": 0.2948, "X9.3": 0.3803, "X10.1": 0.4536, "X10.2": 0.5464,
        "Y1.1": 0.1577, "Y1.2": 0.175, "Y1.3": 0.1717, "Y1.4": 0.1555, "Y1.5": 0.1698,
        "Y1.6": 0.1703, "Y2.1": 0.5028, "Y2.2": 0.4972, "Y3.1": 0.3332, "Y3.2": 0.3309,
        "Y3.3": 0.3359, "M1": 0.2283, "M2": 0.254, "M3": 0.2678, "M4": 0.25,
    },
    "3": {
        "X1.1": 0.2535, "X1.2": 0.2735, "X1.3": 0.2518, "X1.4": 0.2213, "X2.1": 0.1685,
        "X2.2": 0.1694, "X2.3": 0.1699, "X2.4": 0.1641, "X2.5": 0.1554, "X2.6": 0.1727,
        "X3.1": 0.3438, "X3.2": 0.3293, "X3.3": 0.3269, "X4.1": 0.3437, "X4.2": 0.3452,
        "X4.3": 0.3111, "X5.1": 0.5255, "X5.2": 0.4745, "X6.1": 0.3316, "X6.2": 0.3292,
        "X6.3": 0.3393, "X7.1": 0.4715, "X7.2": 0.5285, "X8.1": 0.5397, "X8.2": 0.4603,
        "X9.1": 0.3314, "X9.2": 0.3541, "X9.3": 0.3144, "X10.1": 0.5008, "X10.2": 0.4992,
        "Y1.1": 0.1767, "Y1.2": 0.1616, "Y1.3": 0.1645, "Y1.4": 0.1603, "Y1.5": 0.1709,
        "Y1.6": 0.166, "Y2.1": 0.4872, "Y2.2": 0.5128, "Y3.1": 0.3335, "Y3.2": 0.3478,
        "Y3.3": 0.3187, "M1": 0.277, "M2": 0.2333, "M3": 0.2386, "M4": 0.2511,
    },
    "4": {
        "X1.1": 0.2696, "X1.2": 0.2504, "X1.3": 0.2397, "X1.4": 0.2402, "X2.1": 0.1828,
        "X2.2": 0.1708, "X2.3": 0.1685, "X2.4": 0.1639, "X2.5": 0.1603, "X2.6": 0.1537,
        "X3.1": 0.3234, "X3.2": 0.3096, "X3.3": 0.367, "X4.1": 0.3425, "X4.2": 0.339,
        "X4.3": 0.3185, "X5.1": 0.4501, "X5.2": 0.5499, "X6.1": 0.3231, "X6.2": 0.3579,
        "X6.3": 0.319, "X7.1": 0.5901, "X7.2": 0.4099, "X8.1": 0.5087, "X8.2": 0.4913,
        "X9.1": 0.3562, "X9.2": 0.34, "X9.3": 0.3037, "X10.1": 0.5115, "X10.2": 0.4885,
        "Y1.1": 0.1719, "Y1.2": 0.1711, "Y1.3": 0.1659, "Y1.4": 0.1639, "Y1.5": 0.167,
        "Y1.6": 0.1602, "Y2.1": 0.4937, "Y2.2": 0.5063, "Y3.1": 0.3215, "Y3.2": 0.3456,
        "Y3.3": 0.3329, "M1": 0.2722, "M2": 0.2279, "M3": 0.2641, "M4": 0.2358,
    },
    "5": {
        "X1.1": 0.2389, "X1.2": 0.2402, "X1.3": 0.2706, "X1.4": 0.2503, "X2.1": 0.1776,
        "X2.2": 0.1538, "X2.3": 0.1658, "X2.4": 0.1712, "X2.5": 0.1589, "X2.6": 0.1726,
        "X3.1": 0.3717, "X3.2": 0.2854, "X3.3": 0.3429, "X4.1": 0.3155, "X4.2": 0.3337,
        "X4.3": 0.3508, "X5.1": 0.4431, "X5.2": 0.5569, "X6.1": 0.3833, "X6.2": 0.3256,
        "X6.3": 0.291, "X7.1": 0.5125, "X7.2": 0.4875, "X8.1": 0.4728, "X8.2": 0.5272,
        "X9.1": 0.3372, "X9.2": 0.3577, "X9.3": 0.3051, "X10.1": 0.4311, "X10.2": 0.5689,
        "Y1.1": 0.165, "Y1.2": 0.1515, "Y1.3": 0.1774, "Y1.4": 0.1661, "Y1.5": 0.1649,
        "Y1.6": 0.175, "Y2.1": 0.5013, "Y2.2": 0.4987, "Y3.1": 0.3328, "Y3.2": 0.3234,
        "Y3.3": 0.3438, "M1": 0.2503, "M2": 0.243, "M3": 0.2582, "M4": 0.2485,
    },
}

SEM_MEANS = {
    "1": {"X1": 3.0525, "X2": 2.95, "X3": 3.0983, "X4": 2.975, "X5": 2.9675, "X6": 3.0683, "X7": 2.93, "X8": 2.9375, "X9": 2.9217, "X10": 3.0225, "Y1": 3.0058, "Y2": 3.1175, "Y3": 3.0183, "M": 2.9987},
    "2": {"X1": 2.9863, "X2": 2.9867, "X3": 3.0067, "X4": 2.9383, "X5": 3.1275, "X6": 2.9217, "X7": 2.94, "X8": 3.0275, "X9": 3.0133, "X10": 3.005, "Y1": 2.9433, "Y2": 2.97, "Y3": 2.9667, "M": 3.0412},
    "3": {"X1": 3.0537, "X2": 3.0667, "X3": 3.0267, "X4": 3.03, "X5": 3.09, "X6": 3.0433, "X7": 2.95, "X8": 3.0625, "X9": 2.93, "X10": 3.07, "Y1": 3.0817, "Y2": 2.9375, "Y3": 3.0567, "M": 3.0587},
    "4": {"X1": 2.9525, "X2": 3.0633, "X3": 3.0433, "X4": 3.1067, "X5": 3.0125, "X6": 3.0983, "X7": 3.03, "X8": 3.09, "X9": 3.0283, "X10": 2.97, "Y1": 3.0617, "Y2": 3.0475, "Y3": 3.0683, "M": 2.9813},
    "5": {"X1": 3.0725, "X2": 3.0242, "X3": 3.01, "X4": 2.99, "X5": 3.0375, "X6": 3.0167, "X7": 2.9975, "X8": 3.0125, "X9": 2.9367, "X10": 2.9825, "Y1": 3.0067, "Y2": 2.9475, "Y3": 2.9433, "M": 3.015},
}

SEM_CENTER = {"Y1": 3.0198, "Y2": 3.004, "M": 3.019}

def map_answer_to_weight(ans):
    """Map jawaban ke angka; toleran terhadap format input."""
    if ans is None:
        return 0
    key = str(ans).strip().upper()
    key = LIKERT_ALIASES.get(key, key)
    return LIKERT_SCALE.get(key, 0)

def calculate_weighted_scores(answers, wilayah_id=None):
    weighted_scores = {}
    item_scores = {}
    if not answers:
        return {}, {}

    wilayah_weights = ITEM_LOADING.get(str(wilayah_id), {})

    for var, data in QUESTIONS_DETAILED.items():
        total_score = 0.0
        total_weight = 0.0
        item_scores[var] = {}
        for item_code, item_data in data["items"].items():
            score = answers.get(item_code, 0)
            if score is None:
                score = 0
            weight = float(wilayah_weights.get(item_code, item_data.get("weight", 1)))
            total_score += float(score) * weight
            total_weight += weight
            item_scores[var][item_code] = {
                "score": float(score),
                "weight": weight,
                "weighted_score": float(score) * weight
            }
        weighted_scores[var] = (total_score / total_weight) if total_weight > 0 else 0.0

    return weighted_scores, item_scores

def calculate_sem_scores(x_scores, wilayah_id):
    if not x_scores:
        return {"Y1": 3.0, "Y2": 3.0, "Y3": 3.0}

    try:
        wid = str(wilayah_id)
        coeffs = SEM_COEFFS.get(wid)
        means = SEM_MEANS.get(wid)
        if not coeffs or not means:
            return {"Y1": 3.0, "Y2": 3.0, "Y3": 3.0}

        def val(var):
            v = x_scores.get(var)
            return float(means[var] if v is None else v)

        # Y1
        y1 = means["Y1"]
        for var, coef in coeffs.get("Y1", {}).items():
            y1 += coef * (val(var) - means[var])

        y2 = means["Y2"]
        for var, coef in coeffs.get("Y2", {}).items():
            if var == "Y1":
                y2 += coef * (y1 - means["Y1"])
            else:
                y2 += coef * (val(var) - means[var])

        # Y3 (jalur Y1, Y2, M, serta moderasi Y1xM dan Y2xM)
        y1c = y1 - SEM_CENTER["Y1"]
        y2c = y2 - SEM_CENTER["Y2"]
        mc = val("M") - SEM_CENTER["M"]
        c3 = coeffs.get("Y3", {})
        y3 = (
            means["Y3"]
            + c3.get("Y1", 0.0) * y1c
            + c3.get("Y2", 0.0) * y2c
            + c3.get("M", 0.0) * mc
            + c3.get("Y1_M", 0.0) * (y1c * mc)
            + c3.get("Y2_M", 0.0) * (y2c * mc)
        )

        def clip(v):
            return round(max(1.0, min(5.0, v)), 2)

        return {"Y1": clip(y1), "Y2": clip(y2), "Y3": clip(y3)}

    except Exception as e:
        print("Error calculate_sem_scores:", e)
        traceback.print_exc()
        return {"Y1": 3.0, "Y2": 3.0, "Y3": 3.0}

def score_to_category(score):
    try:
        s = float(score)
        if s >= 3.5:
            return "Tinggi"
        elif s >= 2.5:
            return "Sedang"
        else:
            return "Rendah"
    except Exception:
        return "Rendah"

def analyze_variable_influences(x_scores, wilayah_id, sem_scores):
    wid = str(wilayah_id)
    coeffs = SEM_COEFFS.get(wid, {})
    results = {}
    y_names = {"Y1": "Digitalisasi Uang Elektronik", "Y2": "Cashless Society", "Y3": "Pemulihan Ekonomi"}

    for y_key in ["Y1", "Y2", "Y3"]:
        y_coeffs = coeffs.get(y_key, {})
        influences = []
        for var, coef in y_coeffs.items():
            if var.startswith("X") or var == "M":
                sc = float(x_scores.get(var, 0) or 0)
                inf = float(coef) * sc
                influences.append({
                    "variable": var,
                    "name": QUESTIONS_DETAILED.get(var, {}).get("name", var),
                    "coefficient": coef,
                    "score": sc,
                    "influence": inf
                })
            elif var == "Y1" and y_key in ("Y2", "Y3"):
                sc = float(sem_scores.get("Y1", 0) or 0)
                inf = float(coef) * sc
                influences.append({
                    "variable": "Y1",
                    "name": "Digitalisasi Uang Elektronik",
                    "coefficient": coef,
                    "score": sc,
                    "influence": inf
                })
            elif var == "Y2" and y_key == "Y3":
                sc = float(sem_scores.get("Y2", 0) or 0)
                inf = float(coef) * sc
                influences.append({
                    "variable": "Y2",
                    "name": "Cashless Society",
                    "coefficient": coef,
                    "score": sc,
                    "influence": inf
                })

        combined = {}
        for inf in influences:
            v = inf["variable"]
            if v not in combined:
                combined[v] = {
                    "variable": v,
                    "name": inf["name"],
                    "total_influence": 0.0,
                    "current_score": inf["score"],
                    "components": []
                }
            combined[v]["total_influence"] += inf["influence"]
            combined[v]["components"].append({
                "coefficient": inf["coefficient"],
                "influence": inf["influence"]
            })

        combined_list = list(combined.values())
        combined_list.sort(key=lambda x: abs(x["total_influence"]), reverse=True)

        positive = [c for c in combined_list if c["total_influence"] > 0]
        positive.sort(key=lambda x: (x["total_influence"], -x["current_score"]), reverse=True)
        recommendation_3 = positive[:3] if len(positive) >= 3 else positive

        results[y_key] = {
            "name": y_names.get(y_key, y_key),
            "all_influences": combined_list,
            "top_3": combined_list[:3],
            "recommendation_3": recommendation_3
        }

    return results

def generate_pekerjaan_avg_chart(stats_pekerjaan):
    """Grouped bar chart dengan nama variabel + angka per bar"""
    try:
        if not stats_pekerjaan:
            return None

        names = []
        y1 = []
        y2 = []
        y3 = []

        for p in stats_pekerjaan:
            nm = p['jenis_pekerjaan']
            if len(nm) > 25:
                nm = nm.replace("/", "/\n") 
            names.append(nm)

            y1.append(float(p.get('avg_Y1') or 0))
            y2.append(float(p.get('avg_Y2') or 0))
            y3.append(float(p.get('avg_Y3') or 0))

        x = np.arange(len(names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(16, 9))

        bars1 = ax.bar(x - width, y1, width, label="Digitalisasi (Y1)", color="#3E8EDE")
        bars2 = ax.bar(x, y2, width, label="Cashless (Y2)", color="#34A853")
        bars3 = ax.bar(x + width, y3, width, label="Ekonomi (Y3)", color="#FBBC05")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')

        ax.set_ylabel("Rata-rata Skor", fontsize=12)
        ax.set_title("Rata-rata Skor per Jenis Pekerjaan", fontsize=16, fontweight='bold')
        ax.legend()

        def add_values(bars):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                        f"{h:.2f}", ha='center', fontsize=10, fontweight='bold')

        add_values(bars1)
        add_values(bars2)
        add_values(bars3)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        out = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return out

    except Exception as e:
        print("Error generate_pekerjaan_avg_chart:", e)
        return None

def generate_overall_bar_chart(overall_stats):
    """Bar chart Y1, Y2, Y3 untuk Statistik Keseluruhan"""
    try:
        if not overall_stats:
            return None

        labels = ["Digitalisasi (Y1)", "Cashless Society (Y2)", "Pemulihan Ekonomi (Y3)"]
        values = [
            overall_stats.get("avg_Y1", 0),
            overall_stats.get("avg_Y2", 0),
            overall_stats.get("avg_Y3", 0)
        ]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(labels, values, color=["#3E8EDE", "#FF7F00", "#34A853"])

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.03,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

        ax.set_ylabel("Rata-rata Skor", fontsize=12)
        ax.set_title("Statistik Keseluruhan (Y1, Y2, Y3)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=0)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130)
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)
        return encoded

    except Exception as e:
        print("Error generate_overall_bar_chart:", e)
        return None


def generate_pekerjaan_pie_chart(stats_pekerjaan):
    """Pie chart versi aman supaya label/teks tidak kepotong."""
    try:
        if not stats_pekerjaan:
            return None

        labels = []
        sizes = []
        for p in stats_pekerjaan:
            nm = str(p.get('jenis_pekerjaan', ''))
            if len(nm) > 25:
                nm = nm[:22] + "..."
            labels.append(nm)
            sizes.append(int(p.get('count', 0) or 0))

        banyak = len(labels)
        tampil_label_dari_pie = banyak <= 5

        fig, ax = plt.subplots(figsize=(12, 9))
        colors = plt.cm.Set3(np.linspace(0, 1, max(1, len(labels))))

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels if tampil_label_dari_pie else ['' for _ in labels],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1},
            textprops={'fontsize': 9, 'weight': 'normal'},
            pctdistance=0.85,
            labeldistance=1.07
        )

        plt.setp(autotexts, size=9, weight="bold", color="black")
        ax.set_title("Sebaran Jenis Pekerjaan", fontsize=14, fontweight='bold', pad=18)
        ax.axis('equal')

        if not tampil_label_dari_pie:
            ax.legend(
                wedges,
                labels,
                title="Jenis Pekerjaan",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=9,
                title_fontsize=10
            )

        try:
            fig.tight_layout()
        except Exception:
            pass

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=160, bbox_inches='tight', pad_inches=0.25, facecolor='white')
        buf.seek(0)
        out = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return out

    except Exception as e:
        print("Error generate_pekerjaan_pie_chart:", e)
        return None


def generate_distribution_chart(stats_wilayah):
    try:
        if not stats_wilayah:
            return None

        wilayah_names = [w['wilayah'] for w in stats_wilayah]
        y1_scores = [w['avg_Y1'] for w in stats_wilayah]
        y2_scores = [w['avg_Y2'] for w in stats_wilayah]
        y3_scores = [w['avg_Y3'] for w in stats_wilayah]

        x = np.arange(len(wilayah_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 8))
        bars1 = ax.bar(x - width, y1_scores, width, label='Digitalisasi (Y1)')
        bars2 = ax.bar(x, y2_scores, width, label='Cashless Society (Y2)')
        bars3 = ax.bar(x + width, y3_scores, width, label='Pemulihan Ekonomi (Y3)')

        ax.set_xticks(x)
        ax.set_xticklabels(wilayah_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylabel('Rata-rata Skor')
        ax.set_title('Rata-rata skor per Wilayah')
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        add_value_labels(bars1); add_value_labels(bars2); add_value_labels(bars3)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return chart_base64

    except Exception as e:
        print("Error generate_distribution_chart:", e)
        traceback.print_exc()
        return None

def generate_improvement_recommendations(analysis_results, sem_scores, mode="detail", consolidated=False):
    if consolidated:
        recommendations = []
        for y_key in ["Y1", "Y2", "Y3"]:
            y_data = analysis_results.get(y_key, {})
            y_name = y_data.get("name", y_key)
            y_score = sem_scores.get(y_key, 0)

            positive_influences = [
                inf for inf in y_data.get("all_influences", [])
                if inf["total_influence"] > 0
            ]
            positive_influences.sort(key=lambda x: x["total_influence"], reverse=True)
            top_3_positive = positive_influences[:3]

            improvement_strategies = []
            for inf in top_3_positive:
                strategy = {
                    "name": inf["name"],
                    "current_score": inf["current_score"],
                    "influence": inf["total_influence"],
                    "strategy": f"Tingkatkan {inf['name']} karena memiliki pengaruh positif tertinggi ({inf['total_influence']:.3f})"
                }
                improvement_strategies.append(strategy)

            recommendations.append({
                "y": y_name,
                "nilai": y_score,
                "kategori": score_to_category(y_score),
                "improvement_strategies": improvement_strategies
            })

        return recommendations

    recommendations = []

    for y_key in ["Y1", "Y2", "Y3"]:
        y_data = analysis_results.get(y_key, {})
        y_name = y_data.get("name", y_key)
        y_score = sem_scores.get(y_key, 0)

        positive_influences = [
            inf for inf in y_data.get("all_influences", [])
            if inf["total_influence"] > 0
        ]
        positive_influences.sort(key=lambda x: x["total_influence"], reverse=True)
        top_3_positive = positive_influences[:3]

        improvement_strategies = []
        short_recommendations = []

        for inf in top_3_positive:
            if mode == "detail":
                strategy = {
                    "name": inf["name"],
                    "current_score": inf["current_score"],
                    "influence": inf["total_influence"],
                    "strategy": f"Tingkatkan {inf['name']} karena memiliki pengaruh positif tertinggi ({inf['total_influence']:.3f})"
                }
                improvement_strategies.append(strategy)
            else: 
                short_recommendations.append(f"{inf['name']} (pengaruh: {inf['total_influence']:.3f})")

        if not improvement_strategies and y_key == "Y3" and mode == "detail":
            improvement_strategies.append({
                "name": "Digitalisasi Uang Elektronik",
                "current_score": sem_scores.get("Y1", 0),
                "influence": 0.2,
                "strategy": "Tingkatkan Digitalisasi Uang Elektronik untuk meningkatkan Pemulihan Ekonomi"
            })
            improvement_strategies.append({
                "name": "Cashless Society",
                "current_score": sem_scores.get("Y2", 0),
                "influence": 0.146,
                "strategy": "Tingkatkan Cashless Society untuk meningkatkan Pemulihan Ekonomi"
            })

        if mode == "detail":
            recommendations.append({
                "y": y_name,
                "nilai": y_score,
                "kategori": score_to_category(y_score),
                "improvement_strategies": improvement_strategies
            })
        else:
            recommendations.append({
                "y": y_name,
                "kategori": score_to_category(y_score),
                "short_recommendations": short_recommendations
            })

    return recommendations

def create_relationship_map(x_scores, wilayah_id, sem_scores, analysis_results):
    try:
        diagrams = {}
        y_names = {"Y1": "Digitalisasi Uang Elektronik", "Y2": "Cashless Society", "Y3": "Pemulihan Ekonomi"}
        sns.set_style("whitegrid")

        for y_key in ["Y1", "Y2", "Y3"]:
            all_influences = analysis_results.get(y_key, {}).get("all_influences", [])
            positive_influences = [inf for inf in all_influences if inf["total_influence"] > 0]
            positive_influences.sort(key=lambda x: x["total_influence"], reverse=True)

            top_positive = positive_influences[:3]

            if not top_positive:
                continue

            G = nx.DiGraph()
            Y_label = y_key
            G.add_node(Y_label)

            for inf in top_positive:
                G.add_node(inf["variable"])
                G.add_edge(inf["variable"], Y_label,
                          weight=abs(inf["total_influence"]),
                          label=f"{inf['total_influence']:.3f}")

            pos = nx.spring_layout(G, seed=42, k=1.5)
            fig, ax = plt.subplots(figsize=(10, 8))

            nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="#4ECDC4", alpha=0.9, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=[Y_label], node_size=3000,
                                 node_color="#FF6B6B", alpha=0.9, ax=ax)

            nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax)

            edges = G.edges(data=True)
            widths = [max(2, d['weight']*10) for (_,_,d) in edges]
            nx.draw_networkx_edges(G, pos, width=widths, edge_color='#2E8B57',
                                 arrowsize=25, alpha=0.7, ax=ax)

            edge_labels = {(u,v): d['label'] for u,v,d in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                       font_size=10, font_weight='bold', ax=ax)

            ax.set_title(f"Top 3 Variabel Positif - {y_names.get(y_key)}", fontsize=16, pad=20)
            ax.axis('off')

            ax.text(0.02, 0.98, f"Total Variabel Positif: {len(positive_influences)}",
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
            buf.seek(0)
            diagrams[y_key] = base64.b64encode(buf.getvalue()).decode("utf-8")
            plt.close(fig)

        return diagrams
    except Exception as e:
        print("Error create_relationship_map:", e)
        traceback.print_exc()
        return {}

def create_wilayah_sem_diagram(wilayah_id, avg_scores):
    try:
        sem_scores = calculate_sem_scores(avg_scores, wilayah_id)
        analysis_results = analyze_variable_influences(avg_scores, wilayah_id, sem_scores)

        diagrams = create_relationship_map(avg_scores, wilayah_id, sem_scores, analysis_results)
        return diagrams

    except Exception as e:
        print(f"Error create_wilayah_sem_diagram for wilayah {wilayah_id}:", e)
        traceback.print_exc()
        return {}

def calculate_wilayah_avg_scores():
    """Hitung rata-rata skor X per wilayah untuk analisis SEM"""
    conn = get_db_connection()
    if not conn:
        return {}

    try:
        cur = conn.cursor(dictionary=True)

        cur.execute(f"SELECT wilayah_id, x_scores FROM {DB_TABLE} WHERE x_scores IS NOT NULL")
        rows = cur.fetchall()

        wilayah_data = defaultdict(list)
        for row in rows:
            wid = row['wilayah_id']
            if wid and row['x_scores']:
                try:
                    x_scores = json.loads(row['x_scores'])
                    wilayah_data[wid].append(x_scores)
                except:
                    continue

        avg_scores_by_wilayah = {}
        for wid, scores_list in wilayah_data.items():
            if not scores_list:
                continue

            avg_scores = {}
            for x_var in [f"X{i}" for i in range(1, 11)] + ["M"]:
                total = sum(score.get(x_var, 0) for score in scores_list)
                avg_scores[x_var] = total / len(scores_list)

            avg_scores_by_wilayah[wid] = avg_scores

        conn.close()
        return avg_scores_by_wilayah

    except Exception as e:
        print("Error calculate_wilayah_avg_scores:", e)
        return {}

def generate_responden_per_wilayah_chart(stats_wilayah):
    try:
        if not stats_wilayah:
            return None

        wilayah_names = []
        counts = []

        for w in stats_wilayah:
            wilayah_names.append(w['wilayah'])
            counts.append(w['count'])

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(wilayah_names, counts, color='#4e73df', alpha=0.8)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}',
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

        ax.set_xlabel('Wilayah', fontsize=12)
        ax.set_ylabel('Jumlah Responden', fontsize=12)
        ax.set_title('Responden per Wilayah', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        return chart_base64

    except Exception as e:
        print("Error generate_responden_per_wilayah_chart:", e)
        return None

def try_mysql_connect(config_override=None):
    cfg = DB_CONFIG.copy()
    if config_override:
        cfg.update(config_override)
    try:
        conn = mysql.connector.connect(**cfg)
        return conn
    except mysql.connector.Error as err:
        print("MySQL connection error:", err)
        try:
            cfg2 = cfg.copy()
            cfg2["auth_plugin"] = "mysql_native_password"
            conn = mysql.connector.connect(**cfg2)
            return conn
        except Exception as e2:
            print("Fallback auth failed:", e2)
            return None
    except Exception as e:
        print("General MySQL exception:", e)
        return None

def get_db_connection():
    conn = None
    try:
        base_cfg = DB_CONFIG.copy()
        dbname = base_cfg.pop("database", None)
        conn = try_mysql_connect(base_cfg)
        if not conn:
            print("Initial connection without DB failed.")
            return None

        cur = conn.cursor()
        cur.execute("SHOW DATABASES")
        existing_dbs = [r[0] for r in cur.fetchall()]
        if dbname not in existing_dbs:
            cur.execute(f"CREATE DATABASE {dbname}")
            conn.commit()
            print(f"Database {dbname} created.")

        cur.close()
        conn.close()
        conn2 = try_mysql_connect(DB_CONFIG)
        if conn2:
            return conn2
        else:
            print("Reconnect to DB failed.")
            return None
    except Exception as e:
        print("Error get_db_connection:", e)
        traceback.print_exc()
        try:
            if conn and conn.is_connected():
                conn.close()
        except:
            pass
        return None

def get_db():
    conn = get_db_connection()
    if not conn:
        raise Exception("Tidak dapat terhubung ke database")
    return conn

def ensure_table_exists():
    conn = get_db_connection()
    if not conn:
        print("[ERROR] DB unavailable")
        return False

    try:
        cur = conn.cursor()

        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema=%s AND table_name=%s
        """, (DB_CONFIG["database"], DB_TABLE))

        exists = cur.fetchone()[0]

        if exists == 0:
            print("[INFO] Membuat tabel responden...")
            create_sql = f"""
            CREATE TABLE {DB_TABLE} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                nama VARCHAR(200),
                usia INT,
                jenis_pekerjaan VARCHAR(200) NOT NULL,
                wilayah VARCHAR(100) NOT NULL,
                wilayah_id VARCHAR(10),
                Y1 FLOAT,
                Y2 FLOAT,
                Y3 FLOAT,
                kategori_Y1 VARCHAR(50),
                kategori_Y2 VARCHAR(50),
                kategori_Y3 VARCHAR(50),
                x_scores TEXT,
                waktu_analisis DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            cur.execute(create_sql)
            conn.commit()
            print("[INFO] Tabel responden berhasil dibuat.")
        else:
            print("[INFO] Tabel responden sudah ada, memeriksa kolom...")

        cur.execute(f"""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s
        """, (DB_CONFIG["database"], DB_TABLE))

        existing_cols = {row[0] for row in cur.fetchall()}

        required = {
            "wilayah_id": "VARCHAR(10)",
            "x_scores": "TEXT",
            "kategori_Y1": "VARCHAR(50)",
            "kategori_Y2": "VARCHAR(50)",
            "kategori_Y3": "VARCHAR(50)"
        }

        for col in ALL_SCORE_COLUMNS:
            required[col] = "FLOAT"

        for col, col_type in required.items():
            if col not in existing_cols:
                print(f"[INFO] Menambahkan kolom: {col}")
                cur.execute(f"ALTER TABLE {DB_TABLE} ADD COLUMN {col} {col_type};")
                conn.commit()

        cur.close()
        conn.close()
        return True

    except Exception as e:
        print("[ERROR ensure_table_exists]:", e)
        return False

def save_result_to_db(payload):
    conn = get_db_connection()
    if not conn:
        print("DB unavailable - cannot save")
        return False

    try:
        cur = conn.cursor()
        columns = [
            "nama", "usia", "jenis_pekerjaan", "wilayah", "wilayah_id",
            "Y1", "Y2", "Y3",
            "kategori_Y1", "kategori_Y2", "kategori_Y3", "x_scores"
        ] + ALL_SCORE_COLUMNS

        col_sql = ", ".join(f"`{c}`" for c in columns)
        placeholders = ", ".join(["%s"] * len(columns))
        insert_sql = f"INSERT INTO {DB_TABLE} ({col_sql}) VALUES ({placeholders})"

        cur.execute(insert_sql, tuple(payload.get(c) for c in columns))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print("Error saving result to DB:", e)
        traceback.print_exc()
        try:
            cur.close()
        except:
            pass
        try:
            conn.close()
        except:
            pass
        return False

def calculate_descriptive_stats(ranking_method='avg'):
    conn = get_db_connection()
    if not conn:
        return {
            "overall": None,
            "stats_wilayah": [],
            "stats_pekerjaan": [],
            "kategori_stats": {},
        }

    try:
        cur = conn.cursor(dictionary=True)

        cur.execute(f"""
            SELECT
                wilayah,
                COUNT(*) as count,
                AVG(Y1) as avg_Y1,
                AVG(Y2) as avg_Y2,
                AVG(Y3) as avg_Y3,
                STDDEV(Y1) as std_Y1,
                STDDEV(Y2) as std_Y2,
                STDDEV(Y3) as std_Y3
            FROM {DB_TABLE}
            WHERE Y1 IS NOT NULL AND Y2 IS NOT NULL AND Y3 IS NOT NULL
            GROUP BY wilayah
        """)
        wilayah_rows = cur.fetchall()

        stats_wilayah = []
        for row in wilayah_rows:
            stats_wilayah.append({
                "wilayah": row["wilayah"],
                "count": row["count"],
                "avg_Y1": float(row["avg_Y1"]) if row["avg_Y1"] else 0,
                "avg_Y2": float(row["avg_Y2"]) if row["avg_Y2"] else 0,
                "avg_Y3": float(row["avg_Y3"]) if row["avg_Y3"] else 0,
                "std_Y1": float(row["std_Y1"] or 0),
                "std_Y2": float(row["std_Y2"] or 0),
                "std_Y3": float(row["std_Y3"] or 0),
            })

        cur.execute(f"""
            SELECT
                jenis_pekerjaan,
                COUNT(*) as count,
                AVG(Y1) as avg_Y1,
                AVG(Y2) as avg_Y2,
                AVG(Y3) as avg_Y3,
                STDDEV(Y1) as std_Y1,
                STDDEV(Y2) as std_Y2,
                STDDEV(Y3) as std_Y3
            FROM {DB_TABLE}
            WHERE Y1 IS NOT NULL AND Y2 IS NOT NULL AND Y3 IS NOT NULL
            GROUP BY jenis_pekerjaan
            ORDER BY COUNT(*) DESC
        """)
        pekerjaan_rows = cur.fetchall()

        stats_pekerjaan = []
        for row in pekerjaan_rows:
            stats_pekerjaan.append({
                "jenis_pekerjaan": row["jenis_pekerjaan"],
                "count": row["count"],
                "avg_Y1": float(row["avg_Y1"]) if row["avg_Y1"] else 0,
                "avg_Y2": float(row["avg_Y2"]) if row["avg_Y2"] else 0,
                "avg_Y3": float(row["avg_Y3"]) if row["avg_Y3"] else 0,
                "std_Y1": float(row["std_Y1"] or 0),
                "std_Y2": float(row["std_Y2"] or 0),
                "std_Y3": float(row["std_Y3"] or 0),
            })

        for w in stats_wilayah:
            a1 = w.get("avg_Y1", 0)
            a2 = w.get("avg_Y2", 0)
            a3 = w.get("avg_Y3", 0)

            if ranking_method == "Y1":
                rv = a1
            elif ranking_method == "Y2":
                rv = a2
            elif ranking_method == "Y3":
                rv = a3
            elif ranking_method == "max":
                rv = max(a1, a2, a3)
            else:
                rv = (a1 + a2 + a3) / 3

            w["ranking_value"] = rv

        stats_wilayah_sorted = sorted(stats_wilayah, key=lambda x: x["ranking_value"], reverse=True)

        kategori_stats = {}
        for y in ["Y1", "Y2", "Y3"]:
            cur.execute(f"""
                SELECT
                    COUNT(CASE WHEN {y} < 2.5 THEN 1 END) AS rendah,
                    COUNT(CASE WHEN {y} >= 2.5 AND {y} < 3.5 THEN 1 END) AS sedang,
                    COUNT(CASE WHEN {y} >= 3.5 THEN 1 END) AS tinggi,
                    COUNT(*) AS total
                FROM {DB_TABLE}
                WHERE {y} IS NOT NULL
            """)
            res = cur.fetchone()
            if res:
                kategori_stats[y] = {
                    "rendah": res.get("rendah", 0),
                    "sedang": res.get("sedang", 0),
                    "tinggi": res.get("tinggi", 0),
                    "total": res.get("total", 0),
                }

        cur.execute(f"""
            SELECT
                COUNT(*) AS total_count,
                AVG(Y1) AS avg1,
                AVG(Y2) AS avg2,
                AVG(Y3) AS avg3
            FROM {DB_TABLE}
            WHERE Y1 IS NOT NULL AND Y2 IS NOT NULL AND Y3 IS NOT NULL
        """)
        overall = cur.fetchone()
        overall_stats = {
            "total_count": overall["total_count"] if overall else 0,
            "avg_Y1": float(overall["avg1"]) if overall and overall["avg1"] else 0,
            "avg_Y2": float(overall["avg2"]) if overall and overall["avg2"] else 0,
            "avg_Y3": float(overall["avg3"]) if overall and overall["avg3"] else 0,
        }

        conn.close()

        return {
            "overall": overall_stats,
            "stats_wilayah": stats_wilayah_sorted,
            "kategori_stats": kategori_stats,
            "stats_pekerjaan": stats_pekerjaan,
        }

    except Exception as e:
        print("Error calculate_descriptive_stats:", e)
        traceback.print_exc()
        try:
            conn.close()
        except:
            pass
        return {
            "overall": None,
            "stats_wilayah": [],
            "kategori_stats": {},
            "stats_pekerjaan": [],
        }

def calculate_wilayah_sem_analysis():
    conn = get_db_connection()
    if not conn:
        return {}

    try:
        cur = conn.cursor(dictionary=True)

        cur.execute(f"""
            SELECT
                wilayah,
                wilayah_id,
                COUNT(*) as count,
                AVG(Y1) as avg_Y1,
                AVG(Y2) as avg_Y2,
                AVG(Y3) as avg_Y3
            FROM {DB_TABLE}
            WHERE Y1 IS NOT NULL AND Y2 IS NOT NULL AND Y3 IS NOT NULL
            GROUP BY wilayah, wilayah_id
        """)

        wilayah_rows = cur.fetchall()
        results = {}
        for row in wilayah_rows:
            wid = row.get('wilayah_id') or str(row.get('wilayah'))
            results[wid] = {
                'wilayah': row.get('wilayah'),
                'count': int(row.get('count') or 0),
                'avg_Y1': float(row.get('avg_Y1') or 0),
                'avg_Y2': float(row.get('avg_Y2') or 0),
                'avg_Y3': float(row.get('avg_Y3') or 0),
            }

        cur.close()
        conn.close()
        return results

    except Exception as e:
        print("Error calculate_wilayah_sem_analysis:", e)
        traceback.print_exc()
        try:
            conn.close()
        except:
            pass
        return {}


def generate_insights(stats):
    insights = []
    try:
        if not stats:
            return insights

        wilayah = stats.get("stats_wilayah", [])
        pekerjaan = stats.get("stats_pekerjaan", [])
        overall = stats.get("overall", {})

        if wilayah:
            try:
                tertinggi_y2 = max(wilayah, key=lambda x: x.get("avg_Y2", 0))
                terendah_y2 = min(wilayah, key=lambda x: x.get("avg_Y2", 0))
                tertinggi_y1 = max(wilayah, key=lambda x: x.get("avg_Y1", 0))
                tertinggi_y3 = max(wilayah, key=lambda x: x.get("avg_Y3", 0))

                insights.append(f"Wilayah dengan Cashless Society tertinggi: {tertinggi_y2['wilayah']} (skor: {tertinggi_y2['avg_Y2']:.2f})")
                insights.append(f"Wilayah dengan Digitalisasi tertinggi: {tertinggi_y1['wilayah']} (skor: {tertinggi_y1['avg_Y1']:.2f})")
                insights.append(f"Wilayah dengan Pemulihan Ekonomi tertinggi: {tertinggi_y3['wilayah']} (skor: {tertinggi_y3['avg_Y3']:.2f})")
                insights.append(f"Wilayah yang perlu perhatian khusus (Cashless terendah): {terendah_y2['wilayah']} (skor: {terendah_y2['avg_Y2']:.2f})")
            except Exception:
                pass
        if pekerjaan:
            try:
                pekerjaan_tertinggi_y2 = max(pekerjaan, key=lambda x: x.get("avg_Y2", 0))
                pekerjaan_terbanyak = max(pekerjaan, key=lambda x: x.get("count", 0))
                insights.append(f"Pekerjaan dengan adopsi Cashless tertinggi: {pekerjaan_tertinggi_y2['jenis_pekerjaan']} (skor: {pekerjaan_tertinggi_y2['avg_Y2']:.2f})")
                insights.append(f"Kelompok pekerjaan terbanyak: {pekerjaan_terbanyak['jenis_pekerjaan']} ({pekerjaan_terbanyak['count']} responden)")
            except Exception:
                pass
        if overall and overall.get("total_count", 0) > 0:
            avg_y2 = overall.get("avg_Y2", 0)
            if avg_y2 >= 3.5:
                status = "TINGGI"
            elif avg_y2 >= 2.5:
                status = "SEDANG"
            else:
                status = "RENDAH"
            insights.append(f"Rata-rata tingkat Cashless Society secara keseluruhan: {status} (skor: {avg_y2:.2f})")

    except Exception as e:
        print("Error generate_insights:", e)
        traceback.print_exc()

    return insights

@app.route("/kuisioner", methods=["GET", "POST"])
def kuisioner():
    if request.method == "POST":
        session["nama"] = request.form.get("nama")
        session["usia"] = request.form.get("usia")
        session["jenis_pekerjaan"] = request.form.get("jenis_pekerjaan")
        session["wilayah"] = request.form.get("wilayah")

        return redirect(url_for("kuisioner_pertanyaan"))

    return render_template(
        "kuisioner.html",
        app_title=APP_TITLE,
        wilayah_options=WILAYAH_MAP.items(),
        jenis_pekerjaan_options=JENIS_PEKERJAAN_OPTIONS
    )

@app.route("/kuisioner_pertanyaan")
def kuisioner_pertanyaan():
    if not session.get("nama"):
        return redirect(url_for("kuisioner"))

    return render_template(
        "kuisioner_pertanyaan.html",
        questions=QUESTIONS_DETAILED,
        likert_options=LIKERT_OPTIONS,
        nama=session["nama"],
        wilayah_name=WILAYAH_MAP.get(session["wilayah"]),
        app_title=APP_TITLE
    )

@app.route("/submit_kuisioner", methods=["POST"])
def submit_kuisioner():
    try:
        wid = session.get("wilayah")

        if wid not in WILAYAH_MAP:
            flash("Sesi berakhir. Silakan isi data diri kembali.", "warning")
            return redirect(url_for("kuisioner"))

        answers = {}
        for var in QUESTIONS_DETAILED:
            for item in QUESTIONS_DETAILED[var]["items"]:
                val = request.form.get(item)
                answers[item] = map_answer_to_weight(val)

        x_scores, item_scores = calculate_weighted_scores(answers, wid)
        sem_scores = calculate_sem_scores(x_scores, wid)

        kategori_Y1 = score_to_category(sem_scores["Y1"])
        kategori_Y2 = score_to_category(sem_scores["Y2"])
        kategori_Y3 = score_to_category(sem_scores["Y3"])

        analysis_results = analyze_variable_influences(x_scores, wid, sem_scores)
        improvement_recommendations = generate_improvement_recommendations(
            analysis_results, sem_scores, mode="short"
        )

        payload = {
            "nama": session.get("nama"),
            "usia": int(session.get("usia")) if session.get("usia") else None,
            "jenis_pekerjaan": session.get("jenis_pekerjaan"),
            "wilayah": WILAYAH_MAP.get(wid),
            "wilayah_id": wid,
            "Y1": sem_scores["Y1"],
            "Y2": sem_scores["Y2"],
            "Y3": sem_scores["Y3"],
            "kategori_Y1": kategori_Y1,
            "kategori_Y2": kategori_Y2,
            "kategori_Y3": kategori_Y3,
            "x_scores": json.dumps(x_scores),
        }

        for col in SCORE_COLUMNS_X:
            payload[col] = round(float(x_scores.get(col, 0) or 0), 4)

        for y_var, col in SCORE_COLUMNS_Y_SURVEY.items():
            payload[col] = round(float(x_scores.get(y_var, 0) or 0), 4)

        save_result_to_db(payload)

        return render_template(
            "hasil_kuisioner.html",
            nama=session.get("nama"),
            usia=session.get("usia"),
            pekerjaan=session.get("jenis_pekerjaan"),
            wilayah_name=WILAYAH_MAP.get(wid),
            predictions=sem_scores,
            analysis_results=analysis_results,
            item_scores=item_scores,
            kategori_Y1=kategori_Y1,
            kategori_Y2=kategori_Y2,
            kategori_Y3=kategori_Y3,
            improvement_recommendations=improvement_recommendations,
            timestamp=datetime.now().strftime("%d/%m/%Y %H:%M"),
            app_title=APP_TITLE,
        )

    except Exception as e:
        print("Error submit_kuisioner:", e)
        traceback.print_exc()
        flash("Terjadi error.", "danger")
        return redirect(url_for("kuisioner"))

def check_admin_auth():
    return session.get("admin_logged_in") is True

def admin_login_required(f):
    @wraps(f)
    def d(*a, **kw):
        if not check_admin_auth():
            return redirect(url_for("admin_login"))
        return f(*a, **kw)
    return d

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form.get("username") == "Admin" and request.form.get("password") == "AcelSkripsi":
            session["admin_logged_in"] = True
            return redirect(url_for("data_responden_admin")) 
        else:
            flash("Login gagal", "danger")

    return render_template("admin_login.html", app_title=APP_TITLE)

@app.route("/admin_logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("index"))

@app.route("/admin/statistik")
@admin_login_required
def clustering_result():
    try:
        selected_wilayah = request.args.get("wilayah", "all")
        selected_ranking = request.args.get("ranking", "avg")

        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        if selected_wilayah == "all":
            cur.execute(f"SELECT * FROM {DB_TABLE} ORDER BY waktu_analisis DESC")
        else:
            wname = WILAYAH_MAP.get(selected_wilayah, selected_wilayah)
            cur.execute(
                f"SELECT * FROM {DB_TABLE} WHERE wilayah=%s ORDER BY waktu_analisis DESC",
                (wname,)
            )

        data = cur.fetchall()
        cur.close()
        conn.close()

        stats = calculate_descriptive_stats(selected_ranking)

        sw = stats["stats_wilayah"]
        ov = stats["overall"] or {}
        chart_data = {
            "wilayah": [w["wilayah"] for w in sw],
            "count":   [w["count"] for w in sw],
            "avg_Y1":  [round(w["avg_Y1"], 2) for w in sw],
            "avg_Y2":  [round(w["avg_Y2"], 2) for w in sw],
            "avg_Y3":  [round(w["avg_Y3"], 2) for w in sw],
            "overall": [
                round(ov.get("avg_Y1", 0), 2),
                round(ov.get("avg_Y2", 0), 2),
                round(ov.get("avg_Y3", 0), 2),
            ],
        }

        return render_template(
            "clustering_result.html",
            app_title=APP_TITLE,
            data=data,
            stats=stats,
            selected_wilayah=selected_wilayah,
            selected_ranking=selected_ranking,
            wilayah_options=WILAYAH_MAP,
            score_to_category=score_to_category,
            chart_data=chart_data,
        )

    except Exception as e:
        print("ERROR halaman statistik:", e)
        traceback.print_exc()
        flash("Error mengambil data", "danger")
        return redirect(url_for("index"))

@app.route("/detail_responden/<int:responden_id>")
@admin_login_required
def detail_responden(responden_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(f"SELECT * FROM {DB_TABLE} WHERE id=%s", (responden_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            flash("Data tidak ditemukan", "danger")
            return redirect(url_for("clustering_result"))

        x_scores = json.loads(row["x_scores"]) if row.get("x_scores") else {}
        sem_scores = {"Y1": row["Y1"], "Y2": row["Y2"], "Y3": row["Y3"]}

        analysis_results = analyze_variable_influences(x_scores, row["wilayah_id"], sem_scores)
        improvement_recommendations = generate_improvement_recommendations(
            analysis_results, sem_scores, mode="detail"
        )
        maps = create_relationship_map(x_scores, row["wilayah_id"], sem_scores, analysis_results)

        return render_template(
            "detail_responden.html",
            row=row,
            x_scores=x_scores,
            sem=sem_scores,
            predictions=sem_scores,
            analysis_results=analysis_results,
            improvement_recommendations=improvement_recommendations,
            relationship_maps=maps,
            score_to_category=score_to_category,
            app_title=APP_TITLE,
        )

    except Exception as e:
        print("Error detail_responden:", e)
        traceback.print_exc()
        flash("Error memproses data", "danger")
        return redirect(url_for("clustering_result"))

@app.route("/")
def index():
    return render_template(
        "index.html",
        app_title=APP_TITLE,
        wilayah_options=list(WILAYAH_MAP.items()),
    )

@app.route("/admin/analisis-peta-hubungan")
@admin_login_required
def analisis_peta_hubungan():
    return render_template(
        "analisis_peta_hubungan.html",
        sem_coeffs=SEM_COEFFS,
        wilayah_map=WILAYAH_MAP,
        x_names=X_VAR_NAME,
        y_names=PATH_VAR_NAME,
        app_title=APP_TITLE,
    )

@app.route("/subscribe", methods=["POST"])
def subscribe():
    phone = request.form.get("phone")
    if phone:
        flash("Terima kasih telah berlangganan! Kami akan menghubungi Anda segera.", "success")
    else:
        flash("Nomor telepon diperlukan.", "danger")
    return redirect(url_for("index"))

@app.route("/admin/data-responden")
@admin_login_required
def data_responden_admin():
    wilayah_filter = request.args.get("wilayah", "all")
    per_page = int(request.args.get("per_page", "10"))
    page = int(request.args.get("page", "1"))

    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    if wilayah_filter == "all":
        cur.execute(f"SELECT COUNT(*) as total FROM {DB_TABLE}")
    else:
        nm = WILAYAH_MAP.get(wilayah_filter, wilayah_filter)
        cur.execute(f"SELECT COUNT(*) as total FROM {DB_TABLE} WHERE wilayah=%s", (nm,))

    total_count = cur.fetchone()["total"]
    total_pages = (total_count + per_page - 1) // per_page

    offset = (page - 1) * per_page
    if wilayah_filter == "all":
        cur.execute(f"SELECT * FROM {DB_TABLE} ORDER BY id DESC LIMIT %s OFFSET %s", (per_page, offset))
    else:
        nm = WILAYAH_MAP.get(wilayah_filter, wilayah_filter)
        cur.execute(f"SELECT * FROM {DB_TABLE} WHERE wilayah=%s ORDER BY id DESC LIMIT %s OFFSET %s", (nm, per_page, offset))

    data = cur.fetchall()
    conn.close()

    return render_template(
        "data_responden_admin.html",
        data=data,
        wilayah_options=WILAYAH_MAP,
        selected_wilayah=wilayah_filter,
        per_page=per_page,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
        app_title=APP_TITLE,
    )

def backfill_score_columns():
    conn = get_db_connection()
    if not conn:
        return

    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT id, x_scores FROM {DB_TABLE} "
            f"WHERE x_scores IS NOT NULL AND X1 IS NULL"
        )
        rows = cur.fetchall()

        updated = 0
        for rid, xs in rows:
            try:
                data = json.loads(xs)
            except Exception:
                continue

            values = {c: data.get(c) for c in SCORE_COLUMNS_X}
            for y_var, col in SCORE_COLUMNS_Y_SURVEY.items():
                values[col] = data.get(y_var)

            set_sql = ", ".join(f"`{c}`=%s" for c in values)
            cur.execute(
                f"UPDATE {DB_TABLE} SET {set_sql} WHERE id=%s",
                (*values.values(), rid)
            )
            updated += 1

        conn.commit()
        cur.close()
        conn.close()
        if updated:
            print(f"[INFO] Backfill kolom skor untuk {updated} responden lama.")

    except Exception as e:
        print("Error backfill_score_columns:", e)

if __name__ == "__main__":
    print("Initializing database table...")
    ensure_table_exists()
    backfill_score_columns()
    print("Starting Flask app...")
    app.run(debug=True, host="0.0.0.0", port=5000)