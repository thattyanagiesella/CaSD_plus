# ============================
# app.py — REVISI LENGKAP (PART 1/3)
# ============================

import os
import io
import base64
import traceback
import json
from datetime import datetime
from functools import wraps
from collections import defaultdict

# Use non-interactive matplotlib backend for server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Basic plotting libs
import networkx as nx
import seaborn as sns

import numpy as np
import pandas as pd

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, session, jsonify
)

# ML libs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# MySQL
import mysql.connector
from mysql.connector import errorcode

# ---------------------------
# APP / STORAGE CONFIG
# ---------------------------
app = Flask(__name__, template_folder="templates", static_folder="public")
app.secret_key = os.environ.get("FLASK_SECRET", "super-secret-key")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/generated", exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------
# DATABASE CONFIG
# ---------------------------
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASS", "acel"),
    "database": os.environ.get("DB_NAME", "cashless_db"),
    "raise_on_warnings": True
}
DB_TABLE = "responden"

# ---------------------------
# APP SETTINGS / CONSTANTS
# ---------------------------
APP_TITLE = "CaSD (Cashless Society Detector)"

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
    "STS": 1,
    "TS": 2,
    "RAGU-RAGU": 3,
    "S": 4,
    "SS": 5
}
LIKERT_OPTIONS = ["STS", "TS", "R", "S", "SS"]  # RAGU-RAGU diganti menjadi R

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

# QUESTIONS (X1..X10) with item weights
QUESTIONS_DETAILED = {
    "X1": {
        "name": "Faktor Psikologi",
        "items": {
            "X1.1": {"text": "Saya termotivasi menggunakan mobile banking karena mudah diakses", "weight": 0.3},
            "X1.2": {"text": "Saya merasa mobile banking mempercepat proses transaksi", "weight": 0.4},
            "X1.3": {"text": "Saya perlu beradaptasi untuk menggunakan mobile banking dalam transaksi sehari-hari", "weight": 0.3}
        }
    },
    "X2": {
        "name": "Faktor Sosial Budaya",
        "items": {
            "X2.1": {"text": "Saya menggunakan mobile banking karena rekomendasi dari teman", "weight": 0.4},
            "X2.2": {"text": "Saya menggunakan mobile banking karena rekomendasi dari keluarga", "weight": 0.3},
            "X2.3": {"text": "Saya menggunakan mobile banking karena mengikuti tren masa kini", "weight": 0.3}
        }
    },
    "X3": {
        "name": "Faktor Pribadi",
        "items": {
            "X3.1": {"text": "Saya menggunakan mobile banking karena sesuai dengan kebutuhan di masa sekarang", "weight": 0.4},
            "X3.2": {"text": "Saya menggunakan mobile banking karena memudahkan aktivitas", "weight": 0.3},
            "X3.3": {"text": "Saya menggunakan mobile banking untuk memudahkan pengelolaan keuangan", "weight": 0.3}
        }
    },
    "X4": {
        "name": "Product",
        "items": {
            "X4.1": {"text": "Mobile banking menawarkan fitur yang lengkap", "weight": 0.5},
            "X4.2": {"text": "Mobile banking selalu memperbarui fitur secara berkala", "weight": 0.5}
        }
    },
    "X5": {
        "name": "Price",
        "items": {
            "X5.1": {"text": "Biaya layanan mobile banking terasa wajar", "weight": 0.6},
            "X5.2": {"text": "Mobile banking menawarkan biaya yang kompetitif", "weight": 0.4}
        }
    },
    "X6": {
        "name": "Promotion",
        "items": {
            "X6.1": {"text": "Promosi mobile banking menarik perhatian saya", "weight": 0.5},
            "X6.2": {"text": "Mobile banking memberikan reward yang menarik", "weight": 0.5}
        }
    },
    "X7": {
        "name": "Place",
        "items": {
            "X7.1": {"text": "Lokasi kantor bank mudah diakses", "weight": 0.5},
            "X7.2": {"text": "Saya merasa nyaman menggunakan mobile banking", "weight": 0.5}
        }
    },
    "X8": {
        "name": "People",
        "items": {
            "X8.1": {"text": "Customer service mobile banking responsif", "weight": 0.5},
            "X8.2": {"text": "Customer service memberikan informasi yang jelas", "weight": 0.5}
        }
    },
    "X9": {
        "name": "Process",
        "items": {
            "X9.1": {"text": "Proses pendaftaran mobile banking mudah", "weight": 0.4},
            "X9.2": {"text": "Proses transaksi mobile banking cepat dan aman", "weight": 0.6}
        }
    },
    "X10": {
        "name": "Physical Evidence",
        "items": {
            "X10.1": {"text": "Tampilan aplikasi mobile banking menarik", "weight": 0.4},
            "X10.2": {"text": "Fitur mobile banking mudah digunakan", "weight": 0.6}
        }
    }
}

# SEM Coefficients per wilayah
SEM_COEFFS = {
    "1": {
        "Y1": {"X1": 0.369, "X2": 0.226, "X3": 0.100, "X4": 0.144, "X5": 0.072, "X6": 0.092, "X7": 0.184, "X8": 0.199, "X9": -0.028, "X10": -0.022},
        "Y2": {"X1": -0.104, "X2": -0.115, "X3": 0.282, "X4": 0.110, "X5": 0.121, "X6": 0.194, "X7": -0.010, "X8": 0.137, "X9": 0.106, "X10": 0.115, "Y1": 0.057},
        "Y3": {"Y1": 0.200, "Y2": 0.146}
    },
    "2": {
        "Y1": {"X1": 0.054, "X2": 0.234, "X3": 0.089, "X4": 0.128, "X5": 0.292, "X6": 0.080, "X7": 0.055, "X8": 0.009, "X9": 0.162, "X10": -0.198},
        "Y2": {"X1": 0.227, "X2": -0.110, "X3": -0.084, "X4": 0.157, "X5": 0.063, "X6": 0.135, "X7": 0.067, "X8": -0.080, "X9": 0.011, "X10": -0.038, "Y1": 0.187},
        "Y3": {"Y1": 0.146, "Y2": 0.141}
    },
    "3": {
        "Y1": {"X1": -0.025, "X2": 0.076, "X3": 0.134, "X4": 0.110, "X5": 0.030, "X6": 0.200, "X7": 0.159, "X8": 0.113, "X9": -0.211, "X10": 0.148},
        "Y2": {"X1": 0.111, "X2": 0.108, "X3": 0.174, "X4": -0.100, "X5": 0.169, "X6": 0.224, "X7": 0.112, "X8": 0.068, "X9": 0.190, "X10": 0.010, "Y1": 0.110},
        "Y3": {"Y1": 0.245, "Y2": 0.229}
    },
    "4": {
        "Y1": {"X1": 0.032, "X2": 0.182, "X3": -0.074, "X4": 0.345, "X5": 0.092, "X6": 0.033, "X7": -0.051, "X8": 0.121, "X9": 0.176, "X10": -0.116},
        "Y2": {"X1": -0.199, "X2": -0.149, "X3": 0.047, "X4": -0.067, "X5": -0.061, "X6": 0.102, "X7": 0.274, "X8": 0.224, "X9": 0.020, "X10": 0.352, "Y1": 0.021},
        "Y3": {"Y1": 0.230, "Y2": 0.035}
    },
    "5": {
        "Y1": {"X1": -0.019, "X2": 0.055, "X3": 0.332, "X4": 0.130, "X5": 0.147, "X6": 0.142, "X7": 0.022, "X8": 0.178, "X9": 0.181, "X10": 0.110},
        "Y2": {"X1": 0.010, "X2": -0.043, "X3": 0.143, "X4": 0.168, "X5": 0.094, "X6": 0.178, "X7": 0.053, "X8": 0.100, "X9": 0.087, "X10": 0.153, "Y1": 0.045},
        "Y3": {"Y1": 0.335, "Y2": 0.287}
    }
}

# Global runtime storage
global_clusters = {}
global_models = {}

# ---------------------------
# UTIL FUNCTIONS
# ---------------------------
def map_answer_to_weight(ans):
    """Map jawaban ke angka; toleran terhadap format input."""
    if ans is None:
        return 0
    key = str(ans).strip().upper()
    if key in ("N", "R"):  # Tambahkan mapping untuk "R"
        key = "RAGU-RAGU"
    return LIKERT_SCALE.get(key, 0)

def calculate_weighted_scores(answers):
    """
    Dari jawaban per item (dictionary item_code -> numeric), hitung skor per X (X1..X10)
    menggunakan bobot tiap item. Mengembalikan (weighted_scores, item_scores_detail).
    """
    weighted_scores = {}
    item_scores = {}
    if not answers:
        return {}, {}

    for var, data in QUESTIONS_DETAILED.items():
        total_score = 0.0
        total_weight = 0.0
        item_scores[var] = {}
        for item_code, item_data in data["items"].items():
            score = answers.get(item_code, 0)
            if score is None:
                score = 0
            weight = float(item_data.get("weight", 1))
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
    """
    Hitung Y1,Y2,Y3 berdasarkan SEM_COEFFS wilayah.
    Hasil dinormalisasi ke range ~ [1,5] agar konsisten dengan skala likert.
    """
    if not x_scores:
        return {"Y1": 3.0, "Y2": 3.0, "Y3": 3.0}

    try:
        wid = str(wilayah_id)
        coeffs = SEM_COEFFS.get(wid, {})

        # Y1
        y1_raw = 0.0
        for x_var, coef in coeffs.get("Y1", {}).items():
            if x_var.startswith("X"):
                y1_raw += coef * float(x_scores.get(x_var, 0) or 0)

        # Y2 (memperhitungkan Y1 jika coefficient ada)
        y2_raw = 0.0
        for var, coef in coeffs.get("Y2", {}).items():
            if var == "Y1":
                y2_raw += coef * y1_raw
            elif var.startswith("X"):
                y2_raw += coef * float(x_scores.get(var, 0) or 0)

        # Y3
        y3_raw = 0.0
        for var, coef in coeffs.get("Y3", {}).items():
            if var == "Y1":
                y3_raw += coef * y1_raw
            elif var == "Y2":
                y3_raw += coef * y2_raw

        # Normalisasi: gunakan sum(abs(coef))*max_score sebagai pembagi
        def normalize(raw, coef_map):
            if not coef_map:
                return 3.0
            denom = sum(abs(c) for c in coef_map.values()) * 5.0
            if denom == 0:
                return 3.0
            # scale raw/denom to [1,5] with center at 3
            scaled = 1.0 + (raw / denom) * 4.0
            return round(max(1.0, min(5.0, scaled)), 2)

        y1 = normalize(y1_raw, coeffs.get("Y1", {}))
        y2 = normalize(y2_raw, coeffs.get("Y2", {}))
        y3 = normalize(y3_raw, coeffs.get("Y3", {}))

        return {"Y1": y1, "Y2": y2, "Y3": y3}

    except Exception as e:
        print("Error calculate_sem_scores:", e)
        traceback.print_exc()
        return {"Y1": 3.0, "Y2": 3.0, "Y3": 3.0}

def score_to_category(score):
    """Konversi skor numeric ke kategori: Rendah (<2.5), Sedang [2.5-3.5), Tinggi >=3.5"""
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

# ============================
# app.py — REVISI LENGKAP (PART 2/3)
# ============================

def analyze_variable_influences(x_scores, wilayah_id, sem_scores):
    """
    Hitung pengaruh tiap variabel X terhadap Y berdasarkan koefisien SEM.
    Mengembalikan struktur terurut (by absolute influence).
    """
    wid = str(wilayah_id)
    coeffs = SEM_COEFFS.get(wid, {})
    results = {}
    y_names = {"Y1": "Digitalisasi Uang Elektronik", "Y2": "Cashless Society", "Y3": "Pemulihan Ekonomi"}

    for y_key in ["Y1", "Y2", "Y3"]:
        y_coeffs = coeffs.get(y_key, {})
        influences = []
        for var, coef in y_coeffs.items():
            if var.startswith("X"):
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

        # combine by variable (sum influences)
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

        # recommendation: pick positive influences sorted by (influence, -score)
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
                nm = nm.replace("/", "/\n")  # Auto wrap
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

        # Tambah angka di atas bar
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

        # Tambahkan label nilai di atas bar
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
    """Pie chart yang diperbaiki agar tidak kepotong"""
    try:
        if not stats_pekerjaan:
            return None
        
        labels = []
        sizes = []

        for p in stats_pekerjaan:
            nm = p['jenis_pekerjaan']
            # Potong teks yang terlalu panjang
            if len(nm) > 25:
                nm = nm[:22] + "..."
            labels.append(nm)
            sizes.append(p['count'])

        # Ukuran figure lebih kecil
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Warna yang lebih baik
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        # Buat pie chart dengan parameter yang diperbaiki
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1},
            textprops={'fontsize': 10, 'weight': 'normal'},  # Font lebih kecil
            pctdistance=0.85,
            labeldistance=1.05  # Jarak label dari pie
        )

        # Atur autotext (persentase)
        plt.setp(autotexts, size=9, weight="bold", color="black")
        
        # Title
        ax.set_title("Sebaran Jenis Pekerjaan", fontsize=14, fontweight='bold', pad=20)
        
        # Equal aspect ratio
        ax.axis('equal')

        # Tambahkan legend di luar pie chart jika terlalu banyak item
        if len(labels) > 5:
            ax.legend(wedges, labels,
                     title="Jenis Pekerjaan",
                     loc="center left",
                     bbox_to_anchor=(1, 0, 0.5, 1),
                     fontsize=9)
            # Hapus labels langsung dari pie
            for text in texts:
                text.set_text('')

        # Adjust layout
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        out = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return out

    except Exception as e:
        print("Error generate_pekerjaan_pie_chart:", e)
        return None


def generate_distribution_chart(stats_wilayah):
    """Generate bar chart untuk distribusi kategori per wilayah"""
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
    """Generate rekomendasi peningkatan berdasarkan analisis pengaruh variabel"""
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

        # Ambil 3 variabel positif dengan pengaruh tertinggi
        positive_influences = [
            inf for inf in y_data.get("all_influences", [])
            if inf["total_influence"] > 0
        ]
        positive_influences.sort(key=lambda x: x["total_influence"], reverse=True)
        top_3_positive = positive_influences[:3]

        # Generate rekomendasi berdasarkan mode
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
            else:  # mode singkat
                short_recommendations.append(f"{inf['name']} (pengaruh: {inf['total_influence']:.3f})")

        # For Y3 (Pemulihan Ekonomi), ensure at least default strategies if none
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
    """
    Create simple network diagrams (base64 PNG) for top-3 POSITIVE influences per Y.
    Returns dict { 'Y1': base64_png, ... }
    """
    try:
        diagrams = {}
        y_names = {"Y1": "Digitalisasi Uang Elektronik", "Y2": "Cashless Society", "Y3": "Pemulihan Ekonomi"}
        sns.set_style("whitegrid")

        for y_key in ["Y1", "Y2", "Y3"]:
            # Hanya ambil variabel dengan pengaruh POSITIF dan urutkan dari tertinggi
            all_influences = analysis_results.get(y_key, {}).get("all_influences", [])
            positive_influences = [inf for inf in all_influences if inf["total_influence"] > 0]
            positive_influences.sort(key=lambda x: x["total_influence"], reverse=True)
            
            # Ambil maksimal 3 variabel positif tertinggi
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
            
            # Node styling
            nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="#4ECDC4", alpha=0.9, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=[Y_label], node_size=3000, 
                                 node_color="#FF6B6B", alpha=0.9, ax=ax)
            
            # Label styling
            nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax)
            
            # Edge styling - hanya hijau untuk positif
            edges = G.edges(data=True)
            widths = [max(2, d['weight']*10) for (_,_,d) in edges]
            nx.draw_networkx_edges(G, pos, width=widths, edge_color='#2E8B57', 
                                 arrowsize=25, alpha=0.7, ax=ax)
            
            # Edge labels
            edge_labels = {(u,v): d['label'] for u,v,d in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                       font_size=10, font_weight='bold', ax=ax)

            ax.set_title(f"Top 3 Variabel Positif - {y_names.get(y_key)}", fontsize=16, pad=20)
            ax.axis('off')
            
            # Tambahkan legend
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
    """
    Buat diagram SEM untuk wilayah tertentu berdasarkan rata-rata skor
    """
    try:
        # Hitung SEM scores berdasarkan rata-rata
        sem_scores = calculate_sem_scores(avg_scores, wilayah_id)
        analysis_results = analyze_variable_influences(avg_scores, wilayah_id, sem_scores)
        
        # Buat diagram hubungan
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
        
        # Ambil semua data dengan x_scores
        cur.execute(f"SELECT wilayah_id, x_scores FROM {DB_TABLE} WHERE x_scores IS NOT NULL")
        rows = cur.fetchall()
        
        # Group by wilayah_id dan hitung rata-rata
        wilayah_data = defaultdict(list)
        for row in rows:
            wid = row['wilayah_id']
            if wid and row['x_scores']:
                try:
                    x_scores = json.loads(row['x_scores'])
                    wilayah_data[wid].append(x_scores)
                except:
                    continue
        
        # Hitung rata-rata untuk setiap wilayah
        avg_scores_by_wilayah = {}
        for wid, scores_list in wilayah_data.items():
            if not scores_list:
                continue
                
            avg_scores = {}
            # Hitung rata-rata untuk setiap variabel X
            for x_var in [f"X{i}" for i in range(1, 11)]:
                total = sum(score.get(x_var, 0) for score in scores_list)
                avg_scores[x_var] = total / len(scores_list)
            
            avg_scores_by_wilayah[wid] = avg_scores
        
        conn.close()
        return avg_scores_by_wilayah
        
    except Exception as e:
        print("Error calculate_wilayah_avg_scores:", e)
        return {}
    
def generate_responden_per_wilayah_chart(stats_wilayah):
    """Bar chart untuk total responden per wilayah - VERSI AMAN"""
    try:
        if not stats_wilayah:
            return None

        wilayah_names = []
        counts = []

        for w in stats_wilayah:
            wilayah_names.append(w['wilayah'])
            counts.append(w['count'])

        # Buat bar chart sederhana
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(wilayah_names, counts, color='#4e73df', alpha=0.8)

        # Tambahkan nilai di atas bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', 
                   ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')

        # Styling sederhana
        ax.set_xlabel('Wilayah', fontsize=12)
        ax.set_ylabel('Jumlah Responden', fontsize=12)
        ax.set_title('Responden per Wilayah', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Simpan chart
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        return chart_base64

    except Exception as e:
        print("Error generate_responden_per_wilayah_chart:", e)
        return None
    
# ---------------------------
# DATABASE HELPERS
# ---------------------------
def try_mysql_connect(config_override=None):
    """
    Attempt connection with optional override (used for auth fallback).
    """
    cfg = DB_CONFIG.copy()
    if config_override:
        cfg.update(config_override)
    try:
        conn = mysql.connector.connect(**cfg)
        return conn
    except mysql.connector.Error as err:
        print("MySQL connection error:", err)
        # fallback attempt with mysql_native_password plugin for some environments
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
    """
    Returns a connected mysql.connector connection with DB created if missing.
    """
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
        # reconnect with database set
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
    """
    Lightweight wrapper that raises if DB is not available (useful for admin functions).
    """
    conn = get_db_connection()
    if not conn:
        raise Exception("Tidak dapat terhubung ke database")
    return conn

def ensure_table_exists():
    """Pastikan tabel responden ada dan lengkap tanpa error 1050."""
    conn = get_db_connection()
    if not conn:
        print("[ERROR] DB unavailable")
        return False

    try:
        cur = conn.cursor()

        # Cek apakah tabel sudah ada
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema=%s AND table_name=%s
        """, (DB_CONFIG["database"], DB_TABLE))

        exists = cur.fetchone()[0]

        if exists == 0:
            # Tabel belum ada → buat tabel baru
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
                cluster_label INT,
                waktu_analisis DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            cur.execute(create_sql)
            conn.commit()
            print("[INFO] Tabel responden berhasil dibuat.")
        else:
            print("[INFO] Tabel responden sudah ada, memeriksa kolom...")

        # Periksa kolom
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
    """
    Save a result row into DB. payload is a dict with keys:
    nama, usia, jenis_pekerjaan, wilayah, wilayah_id, Y1, Y2, Y3, kategori_Y1, kategori_Y2, kategori_Y3, x_scores (json)
    """
    conn = get_db_connection()
    if not conn:
        print("DB unavailable - cannot save")
        return False

    try:
        cur = conn.cursor()
        insert_sql = f"""
        INSERT INTO {DB_TABLE}
        (nama, usia, jenis_pekerjaan, wilayah, wilayah_id, Y1, Y2, Y3,
         kategori_Y1, kategori_Y2, kategori_Y3, x_scores)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(insert_sql, (
            payload.get("nama"),
            payload.get("usia"),
            payload.get("jenis_pekerjaan"),
            payload.get("wilayah"),
            payload.get("wilayah_id"),
            payload.get("Y1"),
            payload.get("Y2"),
            payload.get("Y3"),
            payload.get("kategori_Y1"),
            payload.get("kategori_Y2"),
            payload.get("kategori_Y3"),
            payload.get("x_scores")
        ))
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

# ---------------------------
# CLUSTERING FUNCTION
# ---------------------------
def run_clustering_on_db(wilayah='all', n_clusters=3):
    """
    Ambil data Y1,Y2,Y3 dan jalankan KMeans clustering.
    Menyimpan hasilnya pada kolom cluster_label.
    """
    conn = get_db_connection()
    if not conn:
        return False, "Database tidak tersedia."

    try:
        cur = conn.cursor(dictionary=True)

        # Filter wilayah jika tidak 'all'
        if wilayah == "all" or wilayah is None:
            cur.execute(f"SELECT id, Y1, Y2, Y3 FROM {DB_TABLE} "
                        f"WHERE Y1 IS NOT NULL AND Y2 IS NOT NULL AND Y3 IS NOT NULL")
        else:
            wilayah_name = WILAYAH_MAP.get(str(wilayah), wilayah)
            cur.execute(f"SELECT id, Y1, Y2, Y3 FROM {DB_TABLE} "
                        f"WHERE wilayah = %s AND Y1 IS NOT NULL AND Y2 IS NOT NULL AND Y3 IS NOT NULL",
                        (wilayah_name,))

        rows = cur.fetchall()

        if not rows:
            return False, "Tidak ada data yang cocok untuk clustering."

        df = pd.DataFrame(rows)
        X = df[['Y1', 'Y2', 'Y3']].astype(float).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        # Save model for runtime
        global_models['kmeans'] = {
            "centers_scaled": model.cluster_centers_.tolist(),
            "n_clusters": int(n_clusters)
        }

        # Update DB
        update_sql = f"UPDATE {DB_TABLE} SET cluster_label = %s WHERE id = %s"
        cur2 = conn.cursor()

        for idx, label in zip(df["id"], labels):
            cur2.execute(update_sql, (int(label), int(idx)))

        conn.commit()
        cur.close()
        cur2.close()
        conn.close()

        return True, f"Clustering selesai. {len(labels)} data berhasil diberi label."

    except Exception as e:
        print("Error run_clustering_on_db:", e)
        traceback.print_exc()
        return False, f"Error: {e}"

# ---------------------------
# DESCRIPTIVE STATISTICS
# ---------------------------
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

        # Statistik per wilayah
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

        # Convert decimal to float untuk JSON serialization
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

        # Statistik per pekerjaan
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

        # Ranking calculation untuk wilayah
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
            else:  # avg
                rv = (a1 + a2 + a3) / 3

            w["ranking_value"] = rv

        # Sort by ranking
        stats_wilayah_sorted = sorted(stats_wilayah, key=lambda x: x["ranking_value"], reverse=True)

        # Kategori stats
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

        # Overall stats
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

# ---------------------------
# ADDITIONAL ANALYSIS HELPERS
# ---------------------------

def calculate_wilayah_sem_analysis():
    """
    Hitung rata-rata Y1,Y2,Y3 per wilayah untuk analisis SEM.
    Mengembalikan dict keyed by wilayah_id:
        { wid: { 'wilayah': ..., 'count': ..., 'avg_Y1': ..., 'avg_Y2': ..., 'avg_Y3': ... }, ... }
    """
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
    """
    Generate short textual insights from descriptive stats.
    Returns list of insight strings (ready to render in template).
    """
    insights = []
    try:
        if not stats:
            return insights

        wilayah = stats.get("stats_wilayah", [])
        pekerjaan = stats.get("stats_pekerjaan", [])
        overall = stats.get("overall", {})

        # insights per wilayah
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
                # safe fallback if unexpected structure
                pass

        # insights per pekerjaan
        if pekerjaan:
            try:
                pekerjaan_tertinggi_y2 = max(pekerjaan, key=lambda x: x.get("avg_Y2", 0))
                pekerjaan_terbanyak = max(pekerjaan, key=lambda x: x.get("count", 0))
                insights.append(f"Pekerjaan dengan adopsi Cashless tertinggi: {pekerjaan_tertinggi_y2['jenis_pekerjaan']} (skor: {pekerjaan_tertinggi_y2['avg_Y2']:.2f})")
                insights.append(f"Kelompok pekerjaan terbanyak: {pekerjaan_terbanyak['jenis_pekerjaan']} ({pekerjaan_terbanyak['count']} responden)")
            except Exception:
                pass

        # overall insight
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

# ---------------------------
# ROUTES
# ---------------------------

# FORM DATA DIRI
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

# TAMPILKAN PERTANYAAN
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

# SUBMIT JAWABAN
@app.route("/submit_kuisioner", methods=["POST"])
def submit_kuisioner():
    try:
        wid = session.get("wilayah")

        # Ambil jawaban
        answers = {}
        for var in QUESTIONS_DETAILED:
            for item in QUESTIONS_DETAILED[var]["items"]:
                val = request.form.get(item)
                answers[item] = map_answer_to_weight(val)

        x_scores, item_scores = calculate_weighted_scores(answers)
        sem_scores = calculate_sem_scores(x_scores, wid)

        kategori_Y1 = score_to_category(sem_scores["Y1"])
        kategori_Y2 = score_to_category(sem_scores["Y2"])
        kategori_Y3 = score_to_category(sem_scores["Y3"])

        # Analysis
        analysis_results = analyze_variable_influences(x_scores, wid, sem_scores)
        improvement_recommendations = generate_improvement_recommendations(
            analysis_results, sem_scores, mode="short"
        )

        # Save DB
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

# ADMIN LOGIN
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
        if request.form.get("username") == "Admin" and request.form.get("password") == "SD23":
            session["admin_logged_in"] = True
            return redirect(url_for("data_responden_admin"))  # Redirect ke Data Responden
        else:
            flash("Login gagal", "danger")

    return render_template("admin_login.html", app_title=APP_TITLE)

@app.route("/admin_logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("index"))

# RUN CLUSTERING
@app.route("/run_clustering")
@admin_login_required
def run_clustering():
    wilayah = request.args.get("wilayah", "all")
    ok, msg = run_clustering_on_db(wilayah=wilayah, n_clusters=3)
    flash(msg, "success" if ok else "danger")
    return redirect(url_for("clustering_result"))

# CLUSTERING RESULT PAGE - STATISTIKA DESKRIPTIF
@app.route("/clustering_result")
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

        # Statistik deskriptif
        stats = calculate_descriptive_stats(selected_ranking)
        overall_chart = generate_overall_bar_chart(stats["overall"])
        responden_per_wilayah_chart = generate_responden_per_wilayah_chart(stats["stats_wilayah"])
        

        # Tambahan BARU → Grafik rata-rata skor per wilayah
        avg_score_wilayah_chart = generate_distribution_chart(stats['stats_wilayah'])

        # Grafik existing
        responden_per_wilayah_chart = generate_responden_per_wilayah_chart(stats["stats_wilayah"])



        return render_template(
        "clustering_result.html",
        app_title=APP_TITLE,
        data=data,
        stats=stats,
        selected_wilayah=selected_wilayah,
        selected_ranking=selected_ranking,
        wilayah_options=WILAYAH_MAP,
        score_to_category=score_to_category,
        responden_per_wilayah_chart=responden_per_wilayah_chart,
        avg_score_wilayah_chart=avg_score_wilayah_chart,
        overall_chart=overall_chart
    )



    except Exception as e:
        print("ERROR clustering_result:", e)
        traceback.print_exc()
        flash("Error mengambil data", "danger")
        return redirect(url_for("index"))


# DETAIL RESPONDEN
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

# HOME PAGE
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
    stats = calculate_descriptive_stats()
    wilayah_analysis = calculate_wilayah_sem_analysis()
    avg_x_scores_by_wilayah = calculate_wilayah_avg_scores()
    
    # Generate charts
    distribution_chart = generate_distribution_chart(stats['stats_wilayah'])
    # pekerjaan: avg chart + pie chart
    pekerjaan_avg_chart = generate_pekerjaan_avg_chart(stats['stats_pekerjaan'])
    pekerjaan_pie_chart = generate_pekerjaan_pie_chart(stats['stats_pekerjaan'])
    
    # Generate rekomendasi overall (dummy X assumed 3.0) and per-wilayah (if available)
    dummy_x_scores = {f"X{i}": 3.0 for i in range(1, 11)}
    dummy_sem_scores = {
        "Y1": stats['overall']['avg_Y1'] if stats['overall'] else 3.0,
        "Y2": stats['overall']['avg_Y2'] if stats['overall'] else 3.0,
        "Y3": stats['overall']['avg_Y3'] if stats['overall'] else 3.0
    }
    dummy_analysis = analyze_variable_influences(dummy_x_scores, "1", dummy_sem_scores)
    overall_recommendations = generate_improvement_recommendations(dummy_analysis, dummy_sem_scores, mode="detail", consolidated=True)

    selected_wilayah = request.args.get('wilayah', '1')
    wilayah_recommendations = []
    sem_diagrams = {}
    if selected_wilayah in avg_x_scores_by_wilayah:
        wilayah_avg_x = avg_x_scores_by_wilayah[selected_wilayah]
        sem_diagrams = create_wilayah_sem_diagram(selected_wilayah, wilayah_avg_x)
        sem_scores_w = calculate_sem_scores(wilayah_avg_x, selected_wilayah)
        analysis_w = analyze_variable_influences(wilayah_avg_x, selected_wilayah, sem_scores_w)
        wilayah_recommendations = generate_improvement_recommendations(analysis_w, sem_scores_w, mode="detail", consolidated=True)

    insights = generate_insights(stats)

    return render_template(
        "analisis_peta_hubungan.html",
        stats=stats,
        wilayah_analysis=wilayah_analysis,
        distribution_chart=distribution_chart,
        pekerjaan_avg_chart=pekerjaan_avg_chart,
        pekerjaan_pie_chart=pekerjaan_pie_chart,
        improvement_recommendations=overall_recommendations,
        overall_recommendations=overall_recommendations,
        wilayah_recommendations=wilayah_recommendations,
        sem_diagrams=sem_diagrams,
        selected_wilayah=selected_wilayah,
        avg_x_scores_by_wilayah=avg_x_scores_by_wilayah,
        WILAYAH_MAP=WILAYAH_MAP,
        X_VAR_NAME=X_VAR_NAME,  # TAMBAHKAN INI
        generate_insights=generate_insights,
        insights=insights,
        app_title=APP_TITLE,
    )
    
# SUBSCRIBE ROUTE
@app.route("/subscribe", methods=["POST"])
def subscribe():
    phone = request.form.get("phone")
    if phone:
        flash("Terima kasih telah berlangganan! Kami akan menghubungi Anda segera.", "success")
    else:
        flash("Nomor telepon diperlukan.", "danger")
    return redirect(url_for("index"))

# ADMIN DATA RESPONDEN
@app.route("/admin/data-responden")
@admin_login_required
def data_responden_admin():
    wilayah_filter = request.args.get("wilayah", "all")
    per_page = int(request.args.get("per_page", "10"))
    page = int(request.args.get("page", "1"))

    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    # Get total count for pagination
    if wilayah_filter == "all":
        cur.execute(f"SELECT COUNT(*) as total FROM {DB_TABLE}")
    else:
        nm = WILAYAH_MAP.get(wilayah_filter, wilayah_filter)
        cur.execute(f"SELECT COUNT(*) as total FROM {DB_TABLE} WHERE wilayah=%s", (nm,))

    total_count = cur.fetchone()["total"]
    total_pages = (total_count + per_page - 1) // per_page

    # Get paginated data
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
    

def ensure_cluster_column():
    """Pastikan kolom cluster_label ada di tabel responden."""
    conn = get_db_connection()
    if not conn:
        print("DB unavailable while ensuring cluster column")
        return

    try:
        cur = conn.cursor()

        cur.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND COLUMN_NAME='cluster_label'
        """, (DB_CONFIG["database"], DB_TABLE))

        exists = cur.fetchone()

        if not exists:
            print("[INFO] Menambahkan kolom cluster_label...")
            cur.execute(f"ALTER TABLE {DB_TABLE} ADD COLUMN cluster_label INT DEFAULT NULL;")
            conn.commit()

        cur.close()
        conn.close()

    except Exception as e:
        print("Error ensure_cluster_column:", e)


# ---------------------------
# MAIN START
# ---------------------------
if __name__ == "__main__":
    print("Initializing database table...")
    ensure_table_exists()
    ensure_cluster_column()
    print("Starting Flask app...")
    app.run(debug=True, host="0.0.0.0", port=5000)

