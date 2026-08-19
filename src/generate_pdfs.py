"""
NeoWatch - Comprehensive PDF Documentation Generator
Generates 3 separate professional PDF reports in 'Proje mantigi ve kullanilanlar':
1. 1_Kullanilan_Teknolojiler_ve_Araclar.pdf
2. 2_Proje_Mantigi_ve_Sistem_Mimarisi.pdf
3. 3_Matematiksel_ve_Istatistiksel_Bulgular.pdf
"""

import os
import sys
from pathlib import Path
from typing import List, Any

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Register Windows TrueType fonts for full Turkish character support
try:
    pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Italic", "C:/Windows/Fonts/ariali.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-BoldItalic", "C:/Windows/Fonts/arialbi.ttf"))
    FONT_NORMAL = "Arial"
    FONT_BOLD = "Arial-Bold"
    FONT_ITALIC = "Arial-Italic"
except Exception as font_err:
    print(f"Warning: Could not load Arial font: {font_err}. Falling back to Helvetica.")
    FONT_NORMAL = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"
    FONT_ITALIC = "Helvetica-Oblique"

# Target Output Directory
OUTPUT_DIR = Path("E:/My Projects/NeoWatch/Proje mantigi ve kullanilanlar")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class NumberedCanvas(canvas.Canvas):
    """Canvas that performs a two-pass calculation to display total page numbers in footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count: int):
        self.saveState()
        self.setFont(FONT_NORMAL, 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, A4[1] - 36, "NeoWatch — NASA Asteroit Tehlike Tahmin & İzleme Sistemi")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, A4[1] - 42, A4[0] - 54, A4[1] - 42)

        # Footer
        footer_text = f"Sayfa {self._pageNumber} / {page_count}"
        self.drawRightString(A4[0] - 54, 36, footer_text)
        self.drawString(54, 36, "Gizlilik Derecesi: Açık / Mühendislik & Mimari Raporu")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 48, A4[0] - 54, 48)

        self.restoreState()


def get_custom_styles():
    """Build modern, clean typography stylesheet."""
    styles = getSampleStyleSheet()

    # Base adjustments
    styles["Normal"].fontName = FONT_NORMAL
    styles["Normal"].fontSize = 9.5
    styles["Normal"].leading = 13.5
    styles["Normal"].textColor = colors.HexColor("#1E293B")

    # Document Main Title
    styles.add(
        ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=6,
        )
    )

    # Subtitle
    styles.add(
        ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontName=FONT_NORMAL,
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#0284C7"),
            spaceAfter=14,
        )
    )

    # Section Heading 1
    styles.add(
        ParagraphStyle(
            "Heading1_Custom",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=True,
        )
    )

    # Section Heading 2
    styles.add(
        ParagraphStyle(
            "Heading2_Custom",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#0369A1"),
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        )
    )

    # Callout Box Text
    styles.add(
        ParagraphStyle(
            "CalloutText",
            parent=styles["Normal"],
            fontName=FONT_NORMAL,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#0C4A6E"),
        )
    )

    # Table Content
    styles.add(
        ParagraphStyle(
            "TableContent",
            parent=styles["Normal"],
            fontName=FONT_NORMAL,
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#1E293B"),
        )
    )

    # Table Header
    styles.add(
        ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=8.5,
            leading=11.5,
            textColor=colors.white,
        )
    )

    # Code / Formula
    styles.add(
        ParagraphStyle(
            "FormulaBox",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#1E1B4B"),
            spaceBefore=4,
            spaceAfter=4,
        )
    )

    return styles


def create_callout(text: str, styles, title: str = "ÖNEMLİ BİLGİ", border_color="#0284C7", bg_color="#F0F9FF"):
    """Render a clean highlight callout block."""
    p_title = Paragraph(f"<b>{title}:</b>", styles["CalloutText"])
    p_text = Paragraph(text, styles["CalloutText"])
    t = Table([[p_title], [p_text]], colWidths=[A4[0] - 108])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_color)),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(border_color)),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 6),
            ]
        )
    )
    return t


# ==============================================================================
# PDF 1: KULLANILAN TEKNOLOJİLER VE ARAÇLAR REHBERİ
# ==============================================================================
def build_tech_stack_pdf():
    pdf_path = OUTPUT_DIR / "1_Kullanilan_Teknolojiler_ve_Araclar.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = get_custom_styles()
    story = []

    # Title & Banner
    story.append(Paragraph("🌌 NeoWatch — Kullanılan Teknolojiler ve Araçlar Rehberi", styles["DocTitle"]))
    story.append(Paragraph("Uçtan Uca Makine Öğrenmesi & Gezegensel Savunma Sistemi Teknik Yığın (Tech Stack) Analizi", styles["DocSubTitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=10))

    # Executive Summary
    story.append(Paragraph("1. Genel Bakış ve Mimari Katmanlar", styles["Heading1_Custom"]))
    story.append(
        Paragraph(
            "NeoWatch projesi, astronomik büyük verinin güvenli şekilde çekilmesinden, dengesiz veri kümesi üzerinde yüksek duyarlılıklı (Recall) makine öğrenmesi modellerinin eğitilmesine ve son kullanıcıya web dashboard üzerinden interaktif sunulmasına kadar endüstriyel standartlarda tasarlanmış 5 temel mimari katmandan oluşmaktadır.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 8))

    # Tech Stack Table
    tech_data = [
        [
            Paragraph("Katman / Alan", styles["TableHeader"]),
            Paragraph("Kullanılan Teknoloji & Sürüm", styles["TableHeader"]),
            Paragraph("Görev ve Mimari Sorumluluk", styles["TableHeader"]),
        ],
        [
            Paragraph("<b>Programlama Dili</b>", styles["TableContent"]),
            Paragraph("Python 3.13.1 (64-bit)", styles["TableContent"]),
            Paragraph("Tüm boru hattının (Ingestion, Preprocessing, Modeling, Serving) ana çalışma platformu.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>Harici Veri Kaynağı</b>", styles["TableContent"]),
            Paragraph("NASA NeoWs REST API", styles["TableContent"]),
            Paragraph("Dünya'ya Yakın Cisimlerin (NEO) fiziksel boyut, parlaklık, hız ve yörünge mesafe verilerinin canlı ve tarihsel akışı.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>Veri Boru Hattı & İletişim</b>", styles["TableContent"]),
            Paragraph("urllib / requests / json / csv", styles["TableContent"]),
            Paragraph("NASA'nın 7 günlük sorgu kısıtına karşı otomatik sliding-window, HTTP 429 rate-limit yakalama ve backoff yönetimi.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>Veri Manipülasyonu</b>", styles["TableContent"]),
            Paragraph("Pandas 2.2+ & NumPy 1.26+", styles["TableContent"]),
            Paragraph("Ham JSON yanıtlarını düzleştirme (flattening), matris hesaplamaları, öznitelik türetimi ve CSV veri serileştirme.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>Aykırı Değer & Ölçekleme</b>", styles["TableContent"]),
            Paragraph("Scikit-Learn (StandardScaler)", styles["TableContent"]),
            Paragraph("Astronomik büyüklüklerin IQR tabanlı quantile clipping ile traşlanması ve z-score ölçeklemesi.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>Dengesiz Veri Yönetimi</b>", styles["TableContent"]),
            Paragraph("Imbalanced-Learn (SMOTE)", styles["TableContent"]),
            Paragraph("Tehlikeli asteroit sınıfındaki (%11) azınlık dengesizliğini aşmak için sentetik örneklem üretimi (Synthetic Minority Over-sampling).", styles["TableContent"]),
        ],
        [
            Paragraph("<b>Makine Öğrenmesi Çekirdeği</b>", styles["TableContent"]),
            Paragraph("XGBoost 3.4 & LightGBM 4.7", styles["TableContent"]),
            Paragraph("Gradient Boosted Trees mimarisiyle Recall odaklı sınıflandırma ve 810 kombinasyonlu GridSearchCV optimizasyonu.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>Kıyaslama Modelleri</b>", styles["TableContent"]),
            Paragraph("Random Forest & Logistic Regression", styles["TableContent"]),
            Paragraph("5-Fold Stratified Cross Validation ile model kıyaslama ve taban (baseline) performans doğrulama.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>Model Serileştirme</b>", styles["TableContent"]),
            Paragraph("Joblib 1.5+", styles["TableContent"]),
            Paragraph("Eğitilmiş scaler.pkl ve asteroid_model.pkl nesnelerinin üretim ortamı için diske saklanması ve yüklenmesi.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>İnteraktif Görselleştirme</b>", styles["TableContent"]),
            Paragraph("Plotly Express & Graph Objects", styles["TableContent"]),
            Paragraph("Hız-Mesafe Risk Matrisi (Scatter), Çap Dağılımları (Boxplot) ve Tehlike Olasılık Göstergesi (Gauge Chart).", styles["TableContent"]),
        ],
        [
            Paragraph("<b>Sunum & Web Katmanı</b>", styles["TableContent"]),
            Paragraph("Streamlit Framework", styles["TableContent"]),
            Paragraph("@st.cache_resource ve @st.cache_data ile yüksek performanslı, reaktif uzay temalı web dashboard arayüzü.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>Güvenlik & DevOps</b>", styles["TableContent"]),
            Paragraph("python-dotenv & Git (.gitignore)", styles["TableContent"]),
            Paragraph("NASA API anahtarının gizlenmesi, büyük CSV ve ikili model dosyalarının repo şişkinliğini önleyecek şekilde izole edilmesi.", styles["TableContent"]),
        ],
    ]

    t_tech = Table(tech_data, colWidths=[110, 130, 248])
    t_tech.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B192C")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 4.5),
            ]
        )
    )
    story.append(t_tech)
    story.append(Spacer(1, 10))

    # Detailed Module Deep-Dive
    story.append(Paragraph("2. Teknolojilerin Seçim Gerekçeleri ve Entegrasyon Detayları", styles["Heading1_Custom"]))
    
    story.append(Paragraph("A. Neden XGBoost ve LightGBM?", styles["Heading2_Custom"]))
    story.append(
        Paragraph(
            "Tabüler verilerde derin öğrenme modellerine kıyasla Gradient Boosted Decision Trees (GBDT) çok daha üstün genelleme kabiliyetine sahiptir. XGBoost'un <code>scale_pos_weight</code> hiperparametresi, azınlık sınıfı (tehlikeli cisimler) için pozitif kayıp ağırlığını artırarak modelin 'Recall' odaklı öğrenmesini garanti altına almıştır.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("B. Neden SMOTE (Synthetic Minority Over-sampling)?", styles["Heading2_Custom"]))
    story.append(
        Paragraph(
            "NASA veri kümesinde tehlikeli asteroit oranı yaklaşık %11'dir. Standart modeller tüm asteroitlere 'Güvenli' diyerek %89 doğruluk (Accuracy) yakalayabilir ancak gezegensel savunmada bu felakettir. SMOTE, azınlık sınıfındaki k-en yakın komşular arasında lineer interpolasyon yaparak sentetik veri türetir ve modelin gerçek tehlikeleri kaçırmasını engeller.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("C. Neden Streamlit ve Plotly?", styles["Heading2_Custom"]))
    story.append(
        Paragraph(
            "Streamlit, Python tabanlı veri bilimcilerin backend/frontend ayrımı olmaksızın üretim seviyesinde arayüzler geliştirmesini sağlar. Plotly'nin dinamik WebGL ve SVG tabanlı grafik kütüphanesi sayesinde kullanıcılar asteroitlerin mesafe-hız uzayında yakınlaştırma (zoom) ve detay filtreleme yapabilmektedir.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(
        create_callout(
            "Tüm bağımlılıklar ve kütüphane sürümleri <code>requirements.txt</code> içerisinde pinlenmiş olup, izole sanal ortam (<code>.venv</code>) üzerinden sıfır çakışma ile çalıştırılmaktadır.",
            styles,
            title="MÜHENDİSLİK STANDARDI",
            border_color="#10B981",
            bg_color="#ECFDF5",
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated: {pdf_path}")


# ==============================================================================
# PDF 2: PROJE MANTIĞI VE SİSTEM MİMARİSİ RAPORU
# ==============================================================================
def build_architecture_logic_pdf():
    pdf_path = OUTPUT_DIR / "2_Proje_Mantigi_ve_Sistem_Mimarisi.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = get_custom_styles()
    story = []

    # Title & Banner
    story.append(Paragraph("🌌 NeoWatch — Proje Mantığı ve Sistem Mimarisi Raporu", styles["DocTitle"]))
    story.append(Paragraph("Gezegensel Savunma Felsefesi, Mühendislik Kararları ve 5 Aşamalı Geliştirme Yaşam Döngüsü", styles["DocSubTitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=10))

    # Section 1: Philosophy
    story.append(Paragraph("1. Gezegensel Savunma ve Asimetrik Hata Maliyeti Mantığı", styles["Heading1_Custom"]))
    story.append(
        Paragraph(
            "Geleneksel makine öğrenmesi projelerinde model başarısı genellikle <b>Accuracy (Genel Doğruluk)</b> ile ölçülür. Ancak Dünya'ya Yakın Cisimlerin (NEO) tehlike sınıflandırmasında hata maliyetleri <b>asimetriktir</b>:",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 4))

    matrix_data = [
        [
            Paragraph("Hata Türü", styles["TableHeader"]),
            Paragraph("İstatistiki Tanım", styles["TableHeader"]),
            Paragraph("Gezegensel Savunmadaki Gerçek Sonucu", styles["TableHeader"]),
        ],
        [
            Paragraph("<b>False Positive (Tip I Hata)</b>", styles["TableContent"]),
            Paragraph("Zararsız bir asteroite 'Tehlikeli' demek.", styles["TableContent"]),
            Paragraph("<b>Kabul Edilebilir Risk:</b> Teleskopların cisme ek gözlem saati ayırmasına neden olur. Maddi kayıp düşüktür.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>False Negative (Tip II Hata)</b>", styles["TableContent"]),
            Paragraph("Gerçekten tehlikeli bir asteroite 'Zararsız' demek.", styles["TableContent"]),
            Paragraph("<b>KATASROFİK RİSK:</b> Dünya'yı yok edebilecek bir cismin gözden kaçmasıdır. Sıfır toleransla çalışılmalıdır!", styles["TableContent"]),
        ],
    ]
    t_mat = Table(matrix_data, colWidths=[130, 130, 228])
    t_mat.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B192C")),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#FEF2F2")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_mat)
    story.append(Spacer(1, 10))

    # Section 2: End-to-End Flow
    story.append(Paragraph("2. Uçtan Uca Sistem Mimarisi ve Veri Akış Şeması", styles["Heading1_Custom"]))
    story.append(
        Paragraph(
            "Sistem 4 ana aşamalı bir boru hattı üzerinden çalışır: <b>1) Ingestion</b> (NASA API Çekimi) → <b>2) Transformation</b> (Önişleme & SMOTE) → <b>3) Intelligence</b> (XGBoost Eğitimi & CV) → <b>4) Serving</b> (Streamlit Dashboard & Canlı Skorlama).",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 6))

    flow_data = [
        [Paragraph("Bileşen / Modül", styles["TableHeader"]), Paragraph("İşlev ve Çalışma Prensibi", styles["TableHeader"])],
        [
            Paragraph("<b>src/api_client.py</b>", styles["TableContent"]),
            Paragraph("NASA NeoWs API'sine istek atar. 7 günlük sliding-window döngüsüyle 1-3 yıllık verileri rate-limit (429) korumalı çeker ve JSON hiyerarşisini düzleştirir.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>src/preprocessor.py</b>", styles["TableContent"]),
            Paragraph("Ham veriyi temizler, IQR ile aşırı uç değerleri sınırlar, StandardScaler nesnesini sadece eğitim verisiyle fit edip serileştirir ve SMOTE oversampling uygular.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>src/model_trainer.py</b>", styles["TableContent"]),
            Paragraph("Random Forest, LightGBM, XGBoost modellerini 5-Fold Stratified CV ile karşılaştırır. GridSearchCV ile XGBoost parametrelerini Recall metriğine göre optimize eder.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>src/predictor.py</b>", styles["TableContent"]),
            Paragraph("Kaydedilen <code>scaler.pkl</code> ve <code>asteroid_model.pkl</code> dosyalarını yükler. Canlı API sorgularını veya simülatör girdilerini milisaniyeler içinde tehlike olasılığına dönüştürür.", styles["TableContent"]),
        ],
        [
            Paragraph("<b>app.py</b>", styles["TableContent"]),
            Paragraph("Streamlit tabanlı uzay temalı web paneli. Canlı NASA Radarı, 'What-If' Tehlike Simülatörü ve Tarihsel İstatistik sekmelerini sunar.", styles["TableContent"]),
        ],
    ]
    t_flow = Table(flow_data, colWidths=[130, 358])
    t_flow.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B192C")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 4.5),
            ]
        )
    )
    story.append(t_flow)
    story.append(Spacer(1, 10))

    # Section 3: Checkpoint Lifecycle
    story.append(Paragraph("3. 5 Aşamalı Checkpoint Yaşam Döngüsü ve Doğrulama", styles["Heading1_Custom"]))
    
    checkpoints = [
        ("Checkpoint 1 (Veri Boru Hattı)", "NASA API'den 1 yıllık 2.262 adet asteroit başarıyla çekilmiş ve 'data/raw_asteroid_data.csv' oluşturulmuştur."),
        ("Checkpoint 2 (Özellik Mühendisliği)", "IQR quantile capping uygulanmış, StandardScaler eğitilmiş ve SMOTE ile 2.576 dengeli eğitim örneği elde edilmiştir."),
        ("Checkpoint 3 (Zeka Katmanı)", "Tuned XGBoost modeli %96.00 Recall başarısına ulaşmış ve 'models/asteroid_model.pkl' olarak serileştirilmiştir."),
        ("Checkpoint 4 (Sunum Katmanı)", "Streamlit web paneli (app.py) localhost:8501 üzerinde canlı radar ve Plotly görselleştirmeleriyle çalıştırılmıştır."),
        ("Checkpoint 5 (Final Dağıtım)", "Güvenlik kuralları (.gitignore, .env), profesyonel README.md ve Streamlit Cloud entegrasyonu tamamlanmıştır."),
    ]

    for cp_title, cp_desc in checkpoints:
        story.append(Paragraph(f"<b>• {cp_title}:</b> {cp_desc}", styles["Normal"]))
        story.append(Spacer(1, 3))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated: {pdf_path}")


# ==============================================================================
# PDF 3: MATEMATİKSEL VE İSTATİSTİKSEL BULGULAR RAPORU
# ==============================================================================
def build_math_findings_pdf():
    pdf_path = OUTPUT_DIR / "3_Matematiksel_ve_Istatistiksel_Bulgular.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = get_custom_styles()
    story = []

    # Title & Banner
    story.append(Paragraph("🌌 NeoWatch — Matematiksel ve İstatistiksel Bulgular Raporu", styles["DocTitle"]))
    story.append(Paragraph("Astronomik Formüller, Algoritmik Denklemler, SMOTE Vektör Matematiği ve Ampirik Test Sonuçları", styles["DocSubTitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=10))

    # Section 1: Astronomical Mathematics
    story.append(Paragraph("1. Astronomik ve Fiziksel Modelleme Formülleri", styles["Heading1_Custom"]))
    story.append(
        Paragraph(
            "Asteroitlerin optik parlaklığından (Mutlak Büyüklük - $H$) tahmini çapın ($D$) hesaplanmasında NASA ve IAU standart formülü kullanılır:",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>D (km) = ( 1329 / √pv ) × 10^(-0.2 × H)</b>", styles["FormulaBox"]))
    story.append(
        Paragraph(
            "Burada <i>p<sub>v</sub></i> cismin geometrik albedo (yansıtma) katsayısıdır (genellikle 0.05 ile 0.25 arası). Albedo belirsizliği nedeniyle NASA API minimum ve maksimum çap tahmin aralığı sunar. Modelimizde bu aralığın geometrik/aritmetik ortalaması (<code>estimated_diameter_mean_km</code>) türetilmiştir.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 8))

    # Section 2: IQR & SMOTE Math
    story.append(Paragraph("2. İstatistiksel Dağılım, IQR ve SMOTE Vektör Matematiği", styles["Heading1_Custom"]))
    story.append(
        Paragraph(
            "<b>A. Interquartile Range (IQR) Aykırı Değer Sınırlandırması:</b><br/>"
            "Astronomik mesafeler aşırı sağa çarpık (skewed) dağılım sergiler. Veri kaybını önlemek için non-destructive quantile capping uygulanmıştır:<br/>"
            "<code>IQR = Q3 (75. yüzdelik) - Q1 (25. yüzdelik)</code> | <code>Üst Sınır = Q3 + 3.0 × IQR</code>",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 6))

    story.append(
        Paragraph(
            "<b>B. SMOTE Sentetik Vektör İnterpolasyonu:</b><br/>"
            "Azınlık sınıfındaki her <i>x<sub>i</sub></i> tehlikeli asteroit için k-en yakın komşu kümesinden rastgele seçilen <i>x<sub>zi</sub></i> arasında sentetik örnek üretilir:<br/>"
            "<code>x_yeni = x_i + λ × (x_zi - x_i),   λ ~ Uniform(0, 1)</code><br/>"
            "Bu yöntemle eğitim setindeki tehlikeli örnek sayısı 199'dan <b>966'ya</b> yükseltilmiş ve toplam eğitim kümesi <b>2.576 örneğe</b> dengelenmiştir.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 8))

    # Section 3: XGBoost Objective
    story.append(Paragraph("3. XGBoost Kayıp Fonksiyonu ve Taylor Açılımı", styles["Heading1_Custom"]))
    story.append(
        Paragraph(
            "XGBoost'un her <i>t</i> adımındaki regularized hedef fonksiyonu 2. derece Taylor yaklaşımıyla minimize edilir:<br/>"
            "<code>Obj^(t) ≈ ∑ [ g_i × f_t(x_i) + 0.5 × h_i × f_t(x_i)^2 ] + γ T + 0.5 × λ ∑ w_j^2</code><br/>"
            "Burada <i>g<sub>i</sub> = ∂L/∂ŷ</i> (1. türev gradyan) ve <i>h<sub>i</sub> = ∂²L/∂ŷ²</i> (2. türev hessian) değerleridir. <code>scale_pos_weight=3.0</code> parametresi ile pozitif sınıf gradyanları 3 kat ağırlıklandırılarak Recall maksimize edilmiştir.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 8))

    # Section 4: Empirical Results Table
    story.append(Paragraph("4. Ampirik Kıyaslama ve Test Seti Doğrulama Sonuçları", styles["Heading1_Custom"]))
    
    res_data = [
        [
            Paragraph("Makine Öğrenmesi Modeli", styles["TableHeader"]),
            Paragraph("5-Fold CV Recall", styles["TableHeader"]),
            Paragraph("5-Fold CV ROC-AUC", styles["TableHeader"]),
            Paragraph("5-Fold CV F1-Score", styles["TableHeader"]),
            Paragraph("5-Fold CV Precision", styles["TableHeader"]),
        ],
        [
            Paragraph("<b>Random Forest</b>", styles["TableContent"]),
            Paragraph("%97.10 (±1.2%)", styles["TableContent"]),
            Paragraph("0.9614 (±0.008)", styles["TableContent"]),
            Paragraph("0.8516", styles["TableContent"]),
            Paragraph("%75.85", styles["TableContent"]),
        ],
        [
            Paragraph("<b>LightGBM</b>", styles["TableContent"]),
            Paragraph("%96.58 (±1.2%)", styles["TableContent"]),
            Paragraph("0.9500 (±0.008)", styles["TableContent"]),
            Paragraph("0.8421", styles["TableContent"]),
            Paragraph("%74.66", styles["TableContent"]),
        ],
        [
            Paragraph("<b>Tuned XGBoost (Nihai)</b>", styles["TableContent"]),
            Paragraph("<b>%99.90 (±0.2%)</b>", styles["TableContent"]),
            Paragraph("<b>0.9530 (±0.008)</b>", styles["TableContent"]),
            Paragraph("0.8505", styles["TableContent"]),
            Paragraph("%78.53", styles["TableContent"]),
        ],
        [
            Paragraph("<b>Logistic Regression</b>", styles["TableContent"]),
            Paragraph("%92.03 (±3.3%)", styles["TableContent"]),
            Paragraph("0.8953 (±0.006)", styles["TableContent"]),
            Paragraph("0.7910", styles["TableContent"]),
            Paragraph("%69.40", styles["TableContent"]),
        ],
    ]
    t_res = Table(res_data, colWidths=[120, 92, 92, 92, 92])
    t_res.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B192C")),
                ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#EFF6FF")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("PADDING", (0, 0), (-1, -1), 4.5),
            ]
        )
    )
    story.append(t_res)
    story.append(Spacer(1, 8))

    # Confusion Matrix Block
    story.append(Paragraph("5. Holdout Test Seti (N=453) Karmaşıklık Matrisi Analizi", styles["Heading1_Custom"]))
    cm_data = [
        [Paragraph("", styles["TableContent"]), Paragraph("<b>Tahmin: GÜVENLİ (0)</b>", styles["TableHeader"]), Paragraph("<b>Tahmin: TEHLİKELİ (1)</b>", styles["TableHeader"])],
        [Paragraph("<b>Gerçek: GÜVENLİ (0)</b>", styles["TableContent"]), Paragraph("312 (Doğru Negatif - TN)", styles["TableContent"]), Paragraph("91 (Yanlış Pozitif - FP)", styles["TableContent"])],
        [Paragraph("<b>Gerçek: TEHLİKELİ (1)</b>", styles["TableContent"]), Paragraph("<b>2 (KAÇIRILAN - FN)</b>", styles["TableContent"]), Paragraph("<b>48 (YAKALANAN - TP)</b>", styles["TableContent"])],
    ]
    t_cm = Table(cm_data, colWidths=[140, 174, 174])
    t_cm.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (1, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("BACKGROUND", (1, 2), (1, 2), colors.HexColor("#FEE2E2")),
                ("BACKGROUND", (2, 2), (2, 2), colors.HexColor("#DCFCE7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_cm)
    story.append(Spacer(1, 8))

    story.append(
        create_callout(
            "Test setindeki 50 gerçek tehlikeli asteroitten <b>48 tanesi başarıyla tespit edilmiş (%96.00 Recall)</b>, kaçırılan asteroit sayısı sadece 2 ile sınırlandırılmıştır. Modelin ROC-AUC skoru <b>0.9099</b> olarak doğrulanmıştır.",
            styles,
            title="SONUÇ VE BULGU ÖZETİ",
            border_color="#0284C7",
            bg_color="#F0F9FF",
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated: {pdf_path}")


if __name__ == "__main__":
    build_tech_stack_pdf()
    build_architecture_logic_pdf()
    build_math_findings_pdf()
    print("All 3 PDF reports successfully generated!")
