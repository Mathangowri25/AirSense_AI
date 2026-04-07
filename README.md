<div align="center">

<br/>

# 🌐 AQI Intelligence System

### AI-Powered Real-Time Air Quality Analysis, Prediction & Voice Alert Dashboard

<br/>

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-LINK.streamlit.app)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
&nbsp;
![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B?style=flat&logo=streamlit&logoColor=white)
&nbsp;
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1_8B-F55036?style=flat&logoColor=white)
&nbsp;
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat&logo=scikit-learn&logoColor=white)
&nbsp;
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)
&nbsp;
![Status](https://img.shields.io/badge/Status-Live-14b89a?style=flat)

<br/>

> **Upload your sensor readings · Predict AQI instantly · Get AI-generated analysis · Hear voice alerts — all in one futuristic dashboard.**

<br/>

---

</div>

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Live Demo](#-live-demo)
3. [Key Features](#-key-features)
4. [Dashboard Sections](#-dashboard-sections)
5. [Tech Stack](#-tech-stack)
6. [Machine Learning Model](#-machine-learning-model)
7. [Project Structure](#-project-structure)
8. [Input Data Format](#-input-data-format)
9. [AQI Classification](#-aqi-classification)
10. [Getting Started Locally](#-getting-started-locally)
11. [Environment Variables](#-environment-variables)
12. [Deployment on Streamlit Cloud](#-deployment-on-streamlit-cloud)
13. [How the Pipeline Works](#-how-the-pipeline-works)
14. [Team](#-team)
15. [License](#-license)

---

## 🧠 Project Overview

**AQI Intelligence System** is a final year B.E./B.Tech project that integrates machine learning, large language model inference, and text-to-speech synthesis into a single futuristic web dashboard. The system is built on top of Streamlit and powered by Groq's ultra-fast LLaMA 3.1-8B API.

Users upload an Excel file containing air quality sensor readings. The system then:
- Parses **18 engineered features** from the raw sensor data
- Predicts the AQI score using a trained scikit-learn model
- Classifies the health risk into one of 6 categories
- Generates a **detailed 5-section AI explanation** of the air quality situation using LLaMA 3.1-8B
- Synthesizes a **spoken voice alert** using gTTS
- Visualizes all data through an animated **3D globe, gauge chart, radar chart, pollutant bars, and trend plots**

The entire UI is styled with a custom dark neon/cyberpunk aesthetic using Orbitron and JetBrains Mono fonts, animated starfield canvas, and glowing CSS effects — all rendered inside Streamlit using `st.components.v1.html`.

---

## 🔗 Live Demo

> 🚀 **[Click here to open the live dashboard]([https://YOUR-APP-LINK.streamlit.app](https://airsenseai-aqi.streamlit.app/))**

To try the app:
1. Download the `AQI_Input_Template.xlsx` from this repo
2. Upload it on the dashboard
3. Select any sensor reading row from the dropdown
4. Click **⟶ RUN FULL ANALYSIS**

---

## ✨ Key Features

| # | Feature | Description |
|---|---|---|
| 1 | 📁 **Excel / CSV Upload** | Accepts `.xlsx`, `.xls`, or `.csv` sensor data files |
| 2 | 🔢 **18-Feature ML Prediction** | Engineered features: PM ratio, THI, CO2/CO ratio + raw pollutants |
| 3 | 🏷️ **AQI Classification** | 6-tier health risk classification (Good → Hazardous) |
| 4 | 🌐 **3D Globe Visualization** | Three.js animated sphere with rings and particles, color-coded by AQI |
| 5 | 📊 **Animated Gauge Chart** | Canvas-based semicircular AQI gauge with animated needle sweep |
| 6 | 🕸️ **Radar / Spider Chart** | 8-pollutant normalized radar chart with gradient fill |
| 7 | 📉 **Glowing Pollutant Bars** | Individual progress bars for PM2.5, PM10, NO2, SO2, O3, CO, CO2, Traffic |
| 8 | 🌡️ **Atmospheric Conditions** | Temperature, Humidity, Wind Speed, Pressure, Visibility, Traffic Density |
| 9 | 📈 **Trend Analysis** | Matplotlib/Seaborn historical trend plots from the dataset |
| 10 | 📟 **Sensor Metadata** | Battery level, Data Quality Index, PM ratio, THI, CO2/CO ratio |
| 11 | 🤖 **AI Analysis** | Groq LLaMA 3.1-8B generates 5-section analysis: causes, health effects, precautions, sources, suggestions |
| 12 | 🖥️ **Animated Terminal UI** | AI response displayed as typewriter-style terminal with live scan line animation |
| 13 | ⚠️ **Health Advisory Panel** | Precaution cards: Stay Indoors, Wear N95, Air Purifier, Vulnerable Groups |
| 14 | 🔊 **Voice Alert Synthesis** | gTTS converts AI summary to MP3 audio with browser autoplay |
| 15 | 🚨 **Critical Alert Banner** | Pulsing red warning banner triggered automatically when AQI > 150 |

---

## 🗂️ Dashboard Sections

The dashboard is organized into **8 numbered sections**, each with an animated neon section header:

```
Section 00 — UPLOAD SENSOR DATA       Upload .xlsx / .csv file
Section 00 — SELECT SENSOR READING    Preview table + row selector dropdown
──────────────── RUN FULL ANALYSIS button ────────────────
Section 01 — AQI PREDICTION RESULT    3D Globe + Gauge + 6 neon metric cards
Section 02 — RADAR ANALYSIS           8-pollutant spider chart + glowing bars
Section 03 — ATMOSPHERIC CONDITIONS   6 environmental metrics (temp, humidity, etc.)
Section 04 — TREND ANALYSIS           Historical AQI + pollutant trend plots
Section 05 — SENSOR METADATA          Battery, DQI, derived feature ratios
Section 06 — HEALTH ADVISORY          Safety cards + pollution sources grid
Section 07 — AI-GENERATED ANALYSIS    LLaMA 3.1-8B terminal output (typewriter)
Section 08 — VOICE ALERT              Waveform animation + MP3 audio player
```

---

## 🛠 Tech Stack

### Frontend & UI
| Tool | Version | Usage |
|---|---|---|
| [Streamlit](https://streamlit.io/) | 1.56.0 | Web framework and component rendering |
| Custom CSS | — | Dark neon theme, animated headers, glowing cards |
| [Three.js](https://threejs.org/) | r128 | 3D animated AQI globe visualization |
| HTML5 Canvas | — | AQI semicircle gauge and 8-axis radar chart |
| Google Fonts | — | Orbitron (headings), JetBrains Mono (code), Exo 2 (body) |

### Data Processing
| Tool | Version | Usage |
|---|---|---|
| [Pandas](https://pandas.pydata.org/) | 3.0.2 | DataFrame loading, column parsing, feature extraction |
| [NumPy](https://numpy.org/) | 2.4.4 | Array operations for model inference |
| [OpenPyXL](https://openpyxl.readthedocs.io/) | 3.1.5 | Reading `.xlsx` Excel files |

### Machine Learning
| Tool | Version | Usage |
|---|---|---|
| [scikit-learn](https://scikit-learn.org/) | 1.6.1 | Trained AQI score prediction model |
| Pickle | built-in | Model serialization and loading (`predict_AQI_score.pkl`) |

### AI & LLM
| Tool | Version | Usage |
|---|---|---|
| [Groq API](https://console.groq.com/) | 1.1.2 | LLaMA 3.1-8B-Instant for explanation + voice text generation |
| [gTTS](https://gtts.readthedocs.io/) | 2.5.4 | Google Text-to-Speech — MP3 audio generation |

### Visualization
| Tool | Version | Usage |
|---|---|---|
| [Matplotlib](https://matplotlib.org/) | 3.10.8 | Historical trend plots |
| [Seaborn](https://seaborn.pydata.org/) | 0.13.2 | Statistical chart styling |

### Infrastructure
| Tool | Usage |
|---|---|
| python-dotenv | Local `.env` file loading |
| Streamlit Secrets | Production API key management |
| GitHub | Version control and source hosting |
| Streamlit Community Cloud | Free hosting and deployment |

---

## 🤖 Machine Learning Model

### Model File
- **File:** `predict_AQI_score.pkl`
- **Framework:** scikit-learn
- **Task:** AQI score regression (predicts a continuous float value)
- **Training Data:** `enhanced_AQI_dataset.csv`

### Feature Engineering — 18 Input Features

| # | Feature | Type | Unit | Description |
|---|---|---|---|---|
| 1 | `PM2.5` | Raw | µg/m³ | Fine particulate matter |
| 2 | `PM10` | Raw | µg/m³ | Coarse particulate matter |
| 3 | `CO` | Raw | ppm | Carbon monoxide |
| 4 | `CO2` | Raw | ppm | Carbon dioxide |
| 5 | `NO2` | Raw | µg/m³ | Nitrogen dioxide |
| 6 | `SO2` | Raw | µg/m³ | Sulfur dioxide |
| 7 | `O3` | Raw | µg/m³ | Ozone |
| 8 | `Temperature` | Meteorological | °C | Ambient temperature |
| 9 | `Humidity` | Meteorological | % | Relative humidity |
| 10 | `Wind_Speed` | Meteorological | m/s | Wind speed |
| 11 | `Pressure` | Meteorological | hPa | Atmospheric pressure |
| 12 | `Visibility` | Meteorological | km | Atmospheric visibility |
| 13 | `Traffic_Density` | Encoded | index | Low=25 / Medium=55 / High=85 |
| 14 | `Battery` | Sensor | % | Sensor battery level |
| 15 | `Data_Quality_Index` | Sensor | 0–1 | Reading reliability score |
| 16 | `PM_Ratio` | **Engineered** | — | PM2.5 ÷ PM10 |
| 17 | `THI` | **Engineered** | — | Temperature-Humidity Index |
| 18 | `CO2_CO_Ratio` | **Engineered** | — | CO2 ÷ CO ratio |

### Fallback Logic
If the `.pkl` model file is not found, the system uses a weighted formula:
```python
AQI = (PM2.5 × 1.5) + (PM10 × 0.5) + (NO2 × 0.3)
```

---

## 📁 Project Structure

```
aqi-intelligence-system/
│
├── 📄 app.py                               # Main Streamlit app (1200+ lines)
│   ├── Global CSS & animated starfield     #   Dark neon theme
│   ├── Three.js 3D globe renderer          #   Color-coded AQI globe
│   ├── Canvas gauge chart                  #   Animated semicircle gauge
│   ├── Canvas radar/spider chart           #   8-pollutant radar
│   ├── AI terminal renderer                #   Typewriter animation for LLM output
│   ├── Utility functions                   #   classify_aqi, predict_aqi, gen_explanation
│   ├── File upload & row selector          #   Pandas parsing + DataFrame preview
│   └── 8 result sections                  #   Full output rendering pipeline
│
├── 📄 finalyear_project_classifier.py     # AQI classifier + pickle export script
├── 📄 LLM_configure.py                    # Standalone Groq LLM prompt tester
├── 📄 speech_model.py                     # Standalone voice alert generator
│
├── 🤖 predict_AQI_score.pkl               # Trained scikit-learn model
├── 📊 enhanced_AQI_dataset.csv            # Multi-city training dataset
├── 📓 FinalYear_Project.ipynb             # Jupyter notebook: EDA + model training
│
├── 📋 requirements.txt                    # Python dependencies
├── 🐍 runtime.txt                         # Python version → python-3.12
├── 🔒 .gitignore                          # Excludes .env, *.mp3, __pycache__
├── 🔑 .env                                # LOCAL ONLY — never commit this file
└── 📖 README.md                           # Project documentation
```

---

## 📥 Input Data Format

The app accepts an Excel file with the sheet name `AQI_INPUT` (header at row 3), or any `.csv` containing these columns:

| Column Name | Unit | Example Value |
|---|---|---|
| `Timestamp` | datetime | `2024-01-15 08:30` |
| `PM2.5` | µg/m³ | `85.4` |
| `PM10` | µg/m³ | `120.0` |
| `CO (ppm)` | ppm | `2.5` |
| `CO2` | ppm | `450.0` |
| `NO2` | µg/m³ | `55.0` |
| `SO2` | µg/m³ | `30.0` |
| `O3` | µg/m³ | `40.0` |
| `Temperature` | °C | `28.5` |
| `Humidity` | % | `72.0` |
| `Wind_Speed` | m/s | `3.2` |
| `Pressure` | hPa | `1013.0` |
| `Visibility` | km | `8.0` |
| `Traffic_Density` | Low / Medium / High | `Medium` |
| `Battery` | % | `85.0` |
| `Data_Quality_Index` | 0.0 – 1.0 | `0.92` |

> The column parser is **flexible** — it does a case-insensitive substring match on column names, so minor variations in column naming are handled automatically.

---

## 🟢 AQI Classification

| AQI Range | Category | Color | Health Impact |
|---|---|---|---|
| 0 – 50 | 🟢 **Good** | `#14b89a` | Air quality is satisfactory — minimal or no health risk |
| 51 – 100 | 🟡 **Moderate** | `#e8a443` | Acceptable quality; unusually sensitive people may be affected |
| 101 – 150 | 🟠 **Unhealthy for Sensitive Groups** | `#e8723a` | Children, elderly, and people with respiratory/heart disease at risk |
| 151 – 200 | 🔴 **Unhealthy** | `#e85a3a` | Everyone may begin to experience health effects |
| 201 – 300 | 🟣 **Very Unhealthy** | `#8a6fe8` | Health alert — everyone may experience serious effects |
| 301 + | ⚫ **Hazardous** | `#d43a2a` | Emergency conditions — entire population is affected |

> ⚠️ A **pulsing red CRITICAL ALERT banner** is automatically displayed at the top of results when AQI exceeds 150.

---

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.12 or higher
- A free Groq API key — [Get one at console.groq.com](https://console.groq.com)
- Git installed on your machine

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/aqi-intelligence-system.git
cd aqi-intelligence-system
```

### Step 2 — Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set Up API Keys

Create a `.env` file in the root of the project:
```env
GROK_API=your_groq_api_key_for_llm_analysis
GROK_SPEECH=your_groq_api_key_for_voice_alert
```

> Both keys can be the same Groq API key if you have a single account.

### Step 5 — Run the App
```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROK_API` | ✅ Yes | Groq API key used for LLaMA 3.1-8B explanation generation |
| `GROK_SPEECH` | ✅ Yes | Groq API key used for voice alert text generation |

### How Keys Are Loaded (Dual-Environment Support)

```python
# app.py — works in both local and Streamlit Cloud
GROQ_API_KEY_LLM    = st.secrets.get("GROK_API")    or os.getenv("GROK_API")
GROQ_API_KEY_SPEECH = st.secrets.get("GROK_SPEECH") or os.getenv("GROK_SPEECH")
```

| Environment | Where Keys Come From |
|---|---|
| Local development | `.env` file loaded by `python-dotenv` |
| Streamlit Cloud | Streamlit Secrets Manager (App Settings → Secrets) |

> ⚠️ **The `.env` file is in `.gitignore` and must never be pushed to GitHub.**

---

## ☁️ Deployment on Streamlit Cloud

### Step 1 — Push to GitHub
```bash
git add .
git commit -m "Initial deployment"
git push origin main
```

### Step 2 — Create App on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**
3. Fill in the form:
   - **Repository:** `YOUR_USERNAME/aqi-intelligence-system`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**

### Step 3 — Add API Keys as Secrets
In Streamlit Cloud → Your App → **Settings** → **Secrets**:
```toml
GROK_API = "your_groq_api_key_here"
GROK_SPEECH = "your_groq_api_key_here"
```
Click **Save** — the app will restart automatically with keys loaded.

### Step 4 — Python Version
The `runtime.txt` in the repo pins the Python version:
```
python-3.12
```
This ensures full compatibility with scikit-learn and all other packages.

---

## ⚙️ How the Pipeline Works

```
┌──────────────────────────────────────────────────────────────┐
│                    USER UPLOADS EXCEL FILE                    │
│               (.xlsx / .xls / .csv)                           │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│               PANDAS — Parse & Preview                        │
│   Read sheet "AQI_INPUT", display top 10 rows                │
│   User selects one row via dropdown                          │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│            FEATURE ENGINEERING (18 Features)                  │
│  Raw pollutants: PM2.5, PM10, CO, CO2, NO2, SO2, O3          │
│  Meteorological: Temp, Humidity, Wind, Pressure, Visibility  │
│  Sensor: Battery, Data_Quality_Index, Traffic                │
│  Engineered: PM_Ratio, THI, CO2_CO_Ratio                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│          scikit-learn MODEL  (predict_AQI_score.pkl)          │
│          Outputs → AQI Score (float)                          │
│          classify_aqi() → Health Risk Category                │
└─────────────┬────────────────────────────┬───────────────────┘
              │                            │
              ▼                            ▼
┌─────────────────────────┐  ┌────────────────────────────────┐
│    VISUALIZATIONS        │  │   GROQ LLaMA 3.1-8B            │
│  • 3D Globe (Three.js)  │  │   5-section AI Explanation:    │
│  • Gauge Chart (Canvas) │  │   1. Main Causes               │
│  • Radar Chart (Canvas) │  │   2. Health Effects            │
│  • Pollutant Bars       │  │   3. Safety Precautions        │
│  • Trend Plots          │  │   4. Pollution Sources         │
│  • Env Metric Cards     │  │   5. Improvement Suggestions   │
│  • Health Advisory      │  └──────────────┬─────────────────┘
└─────────────────────────┘                 │
                                            ▼
                           ┌────────────────────────────────────┐
                           │  GROQ → 2-3 sentence voice script  │
                           │  gTTS  → MP3 audio file            │
                           │  Autoplay in browser               │
                           └────────────────────────────────────┘
```

---

## 👨‍💻 Team

> B.E. / B.Tech Final Year Project
> Department of [YOUR DEPARTMENT] — [YOUR COLLEGE NAME] — Batch [YEAR]

| Name | Roll Number | Contribution |
|---|---|---|
| **YOUR NAME** | XXXXXXXX | ML Model Development, Streamlit Dashboard, Cloud Deployment |
| **TEAMMATE 2** | XXXXXXXX | LLM Integration (Groq API), Voice Alert Pipeline (gTTS) |
| **TEAMMATE 3** | XXXXXXXX | Dataset Collection, EDA, Feature Engineering, Jupyter Notebook |

**Project Guide:** [Guide Name], [Designation], [Department]

---

## 🙏 Acknowledgements

- [Groq](https://groq.com/) for providing ultra-fast LLaMA 3.1-8B inference API
- [Streamlit](https://streamlit.io/) for the open-source Python web framework
- [Three.js](https://threejs.org/) for 3D WebGL rendering in the browser
- [scikit-learn](https://scikit-learn.org/) for the machine learning toolkit
- [gTTS](https://gtts.readthedocs.io/) for Google Text-to-Speech synthesis
- [Google Fonts](https://fonts.google.com/) for Orbitron, JetBrains Mono, and Exo 2 typefaces

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution.

---

<div align="center">

**Built with ❤️ using Streamlit · Groq · Three.js · scikit-learn · gTTS**

<br/>

⭐ **If this project helped you, please give it a star on GitHub!**

<br/>

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/aqi-intelligence-system?style=social)](https://github.com/YOUR_USERNAME/aqi-intelligence-system)

</div>
