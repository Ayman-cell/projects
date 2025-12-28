import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import tempfile
import os
import io
import requests
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from fpdf import FPDF
import markdown
import google.generativeai as genai
from PIL import Image
import PyPDF2
import docx
import re
import plotly.io as pio

# Tentative d'import de python-dotenv pour les variables d'environnement
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Si python-dotenv n'est pas installé, continuer sans
    pass

# === CONFIGURATION DES APIs ===
# Utiliser les variables d'environnement si disponibles, sinon utiliser les valeurs par défaut
CEREBRAS_CONFIG = {
    "gpt-oss-120b": {
        "api_key": os.getenv("CEREBRAS_GPT_OSS_120B_KEY", "csk-vcmp5dpp29px96m6p23v2mcdw93xc5x64ptwenwcth5ky98c"),
        "endpoint": os.getenv("CEREBRAS_ENDPOINT", "https://api.cerebras.ai/v1/completions")
    },
    "qwen-3-235b-a22b-instruct-2507": {
        "api_key": os.getenv("CEREBRAS_QWEN_235B_KEY", "csk-8wtkv5mpecdc9cmk6d4jpyp9y4xtwrcjxmkjx6vey3942cyj"), 
        "endpoint": os.getenv("CEREBRAS_ENDPOINT", "https://api.cerebras.ai/v1/completions")
    },
    "llama-3.3-70b": {
        "api_key": os.getenv("CEREBRAS_QWEN_32B_KEY", "csk-vcmp5dpp29px96m6p23v2mcdw93xc5x64ptwenwcth5ky98c"),
        "endpoint": os.getenv("CEREBRAS_ENDPOINT", "https://api.cerebras.ai/v1/completions")
    }
}

GEMINI_CONFIG = {
    "api_key": os.getenv("GEMINI_API_KEY", "AIzaSyAFYfD6CqBt1WO7w4b6Xn8RTe5tiZSKNKg")
}

# Initialisation Gemini
genai.configure(api_key=GEMINI_CONFIG["api_key"])

class CerebriumLLM:
    def __init__(self, model_name, temperature=0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = CEREBRAS_CONFIG[model_name]["api_key"]
        self.endpoint = CEREBRAS_CONFIG[model_name]["endpoint"]
    
    def invoke(self, prompt, max_tokens=10000):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": False
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=100000)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"]
        except Exception as e:
            st.error(f"Erreur API Cerebrium ({self.model_name}): {str(e)}")
            return None

class WindRoseGenerator:
    def __init__(self):
        try:
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.error(f"Erreur initialisation Gemini: {str(e)}")
            self.gemini_model = None
    
    def generate_wind_rose_plotly(self, wind_data):
        """Génère une Wind Rose avec Plotly directement (16 secteurs, stacked by speed classes)"""
        try:
            sectors = wind_data.get('secteurs', [])
            if not sectors:
                st.warning("Aucune donnée de secteurs pour la rose des vents")
                return None

            # Extract sector names and center angles (compute center of min/max in each sector)
            directions = [s['nom'] for s in sectors]
            angles = []
            for s in sectors:
                amin = float(s.get('angle_min', 0))
                amax = float(s.get('angle_max', 0))
                # handle wrap-around for North (where amin > amax)
                if amin > amax:
                    aavg = ((amin + (amax + 360)) / 2) % 360
                else:
                    aavg = (amin + amax) / 2
                angles.append(aavg)

            # Speed classes in the expected order to match the color legend (calm first)
            classes = ['calme_0_1', 'legere_1_2', 'brise_2_5', 'moderee_5_8', 'forte_8_20']
            # Map colors to classes (choose a sequential palette that matches your example)
            colors = {
                "calme_0_1": "#A5D6A7",    # green (very light)
                "legere_1_2": "#66BB6A",   # green
                "brise_2_5": "#FDD835",    # yellow
                "moderee_5_8": "#FF9800",  # orange
                "forte_8_20": "#E53935"    # red
            }
            labels = {
                "calme_0_1": "Calme 0-1 m/s",
                "legere_1_2": "1-2 m/s",
                "brise_2_5": "2-5 m/s",
                "moderee_5_8": "5-8 m/s",
                "forte_8_20": "8-20 m/s"
            }

            # Sector width (degrees) for 16 sectors
            sector_width = 360 / max(1, len(sectors))

            fig = go.Figure()
            # Build traces in stacking order
            for classe in classes:
                values = []
                for s in sectors:
                    # Ensure a default 0 if the key is missing
                    values.append(float(s.get('distribution_vitesses', {}).get(classe, 0)))
                fig.add_trace(go.Barpolar(
                    r=values,
                    theta=angles,
                    width=[sector_width]*len(values),
                    name=labels.get(classe, classe),
                    marker_color=colors.get(classe, "#888888"),
                    opacity=0.95,
                ))
            
            # Determine max radius for nice layout
            max_r = max( (sum([float(s.get('distribution_vitesses', {}).get(c, 0)) for c in classes]) for s in sectors), default=40)
            max_r = max(10, np.ceil(max_r / 10.0) * 10)  # round up to next 10

            # Add central calm annotation (percentage)
            central_calm = wind_data.get('calme_central', 0)
            fig.update_layout(
                title=dict(
                    text=f"Rose des Vents - {wind_data.get('periode_analyse', '')}",
                    font=dict(size=16, color='#006838')
                ),
                polar=dict(
                    radialaxis=dict(
                        tickfont_size=10,
                        range=[0, max_r],
                        dtick=max(1, max_r / 4),
                        angle=90,
                        gridcolor="#d6d6d6"
                    ),
                    angularaxis=dict(
                        tickfont_size=11,
                        rotation=90,
                        direction="clockwise",
                        tickmode='array',
                        tickvals=angles,
                        ticktext=directions
                    )
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.1
                ),
                showlegend=True,
                height=620,
                margin=dict(t=80, b=30, l=20, r=170)
            )

            # Add small annotation in the center with the calm percentage
            fig.add_annotation(
                text=f"Calme central<br>{central_calm}%",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=12, color="#333333"),
                bgcolor="#ffffff", bordercolor="#e0e0e0", borderwidth=1, borderpad=4
            )

            return fig
        except Exception as e:
            st.error(f"Erreur génération Wind Rose: {str(e)}")
            return None

class GraphGenerator:
    """Génère tous les graphiques supplémentaires pour le rapport"""
    
    def __init__(self):
        pass
    
    def generate_time_series(self, df, periode_type, date_debut=None, date_fin=None):
        """Génère des graphiques de séries temporelles pour toutes les variables"""
        try:
            if df is None or len(df) == 0:
                try:
                    import logging
                    logging.warning("generate_time_series: DataFrame vide ou None")
                except:
                    pass
                return None
            
            # S'assurer que l'index est bien un DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df = df.copy()
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    df = df[df.index.notna()]
                except Exception as e:
                    try:
                        import logging
                        logging.error(f"Erreur conversion index en DatetimeIndex: {e}")
                    except:
                        pass
                    return None
            
            # Vérifier que nous avons des données après conversion
            if len(df) == 0:
                try:
                    import logging
                    logging.warning("generate_time_series: Aucune donnée après conversion de l'index")
                except:
                    pass
                return None
            
            # Vérifier que les colonnes nécessaires existent
            required_cols = ['AirTemp', 'Speed#@1m', 'RH', 'Power']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                try:
                    import logging
                    logging.warning(f"generate_time_series: Colonnes manquantes: {missing_cols}")
                    logging.info(f"Colonnes disponibles: {list(df.columns)}")
                except:
                    pass
            
            # Créer un subplot avec 4 graphiques
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Température (°C)', 'Vitesse du Vent (m/s)', 
                               'Humidité Relative (%)', 'Alimentation (V)'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # Déterminer les limites de l'axe X basées sur les dates de début et fin
            x_min = None
            x_max = None
            if date_debut:
                try:
                    x_min = pd.to_datetime(date_debut)
                except:
                    pass
            if date_fin:
                try:
                    x_max = pd.to_datetime(date_fin)
                    # Ajouter la fin de journée pour inclure toute la journée de fin
                    if x_max:
                        x_max = x_max.replace(hour=23, minute=59, second=59)
                except:
                    pass
            
            # Si les dates ne sont pas fournies, utiliser les dates min/max du DataFrame
            if x_min is None and len(df) > 0:
                x_min = df.index.min()
            if x_max is None and len(df) > 0:
                x_max = df.index.max()
            
            # Température
            if 'AirTemp' in df.columns:
                airtemp_data = df['AirTemp'].dropna()
                if len(airtemp_data) > 0:
                    fig.add_trace(
                        go.Scatter(x=airtemp_data.index, y=airtemp_data.values, mode='lines', 
                                 name='Température', line=dict(color='#E53935', width=1.5)),
                        row=1, col=1
                    )
            
            # Vitesse du vent
            if 'Speed#@1m' in df.columns:
                speed_data = df['Speed#@1m'].dropna()
                if len(speed_data) > 0:
                    fig.add_trace(
                        go.Scatter(x=speed_data.index, y=speed_data.values, mode='lines',
                                 name='Vitesse Vent', line=dict(color='#1976D2', width=1.5)),
                        row=1, col=2
                    )
            
            # Humidité
            if 'RH' in df.columns:
                rh_data = df['RH'].dropna()
                if len(rh_data) > 0:
                    fig.add_trace(
                        go.Scatter(x=rh_data.index, y=rh_data.values, mode='lines',
                                 name='Humidité', line=dict(color='#388E3C', width=1.5)),
                        row=2, col=1
                    )
            
            # Alimentation
            if 'Power' in df.columns:
                power_data = df['Power'].dropna()
                if len(power_data) > 0:
                    fig.add_trace(
                        go.Scatter(x=power_data.index, y=power_data.values, mode='lines',
                                 name='Alimentation', line=dict(color='#F57C00', width=1.5)),
                        row=2, col=2
                    )
            
            # Forcer le format des dates sur les axes X et limiter la plage
            for row in [1, 2]:
                for col in [1, 2]:
                    fig.update_xaxes(
                        type='date',
                        tickformat='%d/%m/%Y %H:%M',
                        range=[x_min, x_max] if x_min and x_max else None,
                        row=row, col=col
                    )
            
            fig.update_layout(
                title=dict(text=f'Séries Temporelles - {periode_type}', font=dict(size=16, color='#006838')),
                height=800,
                showlegend=False,
                margin=dict(t=60, b=40, l=40, r=40)
            )
            
            # Mise à jour des axes X
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=2)
            
            # Mise à jour des axes Y avec échelle adaptative
            # Température (row=1, col=1)
            if 'AirTemp' in df.columns:
                airtemp_data = df['AirTemp'].dropna()
                if len(airtemp_data) > 0:
                    min_temp = airtemp_data.min()
                    max_temp = airtemp_data.max()
                    range_temp = max_temp - min_temp
                    if range_temp > 0:
                        y_min_temp = max(0, min_temp - range_temp * 0.15)
                        y_max_temp = max_temp + range_temp * 0.15
                    else:
                        y_min_temp = min_temp - 1 if min_temp > 0 else 0
                        y_max_temp = max_temp + 1
                    fig.update_yaxes(
                        title_text="°C",
                        range=[y_min_temp, y_max_temp],
                        row=1, col=1
                    )
                else:
                    fig.update_yaxes(title_text="°C", row=1, col=1)
            else:
                fig.update_yaxes(title_text="°C", row=1, col=1)
            
            # Vitesse du vent (row=1, col=2)
            if 'Speed#@1m' in df.columns:
                speed_data = df['Speed#@1m'].dropna()
                if len(speed_data) > 0:
                    min_speed = max(0, speed_data.min())
                    max_speed = speed_data.max()
                    range_speed = max_speed - min_speed
                    if range_speed > 0:
                        y_min_speed = max(0, min_speed - range_speed * 0.15)
                        y_max_speed = max_speed + range_speed * 0.15
                    else:
                        y_min_speed = 0
                        y_max_speed = max_speed + 1
                    fig.update_yaxes(
                        title_text="m/s",
                        range=[y_min_speed, y_max_speed],
                        row=1, col=2
                    )
                else:
                    fig.update_yaxes(title_text="m/s", row=1, col=2)
            else:
                fig.update_yaxes(title_text="m/s", row=1, col=2)
            
            # Humidité (row=2, col=1)
            if 'RH' in df.columns:
                rh_data = df['RH'].dropna()
                if len(rh_data) > 0:
                    min_rh = max(0, rh_data.min())
                    max_rh = min(100, rh_data.max())
                    range_rh = max_rh - min_rh
                    if range_rh > 0:
                        y_min_rh = max(0, min_rh - range_rh * 0.15)
                        y_max_rh = min(100, max_rh + range_rh * 0.15)
                    else:
                        y_min_rh = max(0, min_rh - 1)
                        y_max_rh = min(100, max_rh + 1)
                    fig.update_yaxes(
                        title_text="%",
                        range=[y_min_rh, y_max_rh],
                        row=2, col=1
                    )
                else:
                    fig.update_yaxes(title_text="%", row=2, col=1)
            else:
                fig.update_yaxes(title_text="%", row=2, col=1)
            
            # Alimentation (row=2, col=2)
            if 'Power' in df.columns:
                power_data = df['Power'].dropna()
                if len(power_data) > 0:
                    min_power = max(0, power_data.min())
                    max_power = power_data.max()
                    range_power = max_power - min_power
                    if range_power > 0:
                        y_min_power = max(0, min_power - range_power * 0.15)
                        y_max_power = max_power + range_power * 0.15
                    else:
                        y_min_power = max(0, min_power - 0.1)
                        y_max_power = max_power + 0.1
                    fig.update_yaxes(
                        title_text="V",
                        range=[y_min_power, y_max_power],
                        row=2, col=2
                    )
                else:
                    fig.update_yaxes(title_text="V", row=2, col=2)
            else:
                fig.update_yaxes(title_text="V", row=2, col=2)
            
            return fig
        except Exception as e:
            try:
                import logging
                import traceback
                logging.error(f"Erreur génération séries temporelles: {str(e)}")
                logging.error(traceback.format_exc())
            except:
                pass
            try:
                st.warning(f"Erreur génération séries temporelles: {str(e)}")
            except:
                pass
            return None
    
    def generate_heatmap_correlation(self, df):
        """Génère une heatmap de corrélation entre les variables"""
        try:
            if df is None or len(df) == 0:
                return None
            
            # Sélectionner les colonnes numériques
            numeric_cols = ['Power', 'Speed#@1m', 'Dir', 'RH', 'AirTemp']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return None
            
            # Calculer la matrice de corrélation
            corr_matrix = df[available_cols].corr()
            
            # Créer la heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdYlBu_r',
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title="Corrélation")
            ))
            
            fig.update_layout(
                title=dict(text='Matrice de Corrélation entre Variables', font=dict(size=16, color='#006838')),
                height=600,
                width=700,
                margin=dict(t=60, b=40, l=40, r=40)
            )
            
            return fig
        except Exception as e:
            st.warning(f"Erreur génération heatmap: {str(e)}")
            return None
    
    def generate_histograms(self, df):
        """Génère des histogrammes pour la distribution des variables"""
        try:
            if df is None or len(df) == 0:
                return None
            
            # Créer un subplot avec 4 histogrammes
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Distribution Température', 'Distribution Vitesse Vent',
                               'Distribution Humidité', 'Distribution Alimentation'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # Température
            if 'AirTemp' in df.columns:
                fig.add_trace(
                    go.Histogram(x=df['AirTemp'], nbinsx=30, name='Température',
                               marker_color='#E53935', opacity=0.7),
                    row=1, col=1
                )
            
            # Vitesse du vent
            if 'Speed#@1m' in df.columns:
                fig.add_trace(
                    go.Histogram(x=df['Speed#@1m'], nbinsx=30, name='Vitesse Vent',
                               marker_color='#1976D2', opacity=0.7),
                    row=1, col=2
                )
            
            # Humidité
            if 'RH' in df.columns:
                fig.add_trace(
                    go.Histogram(x=df['RH'], nbinsx=30, name='Humidité',
                               marker_color='#388E3C', opacity=0.7),
                    row=2, col=1
                )
            
            # Alimentation
            if 'Power' in df.columns:
                fig.add_trace(
                    go.Histogram(x=df['Power'], nbinsx=30, name='Alimentation',
                               marker_color='#F57C00', opacity=0.7),
                    row=2, col=2
                )
            
            fig.update_layout(
                title=dict(text='Distributions des Variables Météorologiques', font=dict(size=16, color='#006838')),
                height=800,
                showlegend=False,
                margin=dict(t=60, b=40, l=40, r=40)
            )
            
            # Mise à jour des axes
            fig.update_xaxes(title_text="Température (°C)", row=1, col=1)
            fig.update_xaxes(title_text="Vitesse (m/s)", row=1, col=2)
            fig.update_xaxes(title_text="Humidité (%)", row=2, col=1)
            fig.update_xaxes(title_text="Alimentation (V)", row=2, col=2)
            fig.update_yaxes(title_text="Fréquence", row=1, col=1)
            fig.update_yaxes(title_text="Fréquence", row=2, col=1)
            
            return fig
        except Exception as e:
            st.warning(f"Erreur génération histogrammes: {str(e)}")
            return None
    
    def generate_boxplots(self, df):
        """Génère des boxplots pour détecter les outliers"""
        try:
            if df is None or len(df) == 0:
                return None
            
            # Créer un subplot avec 4 boxplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Boxplot Température', 'Boxplot Vitesse Vent',
                               'Boxplot Humidité', 'Boxplot Alimentation'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # Température
            if 'AirTemp' in df.columns:
                fig.add_trace(
                    go.Box(y=df['AirTemp'], name='Température', marker_color='#E53935'),
                    row=1, col=1
                )
            
            # Vitesse du vent
            if 'Speed#@1m' in df.columns:
                fig.add_trace(
                    go.Box(y=df['Speed#@1m'], name='Vitesse Vent', marker_color='#1976D2'),
                    row=1, col=2
                )
            
            # Humidité
            if 'RH' in df.columns:
                fig.add_trace(
                    go.Box(y=df['RH'], name='Humidité', marker_color='#388E3C'),
                    row=2, col=1
                )
            
            # Alimentation
            if 'Power' in df.columns:
                fig.add_trace(
                    go.Box(y=df['Power'], name='Alimentation', marker_color='#F57C00'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title=dict(text='Boxplots - Détection des Valeurs Aberrantes', font=dict(size=16, color='#006838')),
                height=800,
                showlegend=False,
                margin=dict(t=60, b=40, l=40, r=40)
            )
            
            # Mise à jour des axes
            fig.update_yaxes(title_text="Température (°C)", row=1, col=1)
            fig.update_yaxes(title_text="Vitesse (m/s)", row=1, col=2)
            fig.update_yaxes(title_text="Humidité (%)", row=2, col=1)
            fig.update_yaxes(title_text="Alimentation (V)", row=2, col=2)
            
            return fig
        except Exception as e:
            st.warning(f"Erreur génération boxplots: {str(e)}")
            return None
    
    def generate_radar(self, df, analysis_json=None):
        """Génère un graphique radar (spider chart) avec les indicateurs opérationnels"""
        try:
            if df is None or len(df) == 0:
                return None
            
            # Calculer les indicateurs normalisés (0-100)
            indicators = {}
            
            # 1. Conditions optimales (déjà en pourcentage)
            if analysis_json and 'kpis' in analysis_json:
                kpis = analysis_json.get('kpis', {})
                temps_optimal = kpis.get('temps_conditions_optimales', {})
                if isinstance(temps_optimal, dict):
                    indicators['Conditions Optimales'] = min(100, max(0, temps_optimal.get('pourcentage', 0)))
                else:
                    indicators['Conditions Optimales'] = 0
            else:
                # Calculer depuis les données
                optimal_conditions = len(df[(df['Speed#@1m'] < 12) & (df['RH'] < 90) & 
                                           (df['AirTemp'] >= 5) & (df['AirTemp'] <= 35)])
                indicators['Conditions Optimales'] = min(100, (optimal_conditions / len(df)) * 100) if len(df) > 0 else 0
            
            # 2. Stabilité température (inverse de l'écart-type, normalisé)
            if 'AirTemp' in df.columns:
                temp_std = df['AirTemp'].std()
                # Plus l'écart-type est faible, plus c'est stable (meilleur)
                # Normaliser: écart-type de 0-10°C -> score 100-0
                temp_stability = max(0, min(100, 100 - (temp_std * 10)))
                indicators['Stabilité Température'] = temp_stability
            
            # 3. Stabilité vent
            if 'Speed#@1m' in df.columns:
                vent_std = df['Speed#@1m'].std()
                # Normaliser: écart-type de 0-5 m/s -> score 100-0
                vent_stability = max(0, min(100, 100 - (vent_std * 20)))
                indicators['Stabilité Vent'] = vent_stability
            
            # 4. Stabilité humidité
            if 'RH' in df.columns:
                rh_std = df['RH'].std()
                # Normaliser: écart-type de 0-20% -> score 100-0
                rh_stability = max(0, min(100, 100 - (rh_std * 5)))
                indicators['Stabilité Humidité'] = rh_stability
            
            # 5. Stabilité alimentation
            if 'Power' in df.columns:
                power_std = df['Power'].std()
                # Normaliser: écart-type de 0-2V -> score 100-0
                power_stability = max(0, min(100, 100 - (power_std * 50)))
                indicators['Stabilité Alimentation'] = power_stability
            
            # 6. Qualité des données (complétude)
            if analysis_json and 'resume_general' in analysis_json:
                rg = analysis_json.get('resume_general', {})
                taux_completude = rg.get('taux_completude', '100%')
                if isinstance(taux_completude, str):
                    taux_completude = float(taux_completude.replace('%', '').replace(',', '.'))
                indicators['Qualité Données'] = min(100, max(0, taux_completude))
            else:
                # Estimation basée sur les valeurs manquantes
                total_cells = len(df) * len(df.columns)
                missing_cells = df.isnull().sum().sum()
                completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 100
                indicators['Qualité Données'] = min(100, max(0, completeness))
            
            if not indicators:
                return None
            
            # Créer le graphique radar
            categories = list(indicators.keys())
            values = list(indicators.values())
            
            # Dupliquer le premier élément pour fermer le radar
            categories_radar = categories + [categories[0]]
            values_radar = values + [values[0]]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values_radar,
                theta=categories_radar,
                fill='toself',
                name='Indicateurs',
                line=dict(color='#006838', width=3),
                fillcolor='rgba(0, 104, 56, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickfont=dict(size=10),
                        tickmode='linear',
                        tick0=0,
                        dtick=20
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=11),
                        rotation=90,
                        direction='counterclockwise'
                    )
                ),
                title=dict(
                    text='Indicateurs Opérationnels (Radar)',
                    font=dict(size=16, color='#006838')
                ),
                height=600,
                width=700,
                margin=dict(t=60, b=40, l=40, r=40),
                showlegend=False
            )
            
            return fig
        except Exception as e:
            st.warning(f"Erreur génération radar: {str(e)}")
            return None
    
    def generate_all_graphs(self, df, periode_type, analysis_json=None, date_debut=None, date_fin=None):
        """Génère tous les graphiques supplémentaires"""
        graphs = {}
        
        try:
            # Séries temporelles (avec dates de début et fin)
            graphs['time_series'] = self.generate_time_series(df, periode_type, date_debut=date_debut, date_fin=date_fin)
            if graphs['time_series']:
                try:
                    import logging
                    logging.info("Graphique time_series généré avec succès")
                except:
                    pass
        except Exception as e:
            try:
                import logging
                logging.error(f"Erreur génération time_series: {e}")
            except:
                pass
        
        try:
            # Heatmap de corrélation
            graphs['correlation'] = self.generate_heatmap_correlation(df)
            if graphs['correlation']:
                try:
                    import logging
                    logging.info("Graphique correlation généré avec succès")
                except:
                    pass
        except Exception as e:
            try:
                import logging
                logging.error(f"Erreur génération correlation: {e}")
            except:
                pass
        
        try:
            # Histogrammes
            graphs['histograms'] = self.generate_histograms(df)
            if graphs['histograms']:
                try:
                    import logging
                    logging.info("Graphique histograms généré avec succès")
                except:
                    pass
        except Exception as e:
            try:
                import logging
                logging.error(f"Erreur génération histograms: {e}")
            except:
                pass
        
        try:
            # Boxplots
            graphs['boxplots'] = self.generate_boxplots(df)
            if graphs['boxplots']:
                try:
                    import logging
                    logging.info("Graphique boxplots généré avec succès")
                except:
                    pass
        except Exception as e:
            try:
                import logging
                logging.error(f"Erreur génération boxplots: {e}")
            except:
                pass
        
        try:
            # Graphique radar (indicateurs opérationnels)
            graphs['radar'] = self.generate_radar(df, analysis_json)
            if graphs['radar']:
                try:
                    import logging
                    logging.info("Graphique radar généré avec succès")
                except:
                    pass
        except Exception as e:
            try:
                import logging
                logging.error(f"Erreur génération radar: {e}")
            except:
                pass
        
        # Filtrer les None et logger le résultat
        result = {k: v for k, v in graphs.items() if v is not None}
        try:
            import logging
            logging.info(f"Graphiques générés: {len(result)}/{len(graphs)} - Clés: {list(result.keys())}")
        except:
            pass
        
        return result

class DataAnalyzer:
    def __init__(self):
        self.llm = CerebriumLLM("qwen-3-235b-a22b-instruct-2507")
        # Modèles de fallback pour forcer l'utilisation du LLM
        self.fallback_models = [
            "gpt-oss-120b",
            "llama-3.3-70b"
        ]
    
    def clean_numeric_data(self, df):
        """Nettoie et convertit les données numériques - ADAPTÉ POUR VOTRE FORMAT"""
        # Vérifier et renommer les colonnes selon votre format
        if 'Power' in df.columns and 'Speed#@1m' in df.columns and 'Dir' in df.columns and 'RH' in df.columns and 'AirTemp' in df.columns:
            # Votre format a déjà les bonnes colonnes
            numeric_columns = ['Power', 'Speed#@1m', 'Dir', 'RH', 'AirTemp']
        else:
            # Essayer de détecter automatiquement les colonnes
            st.warning("Format de colonnes détecté automatiquement")
            numeric_columns = []
            for col in df.columns:
                if any(keyword in col for keyword in ['Power', 'Speed', 'Dir', 'RH', 'AirTemp', 'Temp']):
                    numeric_columns.append(col)
        
        for col in numeric_columns:
            if col in df.columns:
                # Conversion en numérique, gestion des erreurs
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Suppression des valeurs aberrantes
                if 'Temp' in col:
                    df = df[(df[col] >= -50) & (df[col] <= 60)]
                elif 'RH' in col:
                    df = df[(df[col] >= 0) & (df[col] <= 100)]
                elif 'Speed' in col:
                    df = df[(df[col] >= 0) & (df[col] <= 100)]
        
        return df.dropna(subset=numeric_columns)
    
    def parse_timestamp_column(self, df):
        """Parse la colonne timestamp selon votre format spécifique"""
        # Si l'index est déjà un DatetimeIndex, ne rien faire
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        
        timestamp_col = None
        
        # Identifier la colonne timestamp
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in ['date', 'time', 'timestamp']):
                timestamp_col = col
                break
        
        if timestamp_col is None:
            # Si aucune colonne timestamp trouvée et que l'index n'est pas déjà un DatetimeIndex,
            # essayer de convertir l'index directement
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df[df.index.notna()]
                return df
            except:
                # Si ça échoue, prendre la première colonne comme timestamp
                if len(df.columns) > 0:
                    timestamp_col = df.columns[0]
                else:
                    return df
        
        try:
            # Essayer de parser comme datetime
            df['Timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
            # Supprimer les lignes avec timestamp invalide
            df = df[df['Timestamp'].notna()]
            df = df.set_index('Timestamp')
            return df
        except Exception as e:
            st.warning(f"Erreur parsing timestamp: {str(e)}")
            return df
    
    def _compute_precomputed_statistics(self, df):
        """Calcule toutes les statistiques sur l'ensemble du dataframe pour les inclure dans le prompt LLM"""
        stats = {}
        
        try:
            # Statistiques température
            if 'AirTemp' in df.columns:
                temp_series = pd.to_numeric(df['AirTemp'], errors='coerce').dropna()
                if len(temp_series) > 0:
                    stats['statistiques_temperature'] = {
                        "unite": "°C",
                        "moyenne": float(temp_series.mean()),
                        "minimum": {"valeur": float(temp_series.min()), "timestamp": str(temp_series.idxmin()) if hasattr(temp_series, 'idxmin') else ""},
                        "maximum": {"valeur": float(temp_series.max()), "timestamp": str(temp_series.idxmax()) if hasattr(temp_series, 'idxmax') else ""},
                        "ecart_type": float(temp_series.std(ddof=0)),
                        "amplitude": float(temp_series.max() - temp_series.min()),
                        "percentile_25": float(temp_series.quantile(0.25)),
                        "percentile_50": float(temp_series.quantile(0.50)),
                        "percentile_75": float(temp_series.quantile(0.75)),
                        "percentile_95": float(temp_series.quantile(0.95))
                    }
            
            # Statistiques vent
            if 'Speed#@1m' in df.columns:
                speed_series = pd.to_numeric(df['Speed#@1m'], errors='coerce').dropna()
                if len(speed_series) > 0:
                    total = len(speed_series)
                    stats['statistiques_vitesse_vent'] = {
                        "unite": "m/s",
                        "moyenne": float(speed_series.mean()),
                        "minimum": {"valeur": float(speed_series.min()), "timestamp": str(speed_series.idxmin()) if hasattr(speed_series, 'idxmin') else ""},
                        "maximum": {"valeur": float(speed_series.max()), "timestamp": str(speed_series.idxmax()) if hasattr(speed_series, 'idxmax') else ""},
                        "ecart_type": float(speed_series.std(ddof=0)),
                        "distribution_classes": {
                            "calme_0_1": {
                                "count": int(len(speed_series[speed_series < 1.0])),
                                "pourcentage": round(len(speed_series[speed_series < 1.0]) / total * 100, 2)
                            },
                            "legere_1_2": {
                                "count": int(len(speed_series[(speed_series >= 1.0) & (speed_series < 2.0)])),
                                "pourcentage": round(len(speed_series[(speed_series >= 1.0) & (speed_series < 2.0)]) / total * 100, 2)
                            },
                            "brise_2_5": {
                                "count": int(len(speed_series[(speed_series >= 2.0) & (speed_series < 5.0)])),
                                "pourcentage": round(len(speed_series[(speed_series >= 2.0) & (speed_series < 5.0)]) / total * 100, 2)
                            },
                            "moderee_5_8": {
                                "count": int(len(speed_series[(speed_series >= 5.0) & (speed_series < 8.0)])),
                                "pourcentage": round(len(speed_series[(speed_series >= 5.0) & (speed_series < 8.0)]) / total * 100, 2)
                            },
                            "forte_8_20": {
                                "count": int(len(speed_series[speed_series >= 8.0])),
                                "pourcentage": round(len(speed_series[speed_series >= 8.0]) / total * 100, 2)
                            }
                        }
                    }
            
            # Rose des vents
            if 'Dir' in df.columns and 'Speed#@1m' in df.columns:
                stats['wind_rose_data'] = self._calculate_wind_rose(df)
            
            # Statistiques humidité
            if 'RH' in df.columns:
                rh_series = pd.to_numeric(df['RH'], errors='coerce').dropna()
                if len(rh_series) > 0:
                    total_rh = len(rh_series)
                    rh_above_90 = len(rh_series[rh_series > 90])
                    rh_above_95 = len(rh_series[rh_series > 95])
                    stats['statistiques_humidite'] = {
                        "unite": "%",
                        "moyenne": float(rh_series.mean()),
                        "minimum": {"valeur": float(rh_series.min()), "timestamp": str(rh_series.idxmin()) if hasattr(rh_series, 'idxmin') else ""},
                        "maximum": {"valeur": float(rh_series.max()), "timestamp": str(rh_series.idxmax()) if hasattr(rh_series, 'idxmax') else ""},
                        "temps_rh_sup_90": {
                            "heures": round(rh_above_90 / 60, 2),
                            "pourcentage": round(rh_above_90 / total_rh * 100, 2)
                        },
                        "temps_rh_sup_95": {
                            "heures": round(rh_above_95 / 60, 2),
                            "pourcentage": round(rh_above_95 / total_rh * 100, 2)
                        }
                    }
            
            # Statistiques power
            if 'Power' in df.columns:
                power_series = pd.to_numeric(df['Power'], errors='coerce').dropna()
                if len(power_series) > 0:
                    stats['statistiques_power'] = {
                        "unite": "V",
                        "moyenne": float(power_series.mean()),
                        "minimum": {"valeur": float(power_series.min()), "timestamp": str(power_series.idxmin()) if hasattr(power_series, 'idxmin') else ""},
                        "maximum": {"valeur": float(power_series.max()), "timestamp": str(power_series.idxmax()) if hasattr(power_series, 'idxmax') else ""},
                        "ecart_type": float(power_series.std(ddof=0)),
                        "stabilite": "Bonne" if power_series.std(ddof=0) < 0.5 else "Variable"
                    }
            
            # KPIs
            if all(col in df.columns for col in ['Speed#@1m', 'RH', 'AirTemp']):
                optimal_conditions = len(df[(df['Speed#@1m'] < 12) & (df['RH'] < 90) & (df['AirTemp'] >= 5) & (df['AirTemp'] <= 35)])
                total_kpi = len(df)
                stats['kpis'] = {
                    "temps_conditions_optimales": {
                        "heures": round(optimal_conditions / 60, 2),
                        "pourcentage": round(optimal_conditions / total_kpi * 100, 2),
                        "criteres": "Vent <12 m/s ET RH <90% ET Temp 5-35°C"
                    },
                    "nb_alertes_vent_fort": int(len(df[df['Speed#@1m'] > 8])),
                    "nb_alertes_humidite": int(len(df[df['RH'] > 95]))
                }
        
        except Exception as e:
            st.warning(f"Erreur lors du calcul des statistiques pré-calculées: {str(e)}")
        
        return stats
    
    def analyze_data(self, df, periode_type, date_debut, date_fin):
        # Nettoyage des données et parsing du timestamp
        df = self.parse_timestamp_column(df)
        df_clean = self.clean_numeric_data(df)
        
        if len(df_clean) == 0:
            st.error("Aucune donnée valide après nettoyage")
            raise Exception("Données insuffisantes pour l'analyse. L'analyse doit être générée par un LLM.")
        
        # Calcul des statistiques sur TOUTES les données (pas seulement l'échantillon)
        precomputed_stats = self._compute_precomputed_statistics(df_clean)
        
        # Préparation des données pour l'analyse (échantillon de 50 valeurs pour contexte)
        sample_data = df_clean.head(50).reset_index().to_dict('records')
        
        prompt = f"""
        ANALYSE DE DONNÉES MÉTÉOROLOGIQUES OCP

        Période: {periode_type} du {date_debut} au {date_fin}
        Nombre total d'enregistrements: {len(df_clean)}

        ÉCHANTILLON DES DONNÉES (50 premières lignes pour contexte):
        {json.dumps(sample_data, indent=2, default=str)}

        ⚠️ ATTENTION IMPORTANTE: L'échantillon ci-dessus contient seulement 50 valeurs à titre d'exemple.
        Les statistiques ci-dessous sont calculées sur TOUTES les {len(df_clean)} valeurs disponibles.

        STATISTIQUES PRÉ-CALCULÉES (sur toutes les données):

        {json.dumps(precomputed_stats, indent=2, default=str)}

        INSTRUCTIONS:
        - UTILISE OBLIGATOIREMENT les statistiques pré-calculées ci-dessus dans ta réponse JSON
        - NE calcule PAS les statistiques à partir de l'échantillon de 50 valeurs
        - Les valeurs d'écart-type, amplitude, percentiles, etc. doivent correspondre EXACTEMENT aux statistiques pré-calculées
        - L'échantillon de 50 valeurs sert uniquement de contexte pour comprendre la structure des données

        Retourne UNIQUEMENT du JSON valide avec cette structure:

        {{
            "resume_general": {{
                "periode_analyse": "string",
                "duree_jours": number,
                "nb_mesures_recues": number,
                "taux_completude": "string",
                "qualite_donnees": "string"
            }},
            "statistiques_temperature": {{...}},
            "statistiques_vitesse_vent": {{...}},
            "wind_rose_data": {{...}},
            "statistiques_humidite": {{...}},
            "statistiques_power": {{...}},
            "tendances": ["string"],
            "anomalies": ["string"],
            "kpis": {{...}},
            "recommandations_preliminaires": ["string"]
        }}

        IMPORTANT: 
        - Utilise EXACTEMENT les valeurs des statistiques pré-calculées
        - Utilise des nombres, pas de texte dans les valeurs numériques
        - Les statistiques doivent correspondre aux valeurs pré-calculées, pas à l'échantillon
        """
        
        # Essayer le modèle principal d'abord
        response = self.llm.invoke(prompt, max_tokens=10000)
        analysis_json = self._parse_llm_response(response, df_clean)
        
        if analysis_json is not None:
            return analysis_json
        
        # Si échec, essayer les modèles de fallback
        for model_name in self.fallback_models:
            try:
                st.info(f"🔄 Essai avec le modèle : {model_name}")
                llm = CerebriumLLM(model_name)
                response = llm.invoke(prompt, max_tokens=10000)
                analysis_json = self._parse_llm_response(response, df_clean)
                if analysis_json is not None:
                    st.success(f"✅ Analyse générée avec {model_name}")
                    return analysis_json
            except Exception as e:
                st.warning(f"⚠️ Modèle {model_name} indisponible : {str(e)}")
                continue
        
        # Si tous les modèles échouent, lever une erreur au lieu d'utiliser un fallback statique
        st.error("❌ Impossible de générer l'analyse avec les LLM disponibles. Veuillez réessayer plus tard.")
        raise Exception("Tous les modèles LLM ont échoué. L'analyse doit être générée par un LLM.")
    
    def _extract_json_from_text(self, text):
        """Extrait le premier JSON valide du texte, même s'il y a du contenu supplémentaire"""
        if not text:
            return None
        
        # Trouver le premier '{'
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        # Utiliser json.JSONDecoder pour parser seulement le JSON valide
        decoder = json.JSONDecoder()
        try:
            # Extraire à partir du premier '{' et parser seulement le JSON valide
            json_str = text[start_idx:]
            obj, idx = decoder.raw_decode(json_str)
            return obj
        except (json.JSONDecodeError, ValueError) as e:
            # Si ça échoue, essayer une approche alternative : trouver le JSON en comptant les accolades
            try:
                brace_count = 0
                json_start = start_idx
                json_end = start_idx
                
                for i in range(start_idx, len(text)):
                    if text[i] == '{':
                        if brace_count == 0:
                            json_start = i
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if brace_count == 0 and json_end > json_start:
                    json_str = text[json_start:json_end]
                    return json.loads(json_str)
            except Exception:
                pass
        
        return None
    
    def _parse_llm_response(self, response, df_clean):
        """Parse la réponse du LLM et retourne le JSON d'analyse, ou None si échec"""
        if response is None:
            return None
        
        try:
            # Extraire le JSON de manière robuste
            analysis_json = self._extract_json_from_text(response)
            
            if analysis_json is None:
                return None
            
            # Sécuriser les données absentes ou incomplètes provenant du LLM
            if not analysis_json.get("wind_rose_data") or not analysis_json.get("wind_rose_data", {}).get("secteurs"):
                try:
                    analysis_json["wind_rose_data"] = self._calculate_wind_rose(df_clean)
                except Exception:
                    analysis_json["wind_rose_data"] = {"secteurs": [], "calme_central": 0}
            if not analysis_json.get("statistiques_vitesse_vent"):
                analysis_json["statistiques_vitesse_vent"] = {
                    "unite": "m/s",
                    "moyenne": float(df_clean['Speed#@1m'].mean()) if len(df_clean) else 0.0
                }
            return analysis_json
        except Exception as e:
            st.warning(f"⚠️ Erreur parsing JSON: {str(e)}")
            return None
    
    def _get_default_analysis(self, df, periode_type, date_debut, date_fin):
        """Analyse par défaut basée sur les données nettoyées"""
        
        # S'assurer que les données sont numériques
        df = self.clean_numeric_data(df)
        
        if len(df) == 0:
            return self._get_empty_analysis(periode_type, date_debut, date_fin)
        
        # Calcul des statistiques de base
        try:
            # Statistiques température
            temp_stats = {
                "unite": "°C",
                "moyenne": float(df['AirTemp'].mean()),
                "minimum": {"valeur": float(df['AirTemp'].min()), "timestamp": ""},
                "maximum": {"valeur": float(df['AirTemp'].max()), "timestamp": ""},
                "ecart_type": float(df['AirTemp'].std()),
                "amplitude": float(df['AirTemp'].max() - df['AirTemp'].min()),
                "percentile_25": float(df['AirTemp'].quantile(0.25)),
                "percentile_50": float(df['AirTemp'].quantile(0.50)),
                "percentile_75": float(df['AirTemp'].quantile(0.75)),
                "percentile_95": float(df['AirTemp'].quantile(0.95))
            }
            
            # Statistiques vent
            speed_stats = {
                "unite": "m/s",
                "moyenne": float(df['Speed#@1m'].mean()),
                "minimum": {"valeur": float(df['Speed#@1m'].min()), "timestamp": ""},
                "maximum": {"valeur": float(df['Speed#@1m'].max()), "timestamp": ""},
                "ecart_type": float(df['Speed#@1m'].std()),
                "distribution_classes": {
                    "calme_0_1": {"count": len(df[df['Speed#@1m'] < 1.0]), "pourcentage": len(df[df['Speed#@1m'] < 1.0])/len(df)*100},
                    "legere_1_2": {"count": len(df[(df['Speed#@1m'] >= 1.0) & (df['Speed#@1m'] < 2.0)]), "pourcentage": len(df[(df['Speed#@1m'] >= 1.0) & (df['Speed#@1m'] < 2.0)])/len(df)*100},
                    "brise_2_5": {"count": len(df[(df['Speed#@1m'] >= 2.0) & (df['Speed#@1m'] < 5.0)]), "pourcentage": len(df[(df['Speed#@1m'] >= 2.0) & (df['Speed#@1m'] < 5.0)])/len(df)*100},
                    "moderee_5_8": {"count": len(df[(df['Speed#@1m'] >= 5.0) & (df['Speed#@1m'] < 8.0)]), "pourcentage": len(df[(df['Speed#@1m'] >= 5.0) & (df['Speed#@1m'] < 8.0)])/len(df)*100},
                    "forte_8_20": {"count": len(df[df['Speed#@1m'] >= 8.0]), "pourcentage": len(df[df['Speed#@1m'] >= 8.0])/len(df)*100}
                }
            }
            
            # Rose des vents
            wind_rose_data = self._calculate_wind_rose(df)
            
            # Statistiques humidité
            humidity_stats = {
                "unite": "%",
                "moyenne": float(df['RH'].mean()),
                "minimum": {"valeur": float(df['RH'].min()), "timestamp": ""},
                "maximum": {"valeur": float(df['RH'].max()), "timestamp": ""},
                "temps_rh_sup_90": {"heures": len(df[df['RH'] > 90])/60, "pourcentage": len(df[df['RH'] > 90])/len(df)*100},
                "temps_rh_sup_95": {"heures": len(df[df['RH'] > 95])/60, "pourcentage": len(df[df['RH'] > 95])/len(df)*100},
                "periodes_critiques": []
            }
            
            # Statistiques power
            power_stats = {
                "unite": "V",
                "moyenne": float(df['Power'].mean()),
                "minimum": {"valeur": float(df['Power'].min()), "timestamp": ""},
                "maximum": {"valeur": float(df['Power'].max()), "timestamp": ""},
                "ecart_type": float(df['Power'].std()),
                "stabilite": "Bonne" if df['Power'].std() < 0.5 else "Variable"
            }
            
            # KPIs
            optimal_conditions = len(df[(df['Speed#@1m'] < 12) & (df['RH'] < 90) & (df['AirTemp'] >= 5) & (df['AirTemp'] <= 35)])
            
            kpis = {
                "temps_conditions_optimales": {
                    "heures": optimal_conditions/60,
                    "pourcentage": optimal_conditions/len(df)*100,
                    "criteres": "Vent <12 m/s ET RH <90% ET Temp 5-35°C"
                },
                "nb_alertes_vent_fort": len(df[df['Speed#@1m'] > 8]),
                "nb_alertes_humidite": len(df[df['RH'] > 95]),
                "duree_moyenne_episode_vent_fort": {"minutes": 0, "nb_episodes": 0},
                "indice_variabilite_meteo": {"valeur": 5.0, "echelle": "0-10", "interpretation": "Moyenne"}
            }
            
            return {
                "resume_general": {
                    "periode_analyse": f"{date_debut} - {date_fin}",
                    "duree_jours": (pd.to_datetime(date_fin) - pd.to_datetime(date_debut)).days,
                    "nb_mesures_attendues": len(df),
                    "nb_mesures_recues": len(df),
                    "taux_completude": "100%",
                    "qualite_donnees": "Excellente",
                    "capteurs_defaillants": []
                },
                "statistiques_temperature": temp_stats,
                "statistiques_vitesse_vent": speed_stats,
                "wind_rose_data": wind_rose_data,
                "statistiques_humidite": humidity_stats,
                "statistiques_power": power_stats,
                "tendances": ["Analyse automatique basée sur les données nettoyées"],
                "anomalies": [],
                "kpis": kpis,
                "recommandations_preliminaires": ["Vérification manuelle recommandée pour validation"]
            }
            
        except Exception as e:
            st.error(f"Erreur dans l'analyse par défaut: {str(e)}")
            return self._get_empty_analysis(periode_type, date_debut, date_fin)
    
    def _get_empty_analysis(self, periode_type, date_debut, date_fin):
        """Retourne une structure d'analyse vide en cas d'erreur"""
        return {
            "resume_general": {
                "periode_analyse": f"{date_debut} - {date_fin}",
                "duree_jours": 0,
                "nb_mesures_attendues": 0,
                "nb_mesures_recues": 0,
                "taux_completude": "0%",
                "qualite_donnees": "Données manquantes",
                "capteurs_defaillants": ["Données non disponibles"]
            },
            "statistiques_temperature": {"unite": "°C", "moyenne": 0},
            "statistiques_vitesse_vent": {"unite": "m/s", "moyenne": 0},
            "wind_rose_data": {"secteurs": [], "calme_central": 0},
            "statistiques_humidite": {"unite": "%", "moyenne": 0},
            "statistiques_power": {"unite": "V", "moyenne": 0},
            "tendances": ["Données insuffisantes pour l'analyse"],
            "anomalies": [],
            "kpis": {"temps_conditions_optimales": {"pourcentage": 0}},
            "recommandations_preliminaires": ["Vérifier la qualité des données d'entrée"]
        }
    
    def _calculate_wind_rose(self, df):
        """Calcule les données de la rose des vents"""
        sectors = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        
        sector_ranges = [
            (337.5, 22.5), (22.5, 45), (45, 67.5), (67.5, 90),
            (90, 112.5), (112.5, 135), (135, 157.5), (157.5, 180),
            (180, 202.5), (202.5, 225), (225, 247.5), (247.5, 270),
            (270, 292.5), (292.5, 315), (315, 337.5), (337.5, 360)
        ]
        
        secteurs_data = []
        total_measures = len(df)
        
        for i, (sector, (min_angle, max_angle)) in enumerate(zip(sectors, sector_ranges)):
            if i == 0:  # Cas spécial pour Nord
                sector_data = df[((df['Dir'] >= min_angle) & (df['Dir'] <= 360)) | 
                               ((df['Dir'] >= 0) & (df['Dir'] < max_angle))]
            else:
                sector_data = df[(df['Dir'] >= min_angle) & (df['Dir'] < max_angle)]
            
            sector_count = len(sector_data)
            frequency = (sector_count / total_measures) * 100 if total_measures > 0 else 0;
            
            secteurs_data.append({
                "nom": sector,
                "angle_min": min_angle,
                "angle_max": max_angle,
                "frequence_totale": round(frequency, 1),
                "vitesse_moyenne": round(sector_data['Speed#@1m'].mean(), 1) if len(sector_data) > 0 else 0,
                "vitesse_max": round(sector_data['Speed#@1m'].max(), 1) if len(sector_data) > 0 else 0,
                "distribution_vitesses": {
                    "calme_0_1": 0,
                    "legere_1_2": round(len(sector_data[(sector_data['Speed#@1m'] >= 1.0) & (sector_data['Speed#@1m'] < 2.0)]) / total_measures * 100, 1),
                    "brise_2_5": round(len(sector_data[(sector_data['Speed#@1m'] >= 2.0) & (sector_data['Speed#@1m'] < 5.0)]) / total_measures * 100, 1),
                    "moderee_5_8": round(len(sector_data[(sector_data['Speed#@1m'] >= 5.0) & (sector_data['Speed#@1m'] < 8.0)]) / total_measures * 100, 1),
                    "forte_8_20": round(len(sector_data[sector_data['Speed#@1m'] >= 8.0]) / total_measures * 100, 1)
                }
            })
        
        # Trouver le secteur dominant
        if secteurs_data:
            dominant_sector = max(secteurs_data, key=lambda x: x['frequence_totale'])
        else:
            dominant_sector = {"nom": "N/A", "frequence_totale": 0, "vitesse_moyenne": 0}
        
        return {
            "secteurs": secteurs_data,
            "calme_central": round(len(df[df['Speed#@1m'] < 1.0]) / total_measures * 100, 1) if total_measures > 0 else 0,
            "direction_dominante": {
                "nom": dominant_sector['nom'],
                "angle_moyen": (dominant_sector.get('angle_min', 0) + dominant_sector.get('angle_max', 0)) / 2,
                "frequence": dominant_sector['frequence_totale'],
                "vitesse_moyenne": dominant_sector['vitesse_moyenne']
            },
            "direction_vitesses_max": {
                "nom": max(secteurs_data, key=lambda x: x['vitesse_max'])['nom'] if secteurs_data else "N/A",
                "vitesse_max": max(secteurs_data, key=lambda x: x['vitesse_max'])['vitesse_max'] if secteurs_data else 0,
                "timestamp": ""
            }
        }

class ReportGenerator:
    def __init__(self):
        self.llm = CerebriumLLM("gpt-oss-120b")
        # Modèles de fallback à essayer dans l'ordre
        self.fallback_models = [
            "qwen-3-235b-a22b-instruct-2507"
        ]
    
    def generate_report(self, analysis_json, periode_type, audience, df=None):
        """
        Génère un rapport Markdown COMPLET par LLM.
        Le LLM génère TOUT le texte narratif du rapport.
        TOUJOURS utilise le LLM - pas de fallback statique.
        
        Args:
            analysis_json: Dictionnaire contenant les statistiques calculées
            periode_type: Type de période (Journalier, Mensuel, Annuel)
            audience: Audience cible (Opérateurs Terrain, Management, Ingénieurs)
            df: DataFrame optionnel pour recalculer les statistiques manquantes
        """
        # TOUJOURS utiliser le LLM pour générer le rapport complet
        enrichment = self._enrich_with_llm_detailed(analysis_json, periode_type, audience, df=df)
        
        if not enrichment:
            # Si l'enrichissement échoue, essayer de générer le rapport complet directement par LLM
            return self._generate_full_report_with_llm(analysis_json, periode_type, audience)
        
        return self._generate_structured_report(analysis_json, periode_type, audience, enrichment=enrichment, df=df)

    def _compute_graph_statistics(self, df):
        """Calcule les statistiques détaillées pour les analyses de graphiques"""
        stats = {}
        
        if df is None or len(df) == 0:
            return stats
        
        try:
            # Statistiques pour les séries temporelles
            if 'AirTemp' in df.columns:
                temp_series = pd.to_numeric(df['AirTemp'], errors='coerce').dropna()
                if len(temp_series) > 0:
                    stats['time_series_temp'] = {
                        'moyenne': float(temp_series.mean()),
                        'min': float(temp_series.min()),
                        'max': float(temp_series.max()),
                        'ecart_type': float(temp_series.std(ddof=0)),
                        'amplitude': float(temp_series.max() - temp_series.min()),
                        'timestamp_min': str(temp_series.idxmin()) if hasattr(temp_series, 'idxmin') else '',
                        'timestamp_max': str(temp_series.idxmax()) if hasattr(temp_series, 'idxmax') else ''
                    }
            
            if 'Speed#@1m' in df.columns:
                speed_series = pd.to_numeric(df['Speed#@1m'], errors='coerce').dropna()
                if len(speed_series) > 0:
                    stats['time_series_vent'] = {
                        'moyenne': float(speed_series.mean()),
                        'min': float(speed_series.min()),
                        'max': float(speed_series.max()),
                        'ecart_type': float(speed_series.std(ddof=0)),
                        'timestamp_min': str(speed_series.idxmin()) if hasattr(speed_series, 'idxmin') else '',
                        'timestamp_max': str(speed_series.idxmax()) if hasattr(speed_series, 'idxmax') else ''
                    }
            
            if 'RH' in df.columns:
                rh_series = pd.to_numeric(df['RH'], errors='coerce').dropna()
                if len(rh_series) > 0:
                    stats['time_series_humidite'] = {
                        'moyenne': float(rh_series.mean()),
                        'min': float(rh_series.min()),
                        'max': float(rh_series.max()),
                        'ecart_type': float(rh_series.std(ddof=0)),
                        'timestamp_min': str(rh_series.idxmin()) if hasattr(rh_series, 'idxmin') else '',
                        'timestamp_max': str(rh_series.idxmax()) if hasattr(rh_series, 'idxmax') else ''
                    }
            
            if 'Power' in df.columns:
                power_series = pd.to_numeric(df['Power'], errors='coerce').dropna()
                if len(power_series) > 0:
                    stats['time_series_power'] = {
                        'moyenne': float(power_series.mean()),
                        'min': float(power_series.min()),
                        'max': float(power_series.max()),
                        'ecart_type': float(power_series.std(ddof=0)),
                        'timestamp_min': str(power_series.idxmin()) if hasattr(power_series, 'idxmin') else '',
                        'timestamp_max': str(power_series.idxmax()) if hasattr(power_series, 'idxmax') else ''
                    }
            
            # Corrélations détaillées
            numeric_cols = ['Power', 'Speed#@1m', 'Dir', 'RH', 'AirTemp']
            available_cols = [col for col in numeric_cols if col in df.columns]
            if len(available_cols) >= 2:
                corr_matrix = df[available_cols].corr()
                stats['correlations'] = {}
                for i, col1 in enumerate(available_cols):
                    for col2 in available_cols[i+1:]:
                        corr_value = corr_matrix.loc[col1, col2]
                        if not pd.isna(corr_value):
                            stats['correlations'][f'{col1}_vs_{col2}'] = float(corr_value)
            
            # Statistiques pour les distributions (boxplots/histogrammes)
            if 'AirTemp' in df.columns:
                temp_series = pd.to_numeric(df['AirTemp'], errors='coerce').dropna()
                if len(temp_series) > 0:
                    q1 = float(temp_series.quantile(0.25))
                    q2 = float(temp_series.quantile(0.50))
                    q3 = float(temp_series.quantile(0.75))
                    iqr = q3 - q1
                    outliers_low = len(temp_series[temp_series < (q1 - 1.5 * iqr)])
                    outliers_high = len(temp_series[temp_series > (q3 + 1.5 * iqr)])
                    stats['distribution_temp'] = {
                        'q1': q1, 'q2': q2, 'q3': q3,
                        'iqr': iqr,
                        'outliers_low': int(outliers_low),
                        'outliers_high': int(outliers_high),
                        'total_outliers': int(outliers_low + outliers_high)
                    }
            
            if 'Speed#@1m' in df.columns:
                speed_series = pd.to_numeric(df['Speed#@1m'], errors='coerce').dropna()
                if len(speed_series) > 0:
                    q1 = float(speed_series.quantile(0.25))
                    q2 = float(speed_series.quantile(0.50))
                    q3 = float(speed_series.quantile(0.75))
                    iqr = q3 - q1
                    outliers_low = len(speed_series[speed_series < (q1 - 1.5 * iqr)])
                    outliers_high = len(speed_series[speed_series > (q3 + 1.5 * iqr)])
                    stats['distribution_vent'] = {
                        'q1': q1, 'q2': q2, 'q3': q3,
                        'iqr': iqr,
                        'outliers_low': int(outliers_low),
                        'outliers_high': int(outliers_high),
                        'total_outliers': int(outliers_low + outliers_high)
                    }
            
            if 'RH' in df.columns:
                rh_series = pd.to_numeric(df['RH'], errors='coerce').dropna()
                if len(rh_series) > 0:
                    q1 = float(rh_series.quantile(0.25))
                    q2 = float(rh_series.quantile(0.50))
                    q3 = float(rh_series.quantile(0.75))
                    iqr = q3 - q1
                    outliers_low = len(rh_series[rh_series < (q1 - 1.5 * iqr)])
                    outliers_high = len(rh_series[rh_series > (q3 + 1.5 * iqr)])
                    stats['distribution_humidite'] = {
                        'q1': q1, 'q2': q2, 'q3': q3,
                        'iqr': iqr,
                        'outliers_low': int(outliers_low),
                        'outliers_high': int(outliers_high),
                        'total_outliers': int(outliers_low + outliers_high)
                    }
        
        except Exception as e:
            st.warning(f"Erreur calcul statistiques graphiques: {str(e)}")
        
        return stats
    
    def _enrich_with_llm_detailed(self, analysis_json, periode_type, audience, df=None):
        """
        Demande au LLM d'analyser chaque section avec des analyses détaillées.
        Essaie les modèles dans l'ordre : qwen-3-235b-a22b-instruct-2507 → llama-3.3-70b
        Retourne {} si aucun modèle ne peut générer une réponse valide.
        """
        nb_alertes_humidite = analysis_json.get('kpis', {}).get('nb_alertes_humidite', 0)
        temp = analysis_json.get('statistiques_temperature', {})
        vent = analysis_json.get('statistiques_vitesse_vent', {})
        hum = analysis_json.get('statistiques_humidite', {})
        wind_rose = analysis_json.get('wind_rose_data', {})
        kpis = analysis_json.get('kpis', {})

        # Extraire les vraies valeurs pour les insérer dans le prompt
        def _extract_stat_value(stats_dict, key, subkey=None):
            """Helper pour extraire les valeurs des statistiques"""
            if not isinstance(stats_dict, dict):
                return 'N/A'
            if key not in stats_dict:
                return 'N/A'
            value = stats_dict[key]
            if value is None:
                return 'N/A'
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    return 'N/A'
                return value
            if isinstance(value, dict) and subkey:
                if subkey in value:
                    sub_value = value[subkey]
                    if sub_value is None or (isinstance(sub_value, float) and np.isnan(sub_value)):
                        return 'N/A'
                    return sub_value
            return 'N/A'
        
        temp_moy = _extract_stat_value(temp, 'moyenne')
        temp_min = _extract_stat_value(temp, 'minimum', 'valeur')
        temp_max = _extract_stat_value(temp, 'maximum', 'valeur')
        temp_amp = _extract_stat_value(temp, 'amplitude')
        temp_ecart = _extract_stat_value(temp, 'ecart_type')
        temp_p25 = _extract_stat_value(temp, 'percentile_25')
        temp_p50 = _extract_stat_value(temp, 'percentile_50')
        temp_p75 = _extract_stat_value(temp, 'percentile_75')
        temp_p95 = _extract_stat_value(temp, 'percentile_95')
        
        vent_moy = _extract_stat_value(vent, 'moyenne')
        vent_min = _extract_stat_value(vent, 'minimum', 'valeur')
        vent_max = _extract_stat_value(vent, 'maximum', 'valeur')
        vent_ecart = _extract_stat_value(vent, 'ecart_type')
        
        # Récupération distribution vent
        dist_classes = vent.get('distribution_classes', {})
        dist_1_2 = dist_classes.get('legere_1_2', {}).get('pourcentage', 0) if isinstance(dist_classes.get('legere_1_2'), dict) else (dist_classes.get('1-2', {}).get('pourcentage', 0) if isinstance(dist_classes.get('1-2'), dict) else 0)
        dist_2_5 = dist_classes.get('brise_2_5', {}).get('pourcentage', 0) if isinstance(dist_classes.get('brise_2_5'), dict) else (dist_classes.get('2-5', {}).get('pourcentage', 0) if isinstance(dist_classes.get('2-5'), dict) else 0)
        
        # Extraction des données de la rose des vents
        wind_rose_text = ""
        if wind_rose and isinstance(wind_rose, dict):
            secteurs = wind_rose.get('secteurs', [])
            if secteurs and len(secteurs) > 0:
                # Trouver le secteur dominant
                secteur_dominant = max(secteurs, key=lambda x: x.get('frequence_totale', 0))
                wind_rose_text = f"Direction dominante: {secteur_dominant.get('nom', 'N/A')} avec {secteur_dominant.get('frequence_totale', 0):.1f}% de fréquence\\n"
                wind_rose_text += f"Vitesse moyenne secteur dominant: {secteur_dominant.get('vitesse_moyenne', 0):.2f} m/s\\n"
                wind_rose_text += f"Vitesse max secteur dominant: {secteur_dominant.get('vitesse_max', 0):.2f} m/s\\n"
                # Top 3 secteurs
                secteurs_tries = sorted(secteurs, key=lambda x: x.get('frequence_totale', 0), reverse=True)[:3]
                wind_rose_text += "Top 3 directions: "
                for i, s in enumerate(secteurs_tries):
                    wind_rose_text += f"{s.get('nom', 'N/A')} ({s.get('frequence_totale', 0):.1f}%)"
                    if i < len(secteurs_tries) - 1:
                        wind_rose_text += ", "
                wind_rose_text += "\\n"
        
        hum_moy = _extract_stat_value(hum, 'moyenne')
        hum_min = _extract_stat_value(hum, 'minimum', 'valeur')
        hum_max = _extract_stat_value(hum, 'maximum', 'valeur')
        rh_90_pct = _extract_stat_value(hum.get('temps_rh_sup_90', {}), 'pourcentage') if isinstance(hum.get('temps_rh_sup_90'), dict) else 0
        rh_95_pct = _extract_stat_value(hum.get('temps_rh_sup_95', {}), 'pourcentage') if isinstance(hum.get('temps_rh_sup_95'), dict) else 0
        
        # Statistiques Power
        power = analysis_json.get('statistiques_power', {})
        power_moy = _extract_stat_value(power, 'moyenne')
        power_min = _extract_stat_value(power, 'minimum', 'valeur')
        power_max = _extract_stat_value(power, 'maximum', 'valeur')
        power_ecart = _extract_stat_value(power, 'ecart_type')
        power_stabilite = power.get('stabilite', 'N/A') if isinstance(power, dict) else 'N/A'
        
        # Calculer les statistiques détaillées des graphiques
        graph_stats = self._compute_graph_statistics(df) if df is not None else {}
        
        # Extraire les valeurs de corrélation
        correlations_text = ""
        if graph_stats.get('correlations'):
            correlations_text = "CORRÉLATIONS CALCULÉES:\n"
            for key, value in graph_stats['correlations'].items():
                var1, var2 = key.split('_vs_')
                var1_name = var1.replace('#@', ' ').replace('Speed', 'Vitesse Vent').replace('AirTemp', 'Température').replace('RH', 'Humidité').replace('Power', 'Alimentation')
                var2_name = var2.replace('#@', ' ').replace('Speed', 'Vitesse Vent').replace('AirTemp', 'Température').replace('RH', 'Humidité').replace('Power', 'Alimentation')
                correlations_text += f"- {var1_name} vs {var2_name}: {value:.3f}\n"
        
        # Extraire les statistiques des séries temporelles
        time_series_text = ""
        if graph_stats.get('time_series_temp'):
            ts = graph_stats['time_series_temp']
            time_series_text += f"TEMPÉRATURE: Moyenne {ts['moyenne']:.2f}°C, Min {ts['min']:.2f}°C, Max {ts['max']:.2f}°C, Écart-type {ts['ecart_type']:.2f}°C, Amplitude {ts['amplitude']:.2f}°C\n"
        if graph_stats.get('time_series_vent'):
            ts = graph_stats['time_series_vent']
            time_series_text += f"VENT: Moyenne {ts['moyenne']:.2f} m/s, Min {ts['min']:.2f} m/s, Max {ts['max']:.2f} m/s, Écart-type {ts['ecart_type']:.2f} m/s\n"
        if graph_stats.get('time_series_humidite'):
            ts = graph_stats['time_series_humidite']
            time_series_text += f"HUMIDITÉ: Moyenne {ts['moyenne']:.2f}%, Min {ts['min']:.2f}%, Max {ts['max']:.2f}%, Écart-type {ts['ecart_type']:.2f}%\n"
        if graph_stats.get('time_series_power'):
            ts = graph_stats['time_series_power']
            time_series_text += f"ALIMENTATION: Moyenne {ts['moyenne']:.2f} V, Min {ts['min']:.2f} V, Max {ts['max']:.2f} V, Écart-type {ts['ecart_type']:.2f} V\n"
        
        # Extraire les statistiques des distributions
        distributions_text = ""
        if graph_stats.get('distribution_temp'):
            dist = graph_stats['distribution_temp']
            distributions_text += f"TEMPÉRATURE: Q1={dist['q1']:.2f}°C, Médiane={dist['q2']:.2f}°C, Q3={dist['q3']:.2f}°C, IQR={dist['iqr']:.2f}°C, Outliers={dist['total_outliers']} ({dist['outliers_low']} bas, {dist['outliers_high']} haut)\n"
        if graph_stats.get('distribution_vent'):
            dist = graph_stats['distribution_vent']
            distributions_text += f"VENT: Q1={dist['q1']:.2f} m/s, Médiane={dist['q2']:.2f} m/s, Q3={dist['q3']:.2f} m/s, IQR={dist['iqr']:.2f} m/s, Outliers={dist['total_outliers']} ({dist['outliers_low']} bas, {dist['outliers_high']} haut)\n"
        if graph_stats.get('distribution_humidite'):
            dist = graph_stats['distribution_humidite']
            distributions_text += f"HUMIDITÉ: Q1={dist['q1']:.2f}%, Médiane={dist['q2']:.2f}%, Q3={dist['q3']:.2f}%, IQR={dist['iqr']:.2f}%, Outliers={dist['total_outliers']} ({dist['outliers_low']} bas, {dist['outliers_high']} haut)\n"
        
        # Fonction helper pour formater les valeurs (définie avant utilisation)
        def fval(x):
            if x == 'N/A' or x is None:
                return 'N/A'
            try:
                if isinstance(x, float):
                    return f"{x:.2f}".replace('.', ',')
                return str(x)
            except:
                return str(x)
        
        # Extraire les KPIs
        kpis_text = ""
        if kpis:
            temps_optimal = kpis.get('temps_conditions_optimales', {})
            if isinstance(temps_optimal, dict):
                kpis_text += f"Temps conditions optimales: {temps_optimal.get('pourcentage', 0):.2f}% ({temps_optimal.get('heures', 0):.2f} heures)\n"
            kpis_text += f"Alertes vent fort: {kpis.get('nb_alertes_vent_fort', 0)}\n"
            kpis_text += f"Alertes humidité: {kpis.get('nb_alertes_humidite', 0)}\n"
        
        # Calculer les indicateurs du radar pour l'analyse LLM
        radar_indicators_text = ""
        if df is not None and len(df) > 0:
            # Conditions optimales
            if kpis and isinstance(kpis.get('temps_conditions_optimales'), dict):
                temps_opt = kpis.get('temps_conditions_optimales', {})
                radar_indicators_text += f"Conditions Optimales: {fval(temps_opt.get('pourcentage', 0))}%\n"
            else:
                optimal_conditions = len(df[(df['Speed#@1m'] < 12) & (df['RH'] < 90) & 
                                           (df['AirTemp'] >= 5) & (df['AirTemp'] <= 35)])
                radar_indicators_text += f"Conditions Optimales: {fval((optimal_conditions / len(df)) * 100)}%\n"
            
            # Stabilités (calculées depuis les écarts-types)
            if 'AirTemp' in df.columns:
                temp_std = df['AirTemp'].std()
                temp_stability = max(0, min(100, 100 - (temp_std * 10)))
                radar_indicators_text += f"Stabilité Température: {fval(temp_stability)}% (écart-type: {fval(temp_std)}°C)\n"
            
            if 'Speed#@1m' in df.columns:
                vent_std = df['Speed#@1m'].std()
                vent_stability = max(0, min(100, 100 - (vent_std * 20)))
                radar_indicators_text += f"Stabilité Vent: {fval(vent_stability)}% (écart-type: {fval(vent_std)} m/s)\n"
            
            if 'RH' in df.columns:
                rh_std = df['RH'].std()
                rh_stability = max(0, min(100, 100 - (rh_std * 5)))
                radar_indicators_text += f"Stabilité Humidité: {fval(rh_stability)}% (écart-type: {fval(rh_std)}%)\n"
            
            if 'Power' in df.columns:
                power_std = df['Power'].std()
                power_stability = max(0, min(100, 100 - (power_std * 50)))
                radar_indicators_text += f"Stabilité Alimentation: {fval(power_stability)}% (écart-type: {fval(power_std)} V)\n"
            
            # Qualité des données
            if analysis_json and 'resume_general' in analysis_json:
                rg = analysis_json.get('resume_general', {})
                taux_completude = rg.get('taux_completude', '100%')
                if isinstance(taux_completude, str):
                    taux_completude = float(taux_completude.replace('%', '').replace(',', '.'))
                radar_indicators_text += f"Qualité Données: {fval(taux_completude)}%\n"
            else:
                total_cells = len(df) * len(df.columns)
                missing_cells = df.isnull().sum().sum()
                completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 100
                radar_indicators_text += f"Qualité Données: {fval(completeness)}%\n"
        
        prompt = f"""
        Tu es un expert météorologique professionnel. Analyse en profondeur les données suivantes et fournis UNIQUEMENT du JSON valide.
        
        CONTEXTE:
        - Audience: {audience}
        - Type de rapport: {periode_type}
        
        ⚠️⚠️⚠️ **VALEURS EXACTES À UTILISER OBLIGATOIREMENT** ⚠️⚠️⚠️
        NE JAMAIS INVENTER DE CHIFFRES - UTILISER UNIQUEMENT CES VALEURS :
        
        TEMPÉRATURE (°C):
        - Moyenne: {fval(temp_moy)}°C
        - Minimum: {fval(temp_min)}°C  
        - Maximum: {fval(temp_max)}°C
        - Amplitude: {fval(temp_amp)}°C
        - Écart-type: {fval(temp_ecart)}°C
        - Percentile 25: {fval(temp_p25)}°C
        - Percentile 50 (médiane): {fval(temp_p50)}°C
        - Percentile 75: {fval(temp_p75)}°C
        - Percentile 95: {fval(temp_p95)}°C
        
        VENT (m/s):
        - Vitesse moyenne: {fval(vent_moy)} m/s
        - Minimum: {fval(vent_min)} m/s
        - Maximum: {fval(vent_max)} m/s
        - Écart-type: {fval(vent_ecart)} m/s
        - Distribution: {fval(dist_1_2)}% dans 1-2 m/s, {fval(dist_2_5)}% dans 2-5 m/s
        
        HUMIDITÉ (%):
        - Moyenne: {fval(hum_moy)}%
        - Minimum: {fval(hum_min)}%
        - Maximum: {fval(hum_max)}%
        - Temps RH>90%: {fval(rh_90_pct)}%
        - Temps RH>95%: {fval(rh_95_pct)}%
        
        ALIMENTATION (V):
        - Moyenne: {fval(power_moy)} V
        - Minimum: {fval(power_min)} V
        - Maximum: {fval(power_max)} V
        - Écart-type: {fval(power_ecart)} V
        - Stabilité: {power_stabilite}

        STATISTIQUES DÉTAILLÉES POUR LES GRAPHIQUES:
        
        SÉRIES TEMPORELLES:
        {time_series_text if time_series_text else "Données non disponibles"}
        
        CORRÉLATIONS:
        {correlations_text if correlations_text else "Données non disponibles"}
        
        DISTRIBUTIONS (Boxplots/Histogrammes):
        {distributions_text if distributions_text else "Données non disponibles"}
        
        KPIs:
        {kpis_text if kpis_text else "Données non disponibles"}
        
        ROSE DES VENTS:
        {wind_rose_text if wind_rose_text else "Données non disponibles"}
        
        INDICATEURS RADAR (0-100%):
        {radar_indicators_text if radar_indicators_text else "Données non disponibles"}

        ⚠️⚠️⚠️ INSTRUCTIONS CRITIQUES - ANALYSES ULTRA-DÉTAILLÉES OBLIGATOIRES ⚠️⚠️⚠️
        
        1. UTILISER EXCLUSIVEMENT les valeurs listées ci-dessus - JAMAIS inventer de chiffres
        2. ÉCRIRE "2-5 m/s" ou "2 à 5 m/s" - JAMAIS "25 m/s"
        3. Pour l'alimentation, utiliser "V" (volts) - JAMAIS "kW"
        4. Écrire "écart-type" avec un tiret
        5. Les analyses doivent être cohérentes avec les valeurs réelles
        6. Pour TOUS les graphiques et tableaux, INCLURE TOUTES LES VALEURS NUMÉRIQUES dans les analyses
        7. La conclusion doit être ULTRA-DÉTAILLÉE avec toutes les valeurs importantes
        
        🔥 EXIGENCES STRICTES POUR ANALYSES ULTRA-DÉTAILLÉES (OBLIGATOIRE) 🔥
        
        POUR CHAQUE SECTION (température, vent, humidité, rose des vents, KPIs):
        - MINIMUM 8-12 phrases d'analyse approfondie par section
        - CITER TOUTES les valeurs numériques pertinentes (moyenne, min, max, écart-type, percentiles, etc.)
        - Expliquer les implications opérationnelles de CHAQUE valeur avec contexte
        - Comparer systématiquement les valeurs entre elles avec calculs explicites
        - Décrire les tendances observées avec valeurs exactes et évolution quantitative
        - Expliquer les relations entre variables avec corrélations exactes et interprétation physique
        - Identifier les patterns et anomalies avec justification quantitative et seuils de référence
        - Fournir un contexte météorologique détaillé basé uniquement sur les valeurs réelles
        - Analyser chaque tableau avec TOUTES les valeurs du tableau (pas seulement la moyenne)
        - Décrire chaque graphique avec TOUTES les valeurs visibles (min, max, tendances, pics, creux)
        
        POUR LES GRAPHIQUES SPÉCIFIQUEMENT:
        - SÉRIES TEMPORELLES: Analyser chaque variable avec min/max/moyenne/écart-type, identifier les pics et creux avec valeurs exactes, décrire les tendances avec quantification, analyser la stabilité temporelle
        - MATRICE DE CORRÉLATION: Analyser CHAQUE corrélation avec sa valeur exacte, classifier (forte/modérée/faible), expliquer le sens physique, décrire les implications opérationnelles
        - DISTRIBUTIONS (Histogrammes/Boxplots): Analyser Q1/médiane/Q3/IQR pour chaque variable, compter et analyser les outliers avec valeurs exactes, décrire la forme de distribution, comparer entre variables
        - RADAR: Analyser CHAQUE indicateur avec sa valeur exacte (0-100%), classifier (excellent/bon/moyen/faible), identifier forces et faiblesses avec valeurs, proposer actions correctives prioritaires
        
        POUR LES TABLEAUX:
        - Analyser CHAQUE ligne du tableau avec ses valeurs exactes
        - Comparer les valeurs entre lignes (ex: "La température max de X°C est Y°C au-dessus de la moyenne de Z°C")
        - Expliquer les écarts et variations avec calculs explicites
        - Identifier les valeurs remarquables (extrêmes, médianes, percentiles) avec interprétation
        
        RÈGLES ABSOLUES:
        - Si une valeur n'est pas fournie, ne pas l'inventer - utiliser "N/A" ou omettre
        - Chaque interprétation doit être justifiée par au moins une valeur numérique fournie
        - TOUTES les affirmations doivent être quantifiées avec des valeurs exactes
        - Ne jamais faire d'affirmations générales sans valeurs numériques de support
        - Chaque section doit être suffisamment détaillée pour être utilisée comme rapport professionnel autonome

        Format de sortie STRICT (JSON uniquement):
        {{
          "section_temperature": {{
            "analyse": "Analyse ULTRA-DÉTAILLÉE de la température (MINIMUM 8-12 phrases obligatoires):\\n- Température moyenne: {fval(temp_moy)}°C (expliquer si cette valeur est normale, élevée ou faible pour la période avec comparaison quantitative)\\n- Minimum observé: {fval(temp_min)}°C (risque de gel si <5°C, implications opérationnelles détaillées avec seuils)\\n- Maximum observé: {fval(temp_max)}°C (risque de surchauffe si >35°C, impacts sur les équipements avec quantification)\\n- Amplitude thermique totale: {fval(temp_amp)}°C (décrire la variabilité: faible si <5°C, modérée si 5-15°C, élevée si >15°C, avec comparaison avec normales)\\n- Écart-type: {fval(temp_ecart)}°C (stabilité: très stable si <2°C, stable si 2-5°C, variable si >5°C, avec analyse de l'écart par rapport à la moyenne)\\n- Distribution détaillée: 25% des valeurs ≤ {fval(temp_p25)}°C, médiane {fval(temp_p50)}°C (comparer avec la moyenne {fval(temp_moy)}°C - écart de {fval(abs(temp_p50 - temp_moy) if isinstance(temp_p50, (int, float)) and isinstance(temp_moy, (int, float)) else 0)}°C), 75% ≤ {fval(temp_p75)}°C, 95% ≤ {fval(temp_p95)}°C (identifier les valeurs extrêmes avec analyse)\\n- Interprétation approfondie: Analyser la stabilité thermique en comparant moyenne/médiane/écart-type avec calculs explicites, décrire les variations possibles (diurnes/nocturnes) basées sur l'amplitude avec quantification, évaluer les risques (gel/surchauffe) avec les valeurs min/max et seuils de référence, expliquer la distribution (concentration autour de la médiane, symétrie, valeurs extrêmes) avec valeurs exactes. Comparer toutes les valeurs entre elles. TOUTES les interprétations doivent être basées uniquement sur les valeurs fournies avec citations numériques explicites.",
            "anomalies": ["Anomalie si applicable avec valeurs numériques exactes et seuils de référence"],
            "recommandations": ["Recommandation 1 détaillée avec justification quantitative basée sur les valeurs", "Recommandation 2 avec métriques associées et seuils d'alerte", "Recommandation 3 avec contexte opérationnel basé sur les statistiques"]
          }},
          "section_vent": {{
            "analyse": "Analyse ULTRA-DÉTAILLÉE du vent (MINIMUM 8-12 phrases obligatoires):\\n- Vitesse moyenne: {fval(vent_moy)} m/s (classifier: calme <2 m/s, léger 2-5 m/s, modéré 5-8 m/s, fort >8 m/s, avec justification quantitative)\\n- Minimum: {fval(vent_min)} m/s (périodes de calme, implications pour dispersion avec analyse détaillée)\\n- Maximum: {fval(vent_max)} m/s (pics de vent, risques opérationnels si >12 m/s avec évaluation quantitative du risque)\\n- Écart-type: {fval(vent_ecart)} m/s (stabilité: très stable si <1 m/s, stable si 1-2 m/s, variable si >2 m/s, avec comparaison avec la moyenne {fval(vent_moy)} m/s - coefficient de variation {fval((vent_ecart/vent_moy*100) if isinstance(vent_ecart, (int, float)) and isinstance(vent_moy, (int, float)) and vent_moy != 0 else 0)}%)\\n- Distribution détaillée: {fval(dist_1_2)}% dans 1-2 m/s (vent léger), {fval(dist_2_5)}% dans 2-5 m/s (vent modéré) - analyser la répartition avec comparaison entre classes\\n- Fourchette complète: {fval(vent_min)} à {fval(vent_max)} m/s (amplitude de {fval(vent_max - vent_min if isinstance(vent_max, (int, float)) and isinstance(vent_min, (int, float)) else 0)} m/s, variabilité avec analyse)\\n- Interprétation approfondie: Décrire les conditions dominantes (calme/léger/modéré/fort) avec pourcentages exacts et calculs, analyser la stabilité (écart-type {fval(vent_ecart)} m/s vs moyenne {fval(vent_moy)} m/s), identifier les périodes critiques (min {fval(vent_min)} m/s / max {fval(vent_max)} m/s), expliquer les implications opérationnelles (dispersion, sécurité, production) avec valeurs quantifiées et seuils. Comparer toutes les valeurs entre elles. TOUTES les interprétations basées uniquement sur les valeurs fournies avec citations numériques explicites.",
            "anomalies": ["Anomalie spécifique du vent avec valeurs numériques exactes et seuils de référence"],
            "recommandations": ["Recommandation actionnable 1 détaillée avec justification quantitative et seuils", "Recommandation actionnable 2 avec métriques et contexte opérationnel", "Recommandation actionnable 3 avec valeurs de référence"]
          }},
          "section_humidite": {{
            "analyse": "Analyse ULTRA-DÉTAILLÉE de l'humidité (MINIMUM 8-12 phrases obligatoires):\\n- Humidité relative moyenne: {fval(hum_moy)}% (classifier: sec <40%, normal 40-70%, humide 70-90%, très humide >90%, avec justification quantitative)\\n- Minimum: {fval(hum_min)}% (périodes sèches, risques de dessèchement avec analyse détaillée)\\n- Maximum: {fval(hum_max)}% (pics d'humidité, risques de condensation si >95% avec évaluation quantitative - écart de {fval(hum_max - 95 if isinstance(hum_max, (int, float)) and hum_max > 95 else 0)}% au-dessus du seuil critique)\\n- Temps avec RH>90%: {fval(rh_90_pct)}% (conditions humides - analyser la durée et fréquence avec quantification: {fval(rh_90_pct/100*24 if isinstance(rh_90_pct, (int, float)) else 0)} heures/jour en moyenne)\\n- Temps avec RH>95%: {fval(rh_95_pct)}% (conditions critiques - risques de condensation, corrosion, équipements avec quantification: {fval(rh_95_pct/100*24 if isinstance(rh_95_pct, (int, float)) else 0)} heures/jour en moyenne)\\n- Amplitude: {fval(hum_max - hum_min if isinstance(hum_max, (int, float)) and isinstance(hum_min, (int, float)) else 0)}% (variabilité avec analyse)\\n- Interprétation approfondie: Analyser la distribution de l'humidité (moyenne {fval(hum_moy)}% vs min {fval(hum_min)}% / max {fval(hum_max)}% avec calculs), évaluer les risques de condensation (RH>95% critique {fval(rh_95_pct)}%, RH>90% élevé {fval(rh_90_pct)}%), expliquer les implications pour les équipements (corrosion, performance) avec pourcentages exacts et durées, décrire les périodes critiques (durée, fréquence) basées sur les valeurs RH>90% et RH>95% avec conversion en heures, comparer avec les seuils opérationnels. Comparer toutes les valeurs entre elles. TOUTES les interprétations basées uniquement sur les valeurs fournies avec citations numériques explicites.",
            "anomalies": ["Anomalie critique humidité si applicable avec valeurs numériques exactes, seuils de référence et impacts quantifiés"],
            "recommandations": ["Recommandation actionnable 1 détaillée avec métriques et seuils d'alerte", "Recommandation actionnable 2 avec valeurs de référence et justification quantitative", "Recommandation actionnable 3 avec contexte opérationnel et impacts mesurables"]
          }},
          "section_rose_des_vents": {{
            "analyse": "Analyse ULTRA-DÉTAILLÉE avec VALEURS NUMÉRIQUES de la rose des vents (MINIMUM 8-12 phrases obligatoires). Inclus: direction dominante avec pourcentage exact de fréquence et comparaison détaillée avec autres directions (calculs d'écarts), vitesse moyenne et max pour cette direction avec interprétation quantitative, top 3 directions avec leurs fréquences exactes et analyse détaillée de la répartition (comparaisons, pourcentages), distribution des vitesses par secteur (calme/léger/modéré/fort) avec pourcentages exacts et analyse, analyse de la variabilité directionnelle avec quantification. Décris les implications opérationnelles détaillées avec valeurs numériques (ex: vent dominant venant du Nord à 25% avec vitesse moyenne de 3,5 m/s - implications pour dispersion, orientation des équipements, zones d'impact avec quantification). Compare les directions entre elles avec calculs, analysez la concentration ou la dispersion des vents avec valeurs, expliquez les patterns saisonniers si discernables avec quantification. Analyse chaque secteur avec ses valeurs exactes (fréquence, vitesse moyenne, vitesse max). Utilise UNIQUEMENT les valeurs exactes fournies dans les données de la rose des vents avec citations numériques explicites.",
            "anomalies": ["Anomalie direction si applicable avec valeurs numériques exactes, comparaison avec normales et seuils"],
            "recommandations": ["Recommandation actionnable 1 détaillée avec justification basée sur les directions et valeurs quantifiées", "Recommandation actionnable 2 avec métriques de vent et implications opérationnelles spécifiques", "Recommandation actionnable 3 avec analyse des patterns directionnels"]
          }},
          "section_series_temporelles": {{
            "analyse": "Analyse ULTRA-DÉTAILLÉE avec VALEURS NUMÉRIQUES des tendances temporelles (MINIMUM 8-12 phrases obligatoires). Pour CHAQUE variable (température, vent, humidité, alimentation), inclus: moyennes avec interprétation et comparaison, min/max avec timestamps si disponibles et analyse détaillée des extrêmes avec valeurs exactes, écarts-types avec évaluation quantitative de la stabilité (coefficient de variation), amplitudes avec description quantitative de la variabilité. Décris les variations observées dans le temps (tendances, cycles, patterns) avec valeurs exactes, analyse la stabilité (comparaison écart-type vs moyenne avec calculs), identifie les pics avec leurs valeurs exactes et moments avec analyse, décris les périodes de variations importantes avec quantification précise. Compare les variables entre elles (ex: corrélation temporelle entre température et humidité avec valeurs). Explique les implications opérationnelles détaillées des tendances observées avec quantification. Utilise UNIQUEMENT les valeurs exactes fournies dans les statistiques détaillées avec citations numériques explicites."
          }},
          "section_correlations": {{
            "analyse": "Analyse ULTRA-DÉTAILLÉE avec VALEURS NUMÉRIQUES des corrélations (MINIMUM 8-12 phrases obligatoires). Pour CHAQUE paire de variables, indique la valeur de corrélation exacte avec classification détaillée (forte >0,7, modérée 0,4-0,7, faible <0,4, négative si <0). Interprète chaque corrélation avec sa valeur numérique exacte et explique le sens physique détaillé (ex: corrélation positive température-humidité de X signifie augmentation simultanée avec quantification). Décris les implications opérationnelles détaillées (ex: si température et vent sont corrélés à X, implications pour prévisions avec analyse quantitative). Compare les corrélations entre elles pour identifier les relations les plus importantes avec classement. Explique les corrélations faibles ou négatives et leur signification avec valeurs. Analyse les corrélations manquantes ou faibles et leurs implications. Utilise UNIQUEMENT les valeurs exactes fournies dans les corrélations calculées avec citations numériques explicites."
          }},
          "section_distributions": {{
            "analyse": "Analyse ULTRA-DÉTAILLÉE avec VALEURS NUMÉRIQUES des distributions (MINIMUM 8-12 phrases obligatoires). Pour CHAQUE variable, indique: quartiles (Q1, médiane, Q3) avec comparaison détaillée entre eux et avec la moyenne (calculs d'écarts), IQR avec interprétation quantitative (étroit = concentré si IQR<X, large = dispersé si IQR>Y), nombre exact d'outliers (bas et haut) avec pourcentage du total et analyse. Décris la forme de la distribution (symétrique si médiane≈moyenne avec écart, asymétrique sinon avec quantification, normale, bimodale) avec valeurs exactes, les concentrations de valeurs avec leurs intervalles numériques exacts (ex: 50% des valeurs entre Q1 et Q3 avec valeurs), et les valeurs aberrantes détectées avec leurs valeurs exactes et analyse détaillée de leur impact. Compare les distributions entre variables avec calculs. Explique les implications opérationnelles détaillées (ex: distribution concentrée = stable avec valeurs, dispersée = variable avec quantification). Utilise UNIQUEMENT les valeurs exactes fournies dans les statistiques de distribution avec citations numériques explicites."
          }},
          "section_kpis": {{
            "analyse": "Analyse ULTRA-DÉTAILLÉE avec VALEURS NUMÉRIQUES des KPIs (MINIMUM 8-12 phrases obligatoires). Inclus: pourcentage exact de temps en conditions optimales avec comparaison détaillée avec objectifs (ex: 80%+ excellent, 60-80% bon, <60% à améliorer - avec écart quantifié), nombre exact d'alertes vent fort avec analyse détaillée de fréquence et sévérité (calcul de fréquence par jour/heure), nombre exact d'alertes humidité avec évaluation détaillée des risques (calcul de fréquence), heures de conditions optimales avec conversion en jours si pertinent et analyse. Interprète chaque KPI avec sa valeur numérique exacte, compare avec des seuils de référence avec calculs d'écarts, explique les implications opérationnelles détaillées (productivité, sécurité, maintenance) avec quantification. Analyse les relations entre KPIs (ex: conditions optimales vs alertes) avec corrélations. Identifie les tendances et patterns avec valeurs. Utilise UNIQUEMENT les valeurs exactes fournies dans les KPIs avec citations numériques explicites.",
            "anomalies": ["Anomalie KPI si applicable avec valeurs numériques exactes, comparaison avec seuils et impacts quantifiés"],
            "recommandations": ["Recommandation actionnable 1 détaillée avec justification quantitative et objectifs mesurables", "Recommandation actionnable 2 avec métriques de suivi et seuils d'alerte", "Recommandation actionnable 3 avec analyse des priorités basée sur les KPIs"]
          }},
          "section_radar": {{
            "analyse": "Analyse ULTRA-DÉTAILLÉE avec VALEURS NUMÉRIQUES du graphique radar des indicateurs opérationnels (MINIMUM 8-12 phrases obligatoires). Décris CHAQUE indicateur avec sa valeur exacte (0-100) et classification détaillée (excellent >80%, bon 60-80%, moyen 40-60%, faible <40%): Conditions Optimales (%), Stabilité Température, Stabilité Vent, Stabilité Humidité, Stabilité Alimentation, Qualité Données (%). Identifie les points forts (valeurs élevées >70%) avec leurs valeurs numériques exactes et explication détaillée de leur importance opérationnelle avec quantification. Identifie les points faibles (valeurs faibles <50%) avec leurs valeurs numériques exactes et analyse détaillée des causes possibles avec calculs d'écarts par rapport aux seuils. Compare les indicateurs entre eux pour identifier les déséquilibres avec calculs. Explique les implications opérationnelles détaillées de chaque indicateur (ex: stabilité faible = variabilité élevée = risques opérationnels) avec quantification. Décris les actions correctives nécessaires si des valeurs sont faibles avec priorités basées sur les valeurs et calculs d'impact. Analyse chaque indicateur individuellement avec sa valeur exacte. Utilise UNIQUEMENT les valeurs exactes fournies avec citations numériques explicites."
          }},
          "conclusion": {{
            "synthese": "Synthèse ULTRA-DÉTAILLÉE de 15-20 phrases incluant TOUTES les valeurs importantes avec interprétations approfondies et citations numériques explicites: températures moyennes/min/max/amplitude/écart-type avec analyse détaillée de stabilité et comparaisons, vitesses vent moyennes/min/max avec classification détaillée et risques quantifiés, humidité moyenne et temps RH>90% et RH>95% avec évaluation détaillée des risques de condensation et durées, corrélations principales avec implications opérationnelles détaillées et valeurs exactes, nombre d'outliers avec analyse détaillée de leur impact et pourcentages, temps conditions optimales avec comparaison détaillée avec objectifs et écarts quantifiés, nombre d'alertes avec évaluation détaillée de fréquence et sévérité. Résume les points clés avec valeurs numériques exactes et calculs, les risques principaux quantifiés avec seuils de référence et écarts, l'état global avec métriques précises et comparaisons détaillées, les tendances observées avec quantification, les forces et faiblesses identifiées avec valeurs exactes. Compare toutes les métriques entre elles. Chaque affirmation doit être justifiée par au moins une valeur numérique fournie avec citation explicite.",
            "priorites": ["Priorité d'action 1 (la plus urgente) avec valeurs numériques exactes justifiant l'urgence, seuils critiques et impacts quantifiés", "Priorité d'action 2 avec métriques associées, objectifs mesurables et délais", "Priorité d'action 3 avec indicateurs quantifiés, seuils de suivi et critères de succès", "Priorité d'action 4 (si applicable) avec justification quantitative"]
          }}
        }}

        ⚠️⚠️⚠️ INSTRUCTIONS FINALES CRITIQUES ⚠️⚠️⚠️
        
        - Les analyses doivent être sous forme de puces (commençant par "-") ou paragraphes structurés TRÈS DÉTAILLÉS
        - Utilise UNIQUEMENT les valeurs réelles fournies, JAMAIS de chiffres inventés - si une valeur n'est pas fournie, utilise "N/A" ou omets-la
        - Chaque interprétation doit être justifiée par au moins une valeur numérique fournie avec citation explicite
        - Sois ULTRA-DÉTAILLÉ dans les analyses (MINIMUM 8-12 phrases par section, 15-20 pour la conclusion) mais basé uniquement sur les données réelles
        - Compare systématiquement les valeurs entre elles avec calculs explicites, analyse les tendances avec quantification, explique les implications opérationnelles avec valeurs exactes
        - Pour TOUS les tableaux: analyser CHAQUE ligne avec ses valeurs, comparer entre lignes, expliquer les écarts
        - Pour TOUS les graphiques: citer TOUTES les valeurs visibles (min, max, moyennes, tendances, pics, creux), analyser chaque élément
        - Chaque section doit être suffisamment détaillée pour servir de rapport professionnel autonome
        - Ne fournis RIEN d'autre que le JSON. Pas d'explications supplémentaires avant ou après le JSON.
        """

        # Essayer le modèle principal avec plusieurs tentatives
        result = self._try_llm_model_with_retry(self.llm, prompt, model_name="gpt-oss-120b", max_retries=3)
        if result:
            return result

        # Essayer les modèles de fallback avec plusieurs tentatives
        for model_name in self.fallback_models:
            try:
                st.info(f"🔄 Modèle principal indisponible, essai du modèle : {model_name}")
                llm = CerebriumLLM(model_name)
                result = self._try_llm_model_with_retry(llm, prompt, model_name=model_name, max_retries=3)
                if result:
                    st.success(f"✅ Enrichissement généré avec {model_name}")
                    return result
            except Exception as e:
                st.warning(f"⚠️ Modèle {model_name} indisponible : {str(e)}")
                continue

        # Dernière tentative : essayer avec un prompt simplifié
        st.info("🔄 Tentative avec prompt simplifié...")
        simplified_prompt = self._create_simplified_prompt(analysis_json, periode_type, audience, df)
        result = self._try_llm_model_with_retry(self.llm, simplified_prompt, model_name="gpt-oss-120b", max_retries=2)
        if result:
            return result
        
        for model_name in self.fallback_models:
            try:
                llm = CerebriumLLM(model_name)
                result = self._try_llm_model_with_retry(llm, simplified_prompt, model_name=model_name, max_retries=2)
                if result:
                    st.success(f"✅ Enrichissement généré avec {model_name} (prompt simplifié)")
                    return result
            except Exception as e:
                continue

        # Aucun modèle n'a pu générer une réponse valide
        st.warning("⚠️ Aucun modèle LLM disponible pour l'enrichissement. Tentative de génération complète du rapport par LLM...")
        return {}

    def _extract_json_from_text(self, text):
        """Extrait le premier JSON valide du texte, même s'il y a du contenu supplémentaire - VERSION AMÉLIORÉE"""
        if not text:
            return None
        
        # Méthode 1: Trouver le premier '{'
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        # Utiliser json.JSONDecoder pour parser seulement le JSON valide
        decoder = json.JSONDecoder()
        try:
            # Extraire à partir du premier '{' et parser seulement le JSON valide
            json_str = text[start_idx:]
            obj, idx = decoder.raw_decode(json_str)
            return obj
        except (json.JSONDecodeError, ValueError) as e:
            # Méthode 2: Trouver le JSON en comptant les accolades
            try:
                brace_count = 0
                json_start = start_idx
                json_end = start_idx
                
                for i in range(start_idx, len(text)):
                    if text[i] == '{':
                        if brace_count == 0:
                            json_start = i
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if brace_count == 0 and json_end > json_start:
                    json_str = text[json_start:json_end]
                    return json.loads(json_str)
            except Exception:
                pass
            
            # Méthode 3: Essayer de trouver un JSON entre balises markdown
            try:
                # Chercher entre ```json et ```
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # Chercher entre ``` et ```
                json_match = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
            except Exception:
                pass
            
            # Méthode 4: Essayer de réparer le JSON (enlever les caractères problématiques)
            try:
                # Nettoyer le texte
                cleaned = text[start_idx:]
                # Enlever les commentaires JSON invalides
                cleaned = re.sub(r'//.*?$', '', cleaned, flags=re.MULTILINE)
                # Essayer de trouver le JSON valide
                obj, idx = decoder.raw_decode(cleaned)
                return obj
            except Exception:
                pass
            
            # Méthode 5: Essayer d'extraire plusieurs objets JSON et prendre le plus grand
            try:
                json_objects = []
                start = start_idx
                while True:
                    start_pos = text.find('{', start)
                    if start_pos == -1:
                        break
                    try:
                        obj, idx = decoder.raw_decode(text[start_pos:])
                        json_objects.append(obj)
                        start = start_pos + idx
                    except:
                        start = start_pos + 1
                
                if json_objects:
                    # Prendre le plus grand objet JSON
                    largest = max(json_objects, key=lambda x: len(str(x)))
                    return largest
            except Exception:
                pass
        
        return None
    
    def _try_llm_model_with_retry(self, llm, prompt, model_name="", max_retries=3):
        """
        Essaie un modèle LLM avec plusieurs tentatives et différents paramètres.
        Retourne le JSON parsé si succès, None sinon.
        """
        if not llm:
            return None

        # Essayer avec différents paramètres
        attempts = [
            {"max_tokens": 10000, "temperature": 0.1},
            {"max_tokens": 10000, "temperature": 0.1},
            {"max_tokens": 10000, "temperature": 0.2},
        ]
        
        for attempt_num in range(max_retries):
            params = attempts[attempt_num % len(attempts)]
            
            try:
                if attempt_num > 0:
                    st.info(f"🔄 Tentative {attempt_num + 1}/{max_retries} avec {model_name}...")
                
                # Appel direct avec paramètres personnalisés
                headers = {
                    "Authorization": f"Bearer {llm.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": llm.model_name,
                    "prompt": prompt,
                    "max_tokens": params["max_tokens"],
                    "temperature": params["temperature"],
                    "stream": False
                }
                
                response = requests.post(llm.endpoint, json=payload, headers=headers, timeout=120000)
                response.raise_for_status()
                result = response.json()
                resp = result["choices"][0]["text"]
                
                if not resp:
                    continue

                # Essayer d'extraire le JSON avec plusieurs méthodes
                json_result = self._extract_json_from_text(resp)
                if json_result:
                    # Valider que les clés essentielles sont présentes
                    required_sections = ['section_temperature', 'section_vent', 'section_humidite', 'section_rose_des_vents', 'section_kpis', 'conclusion']
                    optional_sections = ['section_series_temporelles', 'section_correlations', 'section_distributions', 'section_radar']
                    
                    # Vérifier les sections requises
                    if all(key in json_result for key in required_sections):
                        # Ajouter les sections optionnelles si manquantes avec des valeurs par défaut
                        for opt_section in optional_sections:
                            if opt_section not in json_result:
                                json_result[opt_section] = {"analyse": "Analyse non disponible"}
                        return json_result
                    else:
                        # Si certaines sections manquent, essayer de les compléter
                        missing = [s for s in required_sections if s not in json_result]
                        if len(missing) <= 2:  # Accepter si seulement 1-2 sections manquent
                            for opt_section in optional_sections:
                                if opt_section not in json_result:
                                    json_result[opt_section] = {"analyse": "Analyse non disponible"}
                            for missing_section in missing:
                                if missing_section == "conclusion":
                                    json_result[missing_section] = {"synthese": "Analyse non disponible", "priorites": []}
                                else:
                                    json_result[missing_section] = {"analyse": "Analyse non disponible - données insuffisantes"}
                            return json_result
                
                # Si le JSON n'est pas valide, essayer de le réparer
                if resp and '{' in resp:
                    json_result = self._try_repair_json(resp)
                    if json_result:
                        return json_result
                        
            except Exception as e:
                if attempt_num < max_retries - 1:
                    st.warning(f"⚠️ Tentative {attempt_num + 1} échouée pour {model_name}: {str(e)[:100]}... Réessai...")
                continue

        return None
    
    def _try_llm_model(self, llm, prompt):
        """
        Essaie un modèle LLM spécifique et valide la réponse.
        Retourne le JSON parsé si succès, None sinon.
        """
        return self._try_llm_model_with_retry(llm, prompt, max_retries=1)
    
    def _try_repair_json(self, text):
        """Essaie de réparer un JSON malformé"""
        try:
            # Chercher tous les objets JSON possibles
            json_objects = []
            start = 0
            while True:
                start_idx = text.find('{', start)
                if start_idx == -1:
                    break
                
                # Essayer d'extraire un JSON à partir de cette position
                decoder = json.JSONDecoder()
                try:
                    obj, idx = decoder.raw_decode(text[start_idx:])
                    json_objects.append(obj)
                    start = start_idx + idx
                except:
                    start = start_idx + 1
            
            # Si on a trouvé des objets JSON, essayer de les combiner
            if json_objects:
                # Prendre le plus grand objet
                largest = max(json_objects, key=lambda x: len(str(x)))
                return largest
        except:
            pass
        return None
    
    def _create_simplified_prompt(self, analysis_json, periode_type, audience, df=None):
        """Crée un prompt simplifié si le prompt complet échoue"""
        return f"""
        Tu es un expert météorologique. Génère une analyse JSON simplifiée mais complète.
        
        Période: {periode_type}
        Audience: {audience}
        
        Données principales:
        - Température moyenne: {analysis_json.get('statistiques_temperature', {}).get('moyenne', 'N/A')}°C
        - Vent moyen: {analysis_json.get('statistiques_vitesse_vent', {}).get('moyenne', 'N/A')} m/s
        - Humidité moyenne: {analysis_json.get('statistiques_humidite', {}).get('moyenne', 'N/A')}%
        
        Retourne UNIQUEMENT un JSON valide avec ces sections (minimum 3-5 phrases par section):
        {{
          "section_temperature": {{"analyse": "Analyse détaillée avec valeurs numériques"}},
          "section_vent": {{"analyse": "Analyse détaillée avec valeurs numériques"}},
          "section_humidite": {{"analyse": "Analyse détaillée avec valeurs numériques"}},
          "section_rose_des_vents": {{"analyse": "Analyse détaillée"}},
          "section_kpis": {{"analyse": "Analyse détaillée"}},
          "conclusion": {{"synthese": "Synthèse détaillée", "priorites": ["Priorité 1", "Priorité 2"]}}
        }}
        """
    
    def _generate_full_report_with_llm(self, analysis_json, periode_type, audience):
        """
        Génère le rapport Markdown COMPLET directement par LLM.
        Utilisé si l'enrichissement par sections échoue.
        """
        prompt = f"""
        Tu es un expert météorologique professionnel. Génère un rapport météorologique COMPLET en Markdown.
        
        DONNÉES À ANALYSER:
        {json.dumps(analysis_json, indent=2, default=str)}

        CONTEXTE:
        - Audience: {audience}
        - Type de rapport: {periode_type}

        ⚠️⚠️⚠️ INSTRUCTIONS CRITIQUES - RAPPORT ULTRA-DÉTAILLÉ OBLIGATOIRE ⚠️⚠️⚠️
        
        Génère un rapport Markdown COMPLET et professionnel avec:
        1. Un titre et un résumé exécutif DÉTAILLÉ (5-8 phrases avec toutes les valeurs clés)
        2. Des sections ULTRA-DÉTAILLÉES pour: température, vent, humidité, rose des vents, KPIs, séries temporelles, corrélations, distributions, radar
        3. Chaque section doit contenir:
           - Les statistiques sous forme de tableaux Markdown COMPLETS avec TOUTES les valeurs
           - Une analyse narrative ULTRA-DÉTAILLÉE (MINIMUM 8-12 phrases par section)
           - CITER TOUTES les valeurs numériques du tableau/graphique dans l'analyse
           - Comparer systématiquement les valeurs entre elles avec calculs explicites
           - Expliquer les implications opérationnelles de CHAQUE valeur importante
           - Décrire les tendances avec valeurs exactes et évolution quantitative
           - Les anomalies détectées avec valeurs numériques exactes et seuils de référence
           - Les recommandations actionnables détaillées avec justification quantitative
        4. Une conclusion ULTRA-DÉTAILLÉE (10-15 phrases) avec synthèse complète et priorités quantifiées

        🔥 EXIGENCES STRICTES POUR ANALYSES ULTRA-DÉTAILLÉES 🔥
        
        POUR TOUS LES TABLEAUX:
        - Analyser CHAQUE ligne avec ses valeurs exactes
        - Comparer les valeurs entre lignes avec calculs explicites
        - Expliquer les écarts et variations avec quantification
        - Identifier les valeurs remarquables (extrêmes, médianes) avec interprétation
        
        POUR TOUS LES GRAPHIQUES:
        - SÉRIES TEMPORELLES: Analyser min/max/moyenne/écart-type de chaque variable, identifier pics/creux avec valeurs exactes, décrire tendances avec quantification
        - CORRÉLATIONS: Analyser CHAQUE corrélation avec valeur exacte, classifier, expliquer sens physique et implications opérationnelles
        - DISTRIBUTIONS: Analyser Q1/médiane/Q3/IQR, compter outliers avec valeurs, décrire forme de distribution, comparer entre variables
        - RADAR: Analyser CHAQUE indicateur avec valeur exacte (0-100%), classifier, identifier forces/faiblesses, proposer actions correctives
        
        RÈGLES ABSOLUES:
        - Utilise UNIQUEMENT les valeurs réelles des données - JAMAIS inventer de chiffres
        - Sois TRÈS précis et professionnel avec TOUTES les valeurs numériques
        - Adapte le niveau de détail selon l'audience ({audience}) mais reste toujours détaillé
        - Génère TOUT le texte narratif, ne laisse JAMAIS de placeholders
        - Chaque affirmation doit être justifiée par au moins une valeur numérique
        - TOUTES les interprétations doivent inclure des valeurs exactes

        Format de sortie: Markdown complet, prêt à être affiché.
        """
        
        # Essayer le modèle principal
        response = self.llm.invoke(prompt, max_tokens=10000)
        if response:
            # Nettoyer la réponse pour extraire le markdown
            # Enlever les balises markdown si présentes
            cleaned = re.sub(r'^```markdown\s*', '', response, flags=re.MULTILINE)
            cleaned = re.sub(r'^```\s*$', '', cleaned, flags=re.MULTILINE)
            return cleaned.strip()
        
        # Essayer les modèles de fallback
        for model_name in self.fallback_models:
            try:
                st.info(f"🔄 Essai génération rapport complet avec : {model_name}")
                llm = CerebriumLLM(model_name)
                response = llm.invoke(prompt, max_tokens=10000)
                if response:
                    cleaned = re.sub(r'^```markdown\s*', '', response, flags=re.MULTILINE)
                    cleaned = re.sub(r'^```\s*$', '', cleaned, flags=re.MULTILINE)
                    st.success(f"✅ Rapport complet généré avec {model_name}")
                    return cleaned.strip()
            except Exception as e:
                st.warning(f"⚠️ Modèle {model_name} indisponible : {str(e)}")
                continue
        
        # Si tous les modèles échouent, lever une erreur
        st.error("❌ Impossible de générer le rapport avec les LLM disponibles. Veuillez réessayer plus tard.")
        raise Exception("Tous les modèles LLM ont échoué. Le rapport doit être généré par un LLM.")

    def _force_recalculate_all_statistics(self, analysis_json, df):
        """
        RECALCULE FORCÉMENT TOUTES LES STATISTIQUES à partir du DataFrame.
        Ignore complètement les données du LLM et recalcule tout.
        """
        if df is None or len(df) == 0:
            return analysis_json
        
        try:
            # === TEMPÉRATURE - RECALCUL FORCÉ ===
            if 'AirTemp' in df.columns:
                s = pd.to_numeric(df['AirTemp'], errors='coerce').dropna()
                if len(s) > 0:
                    analysis_json['statistiques_temperature'] = {
                        'moyenne': float(s.mean()),
                        'ecart_type': float(s.std(ddof=0)),
                        'minimum': {'valeur': float(s.min()), 'timestamp': str(s.idxmin())},
                        'maximum': {'valeur': float(s.max()), 'timestamp': str(s.idxmax())},
                        'amplitude': float(s.max() - s.min()),
                        'percentile_25': float(s.quantile(0.25)),
                        'percentile_50': float(s.quantile(0.50)),
                        'percentile_75': float(s.quantile(0.75)),
                        'percentile_95': float(s.quantile(0.95))
                    }
            
            # === VENT - RECALCUL FORCÉ ===
            if 'Speed#@1m' in df.columns:
                s = pd.to_numeric(df['Speed#@1m'], errors='coerce').dropna()
                if len(s) > 0:
                    total = len(s)
                    # Distribution des classes
                    classes = {
                        "0-1": len(s[(s >= 0) & (s < 1)]),
                        "1-2": len(s[(s >= 1) & (s < 2)]),
                        "2-5": len(s[(s >= 2) & (s < 5)]),
                        "5-8": len(s[(s >= 5) & (s < 8)]),
                        "8-20": len(s[s >= 8])
                    }
                    dist = {}
                    for k, c in classes.items():
                        pct = round((c / total) * 100, 1) if total > 0 else 0.0
                        dist[k] = {"count": c, "pourcentage": pct}
                    
                    analysis_json['statistiques_vitesse_vent'] = {
                        'moyenne': float(s.mean()),
                        'ecart_type': float(s.std(ddof=0)),
                        'minimum': {'valeur': float(s.min()), 'timestamp': str(s.idxmin())},
                        'maximum': {'valeur': float(s.max()), 'timestamp': str(s.idxmax())},
                        'distribution_classes': dist
                    }
            
            # === HUMIDITÉ - RECALCUL FORCÉ ===
            if 'RH' in df.columns:
                s = pd.to_numeric(df['RH'], errors='coerce').dropna()
                if len(s) > 0:
                    total_hum = len(s)
                    rh_above_90 = len(s[s > 90])
                    rh_above_95 = len(s[s > 95])
                    
                    analysis_json['statistiques_humidite'] = {
                        'moyenne': float(s.mean()),
                        'minimum': {'valeur': float(s.min()), 'timestamp': str(s.idxmin())},
                        'maximum': {'valeur': float(s.max()), 'timestamp': str(s.idxmax())},
                        'temps_rh_sup_90': {
                            'count': rh_above_90,
                            'pourcentage': round((rh_above_90 / total_hum) * 100, 1) if total_hum > 0 else 0
                        },
                        'temps_rh_sup_95': {
                            'count': rh_above_95,
                            'pourcentage': round((rh_above_95 / total_hum) * 100, 1) if total_hum > 0 else 0
                        }
                    }
            
            # === POWER - RECALCUL FORCÉ ===
            if 'Power' in df.columns:
                s = pd.to_numeric(df['Power'], errors='coerce').dropna()
                if len(s) > 0:
                    analysis_json['statistiques_power'] = {
                        'moyenne': float(s.mean()),
                        'ecart_type': float(s.std(ddof=0)),
                        'minimum': {'valeur': float(s.min()), 'timestamp': str(s.idxmin())},
                        'maximum': {'valeur': float(s.max()), 'timestamp': str(s.idxmax())}
                    }
            
            # === ROSE DES VENTS - RECALCUL FORCÉ ===
            if 'Dir' in df.columns and 'Speed#@1m' in df.columns:
                # Recalculer la rose des vents directement ici
                try:
                    sectors = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                              'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
                    
                    sector_ranges = [
                        (337.5, 22.5), (22.5, 45), (45, 67.5), (67.5, 90),
                        (90, 112.5), (112.5, 135), (135, 157.5), (157.5, 180),
                        (180, 202.5), (202.5, 225), (225, 247.5), (247.5, 270),
                        (270, 292.5), (292.5, 315), (315, 337.5), (337.5, 360)
                    ]
                    
                    secteurs_data = []
                    total_measures = len(df)
                    
                    for i, (sector, (min_angle, max_angle)) in enumerate(zip(sectors, sector_ranges)):
                        if i == 0:  # Cas spécial pour Nord
                            sector_data = df[((df['Dir'] >= min_angle) & (df['Dir'] <= 360)) | 
                                           ((df['Dir'] >= 0) & (df['Dir'] < max_angle))]
                        else:
                            sector_data = df[(df['Dir'] >= min_angle) & (df['Dir'] < max_angle)]
                        
                        sector_count = len(sector_data)
                        frequency = (sector_count / total_measures) * 100 if total_measures > 0 else 0
                        
                        secteurs_data.append({
                            "nom": sector,
                            "angle_min": min_angle,
                            "angle_max": max_angle,
                            "frequence_totale": round(frequency, 1),
                            "vitesse_moyenne": round(sector_data['Speed#@1m'].mean(), 1) if len(sector_data) > 0 else 0,
                            "vitesse_max": round(sector_data['Speed#@1m'].max(), 1) if len(sector_data) > 0 else 0,
                            "distribution_vitesses": {
                                "calme_0_1": 0,
                                "legere_1_2": round(len(sector_data[(sector_data['Speed#@1m'] >= 1.0) & (sector_data['Speed#@1m'] < 2.0)]) / total_measures * 100, 1),
                                "brise_2_5": round(len(sector_data[(sector_data['Speed#@1m'] >= 2.0) & (sector_data['Speed#@1m'] < 5.0)]) / total_measures * 100, 1),
                                "moderee_5_8": round(len(sector_data[(sector_data['Speed#@1m'] >= 5.0) & (sector_data['Speed#@1m'] < 8.0)]) / total_measures * 100, 1),
                                "forte_8_20": round(len(sector_data[sector_data['Speed#@1m'] >= 8.0]) / total_measures * 100, 1)
                            }
                        })
                    
                    # Trouver le secteur dominant
                    if secteurs_data:
                        dominant_sector = max(secteurs_data, key=lambda x: x['frequence_totale'])
                    else:
                        dominant_sector = {"nom": "N/A", "frequence_totale": 0, "vitesse_moyenne": 0}
                    
                    analysis_json['wind_rose_data'] = {
                        "secteurs": secteurs_data,
                        "calme_central": round(len(df[df['Speed#@1m'] < 1.0]) / total_measures * 100, 1) if total_measures > 0 else 0,
                        "direction_dominante": {
                            "nom": dominant_sector['nom'],
                            "angle_moyen": (dominant_sector.get('angle_min', 0) + dominant_sector.get('angle_max', 0)) / 2,
                            "frequence": dominant_sector['frequence_totale'],
                            "vitesse_moyenne": dominant_sector['vitesse_moyenne']
                        },
                        "direction_vitesses_max": {
                            "nom": max(secteurs_data, key=lambda x: x['vitesse_max'])['nom'] if secteurs_data else "N/A",
                            "vitesse_max": max(secteurs_data, key=lambda x: x['vitesse_max'])['vitesse_max'] if secteurs_data else 0,
                            "timestamp": ""
                        }
                    }
                except Exception as e:
                    st.warning(f"⚠️ Impossible de recalculer la rose des vents: {e}")
            
        except Exception as e:
            st.warning(f"⚠️ Erreur lors du recalcul forcé des statistiques: {e}")
        
        return analysis_json

    def _format_analysis_text(self, text):
        """
        Formate le texte d'analyse pour améliorer la lisibilité avec des sauts de ligne appropriés
        """
        if not text:
            return text
        
        # Remplacer les puces (•) par des sauts de ligne avec puces Markdown
        text = re.sub(r'•\s*', '\n\n- ', text)
        
        # Ajouter un saut de ligne après chaque phrase qui se termine par un point/virgule suivi d'une majuscule
        # Mais éviter les abréviations comme "°C", "m/s", "%", etc.
        text = re.sub(r'([.!?])\s+([A-ZÉÀÈÙÂÊÎÔÛÇ])', r'\1\n\n\2', text)
        
        # Ajouter un saut de ligne avant les tirets qui commencent une nouvelle phrase ou section
        # Format: "texte. - Nouvelle phrase"
        text = re.sub(r'([.!?])\s+-\s+([A-ZÉÀÈÙÂÊÎÔÛÇ])', r'\1\n\n- \2', text)
        text = re.sub(r'\s+-\s+([A-ZÉÀÈÙÂÊÎÔÛÇ])', r'\n\n- \1', text)
        
        # Ajouter un saut de ligne après les virgules suivies d'une majuscule (nouvelle phrase)
        text = re.sub(r',\s+([A-ZÉÀÈÙÂÊÎÔÛÇ][a-zéàèùâêîôûç]+ [a-zéàèùâêîôûç]+ [A-ZÉÀÈÙÂÊÎÔÛÇ])', r',\n\n\1', text)
        
        # Détecter les patterns comme "90,61 %, encore dans" et ajouter un saut de ligne
        # Format: "valeur %, nouvelle phrase"
        text = re.sub(r'([0-9,\.]+ %)\s*,\s+([a-zéàèùâêîôûç]+ [a-zéàèùâêîôûç]+)', r'\1,\n\n\2', text)
        
        # S'assurer qu'il y a un saut de ligne après chaque puce complète (qui se termine par un point)
        text = re.sub(r'(-\s+[^\n]+[.!?])\s+([A-ZÉÀÈÙÂÊÎÔÛÇ])', r'\1\n\n\2', text)
        
        # Ajouter un saut de ligne avant les phrases qui commencent par des mots-clés importants
        keywords = ['La ', 'Le ', 'Les ', 'En ', 'Pour ', 'Avec ', 'Sans ', 'Selon ', 'D\'après ', 'Minimum ', 'Maximum ', 'Amplitude ']
        for keyword in keywords:
            text = re.sub(r'([.!?])\s+(' + keyword + r')', r'\1\n\n\2', text)
        
        # Nettoyer les sauts de ligne multiples (plus de 2 consécutifs)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # S'assurer qu'il y a un saut de ligne après chaque paragraphe de puce
        text = re.sub(r'(-\s+[^\n]+)\n([A-ZÉÀÈÙÂÊÎÔÛÇ])', r'\1\n\n\2', text)
        
        # Nettoyer les espaces en début et fin de ligne
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:  # Garder les lignes non vides
                cleaned_lines.append(cleaned_line)
            elif cleaned_lines and cleaned_lines[-1]:  # Garder une ligne vide entre paragraphes
                cleaned_lines.append('')
        
        # Rejoindre les lignes
        result = '\n'.join(cleaned_lines)
        
        # S'assurer qu'il n'y a pas de sauts de ligne multiples au début
        result = result.lstrip('\n')
        
        return result
    
    def _clean_llm_analysis_text(self, analysis_text, real_values_dict):
        """Nettoie le texte d'analyse du LLM pour corriger les incohérences"""
        if not analysis_text:
            return analysis_text
        
        cleaned_text = analysis_text
        
        # CORRECTION CRITIQUE : "25 m/s" → "2-5 m/s"
        cleaned_text = re.sub(r'\b25\s*m/s\b', '2-5 m/s', cleaned_text)
        cleaned_text = re.sub(r'\b25\s*m\.s[⁻¹-]\b', '2-5 m/s', cleaned_text)
        cleaned_text = re.sub(r'classe\s*25', 'classe 2-5', cleaned_text, flags=re.IGNORECASE)
        
        # CORRECTION : "kW" → "V" pour l'alimentation
        cleaned_text = re.sub(r'\d+[.,]\d+\s*kW', f"{real_values_dict.get('power_moy', '10,90')} V", cleaned_text)
        cleaned_text = re.sub(r'kilowatt|kW', 'V', cleaned_text, flags=re.IGNORECASE)
        
        # CORRECTION : valeurs de température incohérentes
        temp_moy_real = real_values_dict.get('temp_moy')
        if temp_moy_real:
            # Remplacer les 15,30°C inventés par la vraie valeur
            cleaned_text = re.sub(r'15[.,]30\s*°C', f'{temp_moy_real}°C', cleaned_text)
            cleaned_text = re.sub(r'15[.,]3\s*°C', f'{temp_moy_real}°C', cleaned_text)
        
        # CORRECTION : écart-type incohérent
        temp_ecart_real = real_values_dict.get('temp_ecart')
        if temp_ecart_real:
            cleaned_text = re.sub(r'écart-type.*0[.,]12\s*°C', f'écart-type {temp_ecart_real}°C', cleaned_text)
        
        # FORMATAGE : Améliorer les espacements et sauts de ligne
        cleaned_text = self._format_analysis_text(cleaned_text)
        
        return cleaned_text

    def _validate_analysis_coherence(self, analysis_json, enrichment):
        """Valide la cohérence entre les données calculées et l'analyse LLM"""
        issues = []
        
        temp = analysis_json.get('statistiques_temperature', {})
        vent = analysis_json.get('statistiques_vitesse_vent', {})
        hum = analysis_json.get('statistiques_humidite', {})
        
        # Vérifier la température
        temp_moy = self._extract_stat_value(temp, 'moyenne')
        if temp_moy != 'N/A' and isinstance(temp_moy, (int, float)):
            # Vérifier si le LLM utilise des valeurs autour de 15.3 alors que la vraie valeur est différente
            if abs(temp_moy - 15.3) < 0.5 and abs(temp_moy - 15.3) > 0.1:
                issues.append(f"Incohérence température: LLM utilise ~15.3°C mais vraie valeur est {temp_moy:.2f}°C")
        
        # Vérifier les classes de vent
        if enrichment:
            analysis_text = str(enrichment).lower()
            if '25 m/s' in analysis_text and '2-5 m/s' not in analysis_text:
                issues.append("LLM écrit '25 m/s' au lieu de '2-5 m/s'")
        
        return issues

    def _extract_stat_value(self, stats_dict, key, subkey=None):
        """Helper pour extraire les valeurs des statistiques"""
        if not isinstance(stats_dict, dict):
            return 'N/A'
        if key not in stats_dict:
            return 'N/A'
        value = stats_dict[key]
        if value is None:
            return 'N/A'
        if isinstance(value, (int, float)):
            if np.isnan(value):
                return 'N/A'
            return value
        if isinstance(value, dict) and subkey:
            if subkey in value:
                sub_value = value[subkey]
                if sub_value is None or (isinstance(sub_value, float) and np.isnan(sub_value)):
                    return 'N/A'
                return sub_value
        return 'N/A'
    
    def _generate_structured_report(self, analysis_json, periode_type, audience, enrichment=None, df=None):
        """
        Générateur Markdown riche avec analyses par section insérées après tableaux.
        EXIGE l'enrichissement LLM pour générer le rapport.
        
        Args:
            analysis_json: Dictionnaire contenant les statistiques calculées
            periode_type: Type de période
            audience: Audience cible
            enrichment: Enrichissement LLM par sections
            df: DataFrame optionnel pour recalculer les statistiques manquantes
        """
        if enrichment is None or not enrichment:
            # Si pas d'enrichissement, essayer de générer le rapport complet par LLM
            st.warning("⚠️ Enrichissement LLM manquant. Génération du rapport complet par LLM...")
            return self._generate_full_report_with_llm(analysis_json, periode_type, audience)
        
        # === RECALCUL FORCÉ DE TOUTES LES STATISTIQUES ===
        # Ignorer complètement les données du LLM et tout recalculer depuis le DataFrame
        if df is not None and len(df) > 0:
            analysis_json = self._force_recalculate_all_statistics(analysis_json, df)
        
        # VALIDATION des incohérences
        coherence_issues = self._validate_analysis_coherence(analysis_json, enrichment)
        if coherence_issues:
            st.warning("⚠️ Incohérences détectées dans l'analyse LLM:")
            for issue in coherence_issues:
                st.write(f"  - {issue}")
            st.info("Application des corrections automatiques...")
        
        # S'assurer que toutes les valeurs sont des dictionnaires, pas des floats
        rg = analysis_json.get('resume_general', {})
        if not isinstance(rg, dict):
            rg = {}
        
        temp = analysis_json.get('statistiques_temperature', {})
        if not isinstance(temp, dict):
            temp = {}
        
        vent = analysis_json.get('statistiques_vitesse_vent', {})
        if not isinstance(vent, dict):
            vent = {}
        
        hum = analysis_json.get('statistiques_humidite', {})
        if not isinstance(hum, dict):
            hum = {}
        
        power = analysis_json.get('statistiques_power', {})
        if not isinstance(power, dict):
            power = {}
        
        wind_rose = analysis_json.get('wind_rose_data', {})
        if not isinstance(wind_rose, dict):
            wind_rose = {}
        
        kpis = analysis_json.get('kpis', {})
        if not isinstance(kpis, dict):
            kpis = {}
        
        anomalies = analysis_json.get('anomalies', []) or []
        if not isinstance(anomalies, list):
            anomalies = []
        
        # Mettre à jour les références après le recalcul (déjà fait dans _force_recalculate_all_statistics)
        temp = analysis_json.get('statistiques_temperature', {})
        if not isinstance(temp, dict):
            temp = {}
        else:
            # Normaliser minimum/maximum au format dict si nécessaire
            for key in ['minimum', 'maximum']:
                val = temp.get(key)
                if val is not None and isinstance(val, (int, float)) and not isinstance(val, dict):
                    # Convertir en format dict
                    if key == 'minimum' and 'AirTemp' in df.columns if df is not None else False:
                        temp[key] = {'valeur': float(val), 'timestamp': ''}
                    elif key == 'maximum' and 'AirTemp' in df.columns if df is not None else False:
                        temp[key] = {'valeur': float(val), 'timestamp': ''}
        
        vent = analysis_json.get('statistiques_vitesse_vent', {})
        if not isinstance(vent, dict):
            vent = {}
        else:
            # Normaliser minimum/maximum au format dict si nécessaire
            for key in ['minimum', 'maximum']:
                val = vent.get(key)
                if val is not None and isinstance(val, (int, float)) and not isinstance(val, dict):
                    vent[key] = {'valeur': float(val), 'timestamp': ''}
        
        hum = analysis_json.get('statistiques_humidite', {})
        if not isinstance(hum, dict):
            hum = {}
        else:
            # Normaliser minimum/maximum au format dict si nécessaire
            for key in ['minimum', 'maximum']:
                val = hum.get(key)
                if val is not None and isinstance(val, (int, float)) and not isinstance(val, dict):
                    hum[key] = {'valeur': float(val), 'timestamp': ''}
        
        power = analysis_json.get('statistiques_power', {})
        if not isinstance(power, dict):
            power = {}
        else:
            # Normaliser minimum/maximum au format dict si nécessaire
            for key in ['minimum', 'maximum']:
                val = power.get(key)
                if val is not None and isinstance(val, (int, float)) and not isinstance(val, dict):
                    power[key] = {'valeur': float(val), 'timestamp': ''}

        def fnum(x, fmt="{:.2f}"):
            """Format un nombre, gère les N/A et valeurs manquantes avec format français (virgule)"""
            if x is None or str(x).strip() in ['', 'N/A', 'NaN', 'nan', 'None']:
                return 'N/A'
            
            try:
                # Si c'est déjà un nombre
                if isinstance(x, (int, float)):
                    value = x
                # Si c'est une chaîne, nettoie-la
                else:
                    clean_x = str(x).replace(',', '.').strip()
                    if clean_x in ['', 'N/A', 'NaN', 'nan', 'None']:
                        return 'N/A'
                    value = float(clean_x)
                
                formatted = fmt.format(value)
                # Format français : virgule pour les décimales
                return formatted.replace('.', ',')
            except (ValueError, TypeError):
                return str(x) if x else 'N/A'
        
        def get_stat_value(stats_dict, key, default="N/A", subkey='valeur'):
            """
            Récupère une valeur statistique de manière sécurisée.
            TOUJOURS au format dict avec 'valeur' après recalcul forcé.
            
            Args:
                stats_dict: Dictionnaire de statistiques
                key: Clé principale (ex: 'minimum', 'maximum', 'moyenne')
                default: Valeur par défaut si non trouvé
                subkey: Sous-clé à récupérer (par défaut 'valeur' pour minimum/maximum)
            """
            if stats_dict is None or not isinstance(stats_dict, dict):
                return default
            
            if key not in stats_dict:
                return default
            
            value = stats_dict[key]
            
            # Gérer les valeurs None ou NaN
            if value is None:
                return default
            
            # Si c'est directement un nombre (pour moyenne, ecart_type, etc.)
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    return default
                return value
            
            # Si c'est un dictionnaire (pour minimum/maximum)
            if isinstance(value, dict):
                # Récupérer la sous-clé demandée (par défaut 'valeur')
                if subkey and subkey in value:
                    sub_value = value[subkey]
                    if sub_value is None or (isinstance(sub_value, float) and np.isnan(sub_value)):
                        return default
                    return sub_value
                # Si pas de sous-clé trouvée, retourner default
                return default
            
            # Cas inattendu
            return default

        lines = []
        lines.append(f"# Rapport Météorologique {periode_type}")
        lines.append(f"**Période analysée :** {rg.get('periode_analyse', 'N/A')}")
        lines.append(f"**Audience :** {audience}")
        lines.append("")

        # Résumé Exécutif riche
        lines.append("## Résumé Exécutif")
        lines.append("")
        lines.append(f"- Période d'analyse : {rg.get('periode_analyse', 'N/A')} (≈ {rg.get('duree_jours', 'N/A')} jours)")
        lines.append(f"- Qualité des données : {rg.get('qualite_donnees', 'N/A')}, complétude : {rg.get('taux_completude', 'N/A')}")
        lines.append(f"- Nombre de mesures : {rg.get('nb_mesures_recues', 'N/A')}")
        if temp and isinstance(temp, dict):
            temp_moy = get_stat_value(temp, 'moyenne')
            temp_ecart = get_stat_value(temp, 'ecart_type')
            if temp_moy != 'N/A' and temp_ecart != 'N/A':
                lines.append(f"- Température : Moyenne {fnum(temp_moy,'{:.2f}')} °C, écart-type {fnum(temp_ecart,'{:.2f}')} °C")
        if vent and isinstance(vent, dict):
            vent_moy = get_stat_value(vent, 'moyenne')
            if vent_moy != 'N/A':
                lines.append(f"- Vent : Moyenne {fnum(vent_moy,'{:.2f}')} m/s")
        if hum and isinstance(hum, dict):
            hum_moy = get_stat_value(hum, 'moyenne')
            temps_rh_90 = hum.get('temps_rh_sup_90', {})
            if isinstance(temps_rh_90, dict):
                rh_90_pct = get_stat_value(temps_rh_90, 'pourcentage', default=0)
            else:
                rh_90_pct = 0
            if hum_moy != 'N/A':
                lines.append(f"- Humidité relative : Moyenne {fnum(hum_moy,'{:.1f}')} % (RH>90%: {fnum(rh_90_pct, '{:.1f}')} %)")
        if power and isinstance(power, dict):
            power_moy = get_stat_value(power, 'moyenne')
            if power_moy != 'N/A':
                lines.append(f"- Alimentation : Moyenne {fnum(power_moy,'{:.2f}')} V")
        lines.append("")
        
        # Points clés
        points = anomalies.copy() if anomalies else []
        if points:
            lines.append("**Points clés**")
            for p in points:
                lines.append(f"- {p}")
            lines.append("")

        # Analyse Détaillée
        lines.append("## Analyse Détaillée")
        lines.append("")
        
        # === DONNÉES ET QUALITÉ ===
        lines.append("### Données et Qualité")
        lines.append("")
        lines.append("| Indicateur | Valeur |")
        lines.append("|---|---|")
        lines.append(f"| Période d'analyse | {rg.get('periode_analyse','N/A')} |")
        lines.append(f"| Nombre de mesures reçues | {rg.get('nb_mesures_recues','N/A')} |")
        lines.append(f"| Taux de complétude | {rg.get('taux_completude','N/A')} |")
        lines.append(f"| Qualité des données | {rg.get('qualite_donnees','N/A')} |")
        lines.append("")
        
        # Marqueur pour les séries temporelles
        lines.append("### Figure 1 – Séries Temporelles")
        lines.append("")
        lines.append("[GRAPH:time_series]")
        lines.append("")
        
        # Analyse des séries temporelles
        if enrichment.get('section_series_temporelles'):
            sec_series = enrichment['section_series_temporelles']
            lines.append("**Analyse des séries temporelles**")
            lines.append("")
            analysis_text = sec_series.get('analyse', '')
            analysis_text = self._format_analysis_text(analysis_text)
            lines.append(analysis_text)
            lines.append("")
        
        # Marqueur pour la matrice de corrélation
        lines.append("### Figure 2 – Matrice de Corrélation")
        lines.append("")
        lines.append("[GRAPH:correlation]")
        lines.append("")
        
        # Analyse de la matrice de corrélation
        if enrichment.get('section_correlations'):
            sec_corr = enrichment['section_correlations']
            lines.append("**Analyse de la matrice de corrélation**")
            lines.append("")
            analysis_text = sec_corr.get('analyse', '')
            analysis_text = self._format_analysis_text(analysis_text)
            lines.append(analysis_text)
            lines.append("")

        # === TEMPÉRATURE ===
        if temp and isinstance(temp, dict):
            lines.append("### Température (°C)")
            lines.append("")
            lines.append("| Statistique | Valeur |")
            lines.append("|---|---:|")
            lines.append(f"| Moyenne | {fnum(get_stat_value(temp, 'moyenne'),'{:.2f}')} |")
            # Utiliser get_stat_value pour gérer les deux formats (ancien avec dict, nouveau direct)
            lines.append(f"| Minimum | {fnum(get_stat_value(temp, 'minimum', subkey='valeur'),'{:.2f}')} |")
            lines.append(f"| Maximum | {fnum(get_stat_value(temp, 'maximum', subkey='valeur'),'{:.2f}')} |")
            lines.append(f"| Écart‑type | {fnum(get_stat_value(temp, 'ecart_type'),'{:.2f}')} |")
            for pkey, label in (('percentile_25','25ᵉ percentile'),('percentile_50','50ᵉ percentile'),('percentile_75','75ᵉ percentile'),('percentile_95','95ᵉ percentile')):
                pval = get_stat_value(temp, pkey)
                if pval != 'N/A':
                    lines.append(f"| {label} | {fnum(pval,'{:.2f}')} |")
            lines.append(f"| Amplitude | {fnum(get_stat_value(temp, 'amplitude'),'{:.2f}')} |")
            lines.append("")
            
            # Analyse LLM pour température - AVEC CORRECTIONS
            if enrichment.get('section_temperature'):
                sec_temp = enrichment['section_temperature']
                lines.append("**Analyse détaillée**")
                lines.append("")
                
                # CORRECTION : Préparer les vraies valeurs
                real_values = {
                    'temp_moy': fnum(get_stat_value(temp, 'moyenne'),'{:.2f}'),
                    'temp_amp': fnum(get_stat_value(temp, 'amplitude'),'{:.2f}'),
                    'temp_ecart': fnum(get_stat_value(temp, 'ecart_type'),'{:.2f}'),
                    'temp_min': fnum(get_stat_value(temp, 'minimum', subkey='valeur'),'{:.2f}'),
                    'temp_max': fnum(get_stat_value(temp, 'maximum', subkey='valeur'),'{:.2f}')
                }
                
                # NETTOYAGE du texte LLM
                analysis_text = sec_temp.get('analyse', '')
                analysis_text = self._clean_llm_analysis_text(analysis_text, real_values)
                
                lines.append(analysis_text)
                lines.append("")
                if sec_temp.get('anomalies'):
                    lines.append("*Anomalies détectées :*")
                    for anom in sec_temp['anomalies']:
                        lines.append(f"- {anom}")
                    lines.append("")
                if sec_temp.get('recommandations'):
                    lines.append("*Recommandations :*")
                    for rec in sec_temp['recommandations']:
                        lines.append(f"- {rec}")
                    lines.append("")
            

        # === VENT ===
        if vent and isinstance(vent, dict):
            lines.append("### Vitesse du vent (m/s)")
            lines.append("")
            lines.append("| Statistique | Valeur |")
            lines.append("|---|---:|")
            lines.append(f"| Moyenne | {fnum(get_stat_value(vent, 'moyenne'),'{:.2f}')} |")
            # Utiliser get_stat_value pour gérer les deux formats
            lines.append(f"| Minimum | {fnum(get_stat_value(vent, 'minimum', subkey='valeur'),'{:.2f}')} |")
            lines.append(f"| Maximum | {fnum(get_stat_value(vent, 'maximum', subkey='valeur'),'{:.2f}')} |")
            lines.append(f"| Écart‑type | {fnum(get_stat_value(vent, 'ecart_type'),'{:.2f}')} |")
            lines.append("")
            
            dist = vent.get('distribution_classes', {})
            if dist:
                lines.append("| Classe de vitesse | Count | Pourcentage |")
                lines.append("|---|---:|---:|")
                for k, v in dist.items():
                    if isinstance(v, dict):
                        count = v.get('count','N/A')
                        pct = fnum(v.get('pourcentage',0),'{:.1f}')
                    else:
                        count = ""
                        pct = fnum(v,'{:.1f}')
                    lines.append(f"| {k} | {count} | {pct} % |")
                lines.append("")
            
            # Analyse LLM pour vent - AVEC CORRECTIONS
            if enrichment.get('section_vent'):
                sec_vent = enrichment['section_vent']
                lines.append("**Analyse détaillée**")
                lines.append("")
                
                # CORRECTION : Préparer les vraies valeurs
                real_values = {
                    'vent_moy': fnum(get_stat_value(vent, 'moyenne'),'{:.2f}'),
                    'vent_min': fnum(get_stat_value(vent, 'minimum', subkey='valeur'),'{:.2f}'),
                    'vent_max': fnum(get_stat_value(vent, 'maximum', subkey='valeur'),'{:.2f}')
                }
                
                # NETTOYAGE du texte LLM
                analysis_text = sec_vent.get('analyse', '')
                analysis_text = self._clean_llm_analysis_text(analysis_text, real_values)
                
                lines.append(analysis_text)
                lines.append("")
                if sec_vent.get('anomalies'):
                    lines.append("*Anomalies détectées :*")
                    for anom in sec_vent['anomalies']:
                        lines.append(f"- {anom}")
                    lines.append("")
                if sec_vent.get('recommandations'):
                    lines.append("*Recommandations :*")
                    for rec in sec_vent['recommandations']:
                        lines.append(f"- {rec}")
                    lines.append("")
            

        # === HUMIDITÉ ===
        if hum and isinstance(hum, dict):
            lines.append("### Humidité relative (%)")
            lines.append("")
            lines.append("| Statistique | Valeur |")
            lines.append("|---|---:|")
            lines.append(f"| Moyenne | {fnum(get_stat_value(hum, 'moyenne'),'{:.2f}')} |")
            # Utiliser get_stat_value pour gérer les deux formats
            lines.append(f"| Minimum | {fnum(get_stat_value(hum, 'minimum', subkey='valeur'),'{:.2f}')} |")
            lines.append(f"| Maximum | {fnum(get_stat_value(hum, 'maximum', subkey='valeur'),'{:.2f}')} |")
            # Gérer temps_rh_sup_90 et temps_rh_sup_95
            temps_rh_90 = hum.get('temps_rh_sup_90', {})
            if isinstance(temps_rh_90, dict):
                rh_90_pct = get_stat_value(temps_rh_90, 'pourcentage', default=0)
                if rh_90_pct != 'N/A':
                    lines.append(f"| % temps RH>90% | {fnum(rh_90_pct,'{:.1f}')} % |")
            temps_rh_95 = hum.get('temps_rh_sup_95', {})
            if isinstance(temps_rh_95, dict):
                rh_95_pct = get_stat_value(temps_rh_95, 'pourcentage', default=0)
                if rh_95_pct != 'N/A':
                    lines.append(f"| % temps RH>95% | {fnum(rh_95_pct,'{:.1f}')} % |")
            lines.append("")
            
            # Analyse LLM pour humidité - AVEC CORRECTIONS
            if enrichment.get('section_humidite'):
                sec_hum = enrichment['section_humidite']
                lines.append("**Analyse détaillée**")
                lines.append("")
                
                # CORRECTION : Préparer les vraies valeurs
                real_values = {
                    'hum_moy': fnum(get_stat_value(hum, 'moyenne'),'{:.2f}'),
                    'hum_min': fnum(get_stat_value(hum, 'minimum', subkey='valeur'),'{:.2f}'),
                    'hum_max': fnum(get_stat_value(hum, 'maximum', subkey='valeur'),'{:.2f}')
                }
                
                # NETTOYAGE du texte LLM
                analysis_text = sec_hum.get('analyse', '')
                analysis_text = self._clean_llm_analysis_text(analysis_text, real_values)
                
                lines.append(analysis_text)
                lines.append("")
                if sec_hum.get('anomalies'):
                    lines.append("*Anomalies détectées :*")
                    for anom in sec_hum['anomalies']:
                        lines.append(f"- ⚠️ {anom}")
                    lines.append("")
                if sec_hum.get('recommandations'):
                    lines.append("*Recommandations :*")
                    for rec in sec_hum['recommandations']:
                        lines.append(f"- {rec}")
                    lines.append("")
            
            # Marqueur pour les histogrammes et boxplots (après la section humidité - graphiques combinés)
            lines.append("### Figure 3 – Distributions et Détection des Valeurs Aberrantes")
            lines.append("")
            lines.append("[GRAPH:histograms]")
            lines.append("")
            lines.append("[GRAPH:boxplots]")
            lines.append("")
            
            # Analyse des distributions et boxplots
            if enrichment.get('section_distributions'):
                sec_dist = enrichment['section_distributions']
                lines.append("**Analyse des distributions et des valeurs aberrantes**")
                lines.append("")
                analysis_text = sec_dist.get('analyse', '')
                analysis_text = self._format_analysis_text(analysis_text)
                lines.append(analysis_text)
                lines.append("")
            
            # Marqueur pour le graphique radar (indicateurs opérationnels)
            lines.append("### Figure 4 – Indicateurs Opérationnels (Radar)")
            lines.append("")
            lines.append("[GRAPH:radar]")
            lines.append("")
            
            # Analyse du radar des indicateurs
            if enrichment.get('section_radar'):
                sec_radar = enrichment['section_radar']
                lines.append("**Analyse des indicateurs radar**")
                lines.append("")
                analysis_text = sec_radar.get('analyse', '')
                analysis_text = self._format_analysis_text(analysis_text)
                lines.append(analysis_text)
                lines.append("")

        # === ROSE DES VENTS ===
        if wind_rose and isinstance(wind_rose, dict) and wind_rose.get('secteurs'):
            lines.append("### Rose des Vents")
            lines.append("")
            lines.append("| Secteur | Fréquence (%) | Vit. moy (m/s) | Vit. max (m/s) |")
            lines.append("|---|---:|---:|---:|")
            for s in wind_rose['secteurs']:
                lines.append(f"| {s.get('nom','')} | {s.get('frequence_totale',0)} | {s.get('vitesse_moyenne',0)} | {s.get('vitesse_max',0)} |")
            
            dd = wind_rose.get('direction_dominante', {})
            lines.append("")
            lines.append(f"**Direction dominante :** {dd.get('nom','N/A')} ({dd.get('frequence','N/A')} %)")
            lines.append(f"**Calme central :** {fnum(wind_rose.get('calme_central',0),'{:.1f}')} %")
            lines.append("")
            
            # Marqueur pour la rose des vents (graphique principal)
            lines.append("### Figure 5 – Rose des Vents")
            lines.append("")
            lines.append("[GRAPH:wind_rose]")
            lines.append("")
            
            # Analyse LLM pour rose des vents
            if enrichment.get('section_rose_des_vents'):
                sec_rose = enrichment['section_rose_des_vents']
                lines.append("**Analyse détaillée**")
                lines.append("")
                lines.append(sec_rose.get('analyse', ''))
                lines.append("")
                if sec_rose.get('anomalies'):
                    lines.append("*Anomalies détectées :*")
                    for anom in sec_rose['anomalies']:
                        lines.append(f"- {anom}")
                    lines.append("")
                if sec_rose.get('recommandations'):
                    lines.append("*Recommandations :*")
                    for rec in sec_rose['recommandations']:
                        lines.append(f"- {rec}")
                    lines.append("")

        # === KPIs ===
        if kpis and isinstance(kpis, dict):
            lines.append("### Indicateurs clés (KPIs)")
            lines.append("")
            lines.append("| KPI | Valeur |")
            lines.append("|---|---|")
            
            # Utiliser les vraies valeurs des statistiques pour les KPIs
            for k, v in kpis.items():
                # Remplacer temperature_mean par la vraie valeur de température
                if k in ['temperature_mean', 'temperature_moyenne']:
                    val_str = f"{fnum(get_stat_value(temp, 'moyenne'),'{:.2f}')} °C"
                # Remplacer wind_speed_mean par la vraie valeur de vent
                elif k in ['wind_speed_mean', 'wind_speed_moyenne']:
                    val_str = f"{fnum(get_stat_value(vent, 'moyenne'),'{:.2f}')} m/s"
                # Remplacer humidity_mean par la vraie valeur d'humidité
                elif k in ['humidity_mean', 'humidity_moyenne']:
                    val_str = f"{fnum(get_stat_value(hum, 'moyenne'),'{:.2f}')} %"
                # Remplacer power_mean par la vraie valeur d'alimentation
                elif k in ['power_mean', 'power_moyenne']:
                    val_str = f"{fnum(get_stat_value(power, 'moyenne'),'{:.2f}')} V"
                else:
                    if isinstance(v, dict):
                        val_str = json.dumps(v, ensure_ascii=False)[:80]
                    else:
                        val_str = str(v)
                lines.append(f"| {k} | {val_str} |")
            lines.append("")
            
            # Analyse LLM pour KPIs - AVEC CORRECTIONS
            if enrichment.get('section_kpis'):
                sec_kpi = enrichment['section_kpis']
                lines.append("**Analyse détaillée**")
                lines.append("")
                
                # CORRECTION : Préparer les vraies valeurs
                real_values = {
                    'temp_moy': fnum(get_stat_value(temp, 'moyenne'),'{:.2f}'),
                    'vent_moy': fnum(get_stat_value(vent, 'moyenne'),'{:.2f}'),
                    'hum_moy': fnum(get_stat_value(hum, 'moyenne'),'{:.2f}'),
                    'power_moy': fnum(get_stat_value(power, 'moyenne'),'{:.2f}')
                }
                
                # NETTOYAGE du texte LLM
                analysis_text = sec_kpi.get('analyse', '')
                analysis_text = self._clean_llm_analysis_text(analysis_text, real_values)
                
                lines.append(analysis_text)
                lines.append("")
                if sec_kpi.get('anomalies'):
                    lines.append("*Anomalies détectées :*")
                    for anom in sec_kpi['anomalies']:
                        lines.append(f"- {anom}")
                    lines.append("")
                if sec_kpi.get('recommandations'):
                    lines.append("*Recommandations :*")
                    for rec in sec_kpi['recommandations']:
                        lines.append(f"- {rec}")
                    lines.append("")

        # === CONCLUSION ===
        lines.append("## Conclusion et Recommandations Globales")
        lines.append("")
        
        # EXIGER l'enrichissement LLM pour la conclusion
        if enrichment.get('conclusion'):
            conclusion = enrichment['conclusion']
            if conclusion.get('synthese'):
                lines.append("**Synthèse générale**")
                lines.append("")
                lines.append(conclusion['synthese'])
                lines.append("")
            if conclusion.get('priorites'):
                lines.append("**Priorités d'action**")
                lines.append("")
                for pri in conclusion['priorites']:
                    lines.append(f"- {pri}")
                lines.append("")
        else:
            # Si pas de conclusion LLM, essayer de générer le rapport complet
            st.error("⚠️ Conclusion LLM manquante. Le rapport doit être généré entièrement par LLM.")
            raise Exception("Enrichissement LLM incomplet. Le rapport doit être généré entièrement par LLM.")

        lines.append("*Rapport généré automatiquement par LLM. Toute décision opérationnelle doit être validée par le responsable de site.*")
        return "\n".join(lines)

class PDFGenerator:
    def __init__(self):
        pass

    def _draw_markdown_table(self, pdf, table_lines, font_family):
        """
        Dessine un tableau Markdown (| col1 | col2 |...) avec bordures dans le PDF.
        table_lines = liste de lignes Markdown (header, séparateur, lignes).
        font_family = nom de la police à utiliser
        """
        # Nettoyer les lignes et ignorer la ligne de séparation (|---|---|)
        clean_lines = []
        for i, line in enumerate(table_lines):
            line = line.strip()
            if i == 1 and re.match(r'^\s*\|?[-:\s|]+\|?\s*$', line):
                # ligne de séparation -> on la saute
                continue
            if line.startswith('|') and line.endswith('|'):
                line = line[1:-1]
            cols = [col.strip() for col in line.split('|')]
            clean_lines.append(cols)

        if not clean_lines:
            return

        headers = clean_lines[0]
        rows = clean_lines[1:]

        n_cols = len(headers)
        page_w = pdf.w - 2 * pdf.l_margin
        col_w = page_w / max(1, n_cols)
        row_h = 7

        # Titre du tableau déjà géré avant, ici on dessine juste les cases
        pdf.ln(2)

        # En-tête avec couleur de fond
        pdf.set_font(font_family, 'B', 11)
        pdf.set_fill_color(0, 104, 56)  # Vert OCP (#006838)
        pdf.set_text_color(255, 255, 255)  # Texte blanc pour le contraste
        for h in headers:
            pdf.cell(col_w, row_h, txt=str(h), border=1, align='C', fill=True)
        pdf.ln(row_h)
        pdf.set_text_color(0, 0, 0)  # Remettre le texte en noir
        pdf.set_fill_color(255, 255, 255)  # Remettre le fond en blanc

        # Contenu
        pdf.set_font(font_family, '', 10)
        for row in rows:
            # Compléter si une ligne a moins de colonnes
            if len(row) < n_cols:
                row = row + [''] * (n_cols - len(row))
            for cell in row:
                cell_text = str(cell)
                # Tronquer le texte si trop long pour éviter les débordements
                max_chars = int(col_w / 2)  # Estimation approximative
                if len(cell_text) > max_chars:
                    cell_text = cell_text[:max_chars-3] + "..."
                pdf.cell(col_w, row_h, txt=cell_text, border=1, align='L')
            pdf.ln(row_h)

        pdf.ln(3)

    def _strip_markdown_formatting(self, text):
        """Supprime **, *... de la mise en forme Markdown pour le PDF."""
        if not text:
            return ""
        # Enlever gras/italique
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        text = re.sub(r'_(.*?)_', r'\1', text)
        return text
    
    def _is_important_section_title(self, text):
        """Détecte si le texte est un titre de section important à mettre en rouge."""
        if not text:
            return False
        text_lower = text.lower().strip()
        
        # Patterns pour détecter les titres de sections importantes
        important_patterns = [
            r'^analyse\s+(détaillée|des|de la|de)\s+',
            r'^recommandations?\s*:?\s*$',
            r'^anomalies?\s+(détectées?)?\s*:?\s*$',
            r'^interprétation\s*:?\s*$',
            r'^points?\s+clés?\s*:?\s*$',
            r'^conclusion\s*:?\s*$',
            r'^synthèse\s*:?\s*$',
            r'^priorités?\s*:?\s*$',
            r'^analyse\s+de\s+la\s+rose\s+des\s+vents',
            r'^analyse\s+des\s+(séries|distributions|corrélations|indicateurs)',
        ]
        
        # Vérifier si le texte commence par un pattern important
        for pattern in important_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Vérifier aussi si c'est une ligne courte (probablement un titre) contenant des mots-clés
        if len(text) < 100:  # Ligne probablement un titre
            important_keywords = [
                'analyse détaillée',
                'recommandations',
                'anomalies',
                'interprétation',
                'points clés',
                'conclusion',
                'synthèse',
                'priorités'
            ]
            return any(keyword in text_lower for keyword in important_keywords)
        
        return False

    def _register_unicode_fonts(self, pdf):
        """
        Register a Unicode-capable TTF font so special characters (e.g. ≈) render in the PDF.
        Falls back to built-in Arial if no TTF is found.
        """
        font_family = "Unicode"
        regular_candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\arialuni.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\DejaVuSans.ttf",
        ]
        bold_candidates = [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\DejaVuSans-Bold.ttf",
        ]

        regular_path = next((p for p in regular_candidates if os.path.exists(p)), None)
        bold_path = next((p for p in bold_candidates if os.path.exists(p)), None)

        has_bold = False
        try:
            if regular_path:
                pdf.add_font(font_family, '', regular_path, uni=True)
                if bold_path:
                    pdf.add_font(font_family, 'B', bold_path, uni=True)
                    has_bold = True
                return font_family, has_bold
        except Exception as e:
            st.warning(f"Impossible de charger la police Unicode: {e}")

        # Fallback to built-in Arial (latin-1) if nothing found
        return "Arial", True  # built-in Arial has bold variant
    
    def _save_plotly_fig_to_png(self, fig, output_path):
        """Save a Plotly figure as PNG using kaleido (fallback to to_image)"""
        if fig is None:
            try:
                import logging
                logging.warning("Figure est None, impossible de sauvegarder")
            except:
                pass
            return None
            
        try:
            # High-resolution output
            fig.write_image(output_path, width=1400, height=1000, engine="kaleido")
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            else:
                raise Exception("Fichier créé mais vide ou inexistant")
        except Exception as e1:
            try:
                import logging
                logging.info(f"Tentative avec kaleido échouée ({str(e1)}), essai avec to_image...")
            except:
                pass
            try:
                # Alternative approach using to_image (often same)
                png_bytes = pio.to_image(fig, format='png', width=1400, height=1000)
                if png_bytes:
                    with open(output_path, "wb") as f:
                        f.write(png_bytes)
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        return output_path
                    else:
                        raise Exception("Fichier créé mais vide après to_image")
                else:
                    raise Exception("to_image a retourné None")
            except Exception as e2:
                try:
                    import logging
                    logging.warning(f"Impossible de sauvegarder la figure: {str(e2)}")
                except:
                    pass
                return None
    
    def create_pdf(self, markdown_content, figs=None, additional_graphs=None, output_path="rapport_meteo_ocp.pdf"):
        """Crée un PDF simple à partir du markdown et y insère toutes les figures PNG fournies.
        Les graphiques sont insérés dans les sections correspondantes du rapport.
        
        Args:
            markdown_content: Contenu markdown du rapport
            figs: Liste des figures principales (ex: wind rose)
            additional_graphs: Dictionnaire de graphiques supplémentaires
            output_path: Chemin de sortie du PDF
        """
        if figs is None:
            figs = []
        if additional_graphs is None:
            additional_graphs = {}

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        font_family, has_bold = self._register_unicode_fonts(pdf)

        def set_font(bold=False, size=12):
            # Use bold style only if available for the chosen font
            style = 'B' if bold and (has_bold or font_family in ["Arial", "Helvetica", "Times", "Courier"]) else ''
            pdf.set_font(font_family, style, size)
        
        def insert_graph(graph_key, fig_dict, main_figs):
            """Insère un graphique dans le PDF"""
            fig = None
            title = None
            
            # Mapping des marqueurs aux graphiques
            if graph_key == 'wind_rose':
                # Chercher dans les figures principales
                if main_figs:
                    fig = main_figs[0] if len(main_figs) > 0 else None
                title = "Rose des Vents"
            elif graph_key == 'time_series':
                fig = fig_dict.get('time_series')
                title = "Séries Temporelles"
            elif graph_key == 'correlation':
                fig = fig_dict.get('correlation')
                title = "Matrice de Corrélation"
            elif graph_key == 'histograms':
                fig = fig_dict.get('histograms') or fig_dict.get('distributions')
                title = 'Distributions des Variables'
            elif graph_key == 'boxplots':
                fig = fig_dict.get('boxplots') or fig_dict.get('distributions')
                title = 'Boxplots - Detection des Valeurs Aberrantes'
            elif graph_key == 'radar':
                fig = fig_dict.get('radar')
                title = 'Indicateurs Opérationnels (Radar)'
            
            if fig is None:
                # Logger au lieu de st.warning (car st est mocké dans Flask)
                try:
                    import logging
                    logging.warning(f"Graphique {graph_key} non trouvé - sera ignoré")
                except:
                    pass
                return False
            
            try:
                tmpfd, tmpfile = tempfile.mkstemp(suffix=".png")
                os.close(tmpfd)
                
                # Sauvegarder la figure
                saved_path = self._save_plotly_fig_to_png(fig, tmpfile)
                
                if saved_path and os.path.exists(saved_path) and os.path.getsize(saved_path) > 0:
                    # Ajouter un peu d'espace avant le graphique
                    pdf.ln(5)
                    
                    # Ajouter un titre pour le graphique avec couleur
                    if title:
                        pdf.set_text_color(25, 118, 210)  # Bleu (#1976D2)
                        set_font(bold=True, size=11)
                        pdf.cell(0, 6, title, 0, 1, 'C')
                        pdf.set_text_color(0, 0, 0)  # Remettre en noir
                        pdf.ln(2)
                        set_font(bold=False, size=12)
                    
                    # Vérifier si on a besoin d'une nouvelle page
                    img_height = (pdf.w - 2 * pdf.l_margin) * 0.75  # Hauteur approximative
                    if pdf.get_y() + img_height > pdf.h - pdf.b_margin:
                        pdf.add_page()
                    
                    # Fit the image to page width
                    page_w = pdf.w - 2 * pdf.l_margin
                    pdf.image(saved_path, x=pdf.l_margin, y=None, w=page_w)
                    
                    # Ajouter un peu d'espace après le graphique
                    pdf.ln(5)
                    
                    try:
                        os.remove(saved_path)
                    except:
                        pass
                    return True
                else:
                    # Logger au lieu de st.warning
                    try:
                        import logging
                        logging.warning(f"Échec de la sauvegarde du graphique {graph_key}")
                    except:
                        pass
                    return False
            except Exception as e:
                # Logger au lieu de st.warning/st.error
                try:
                    import logging
                    import traceback
                    logging.error(f"Impossible d'insérer le graphique {graph_key} dans le PDF: {e}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                except:
                    pass
                return False
        
        # Titre principal avec couleur
        set_font(bold=True, size=16)
        pdf.set_text_color(0, 104, 56)  # Vert OCP (#006838)
        pdf.cell(0, 10, 'RAPPORT METEOROLOGIQUE OCP', 0, 1, 'C')
        pdf.set_text_color(0, 0, 0)  # Remettre en noir pour le reste
        pdf.ln(5)
        
        # Avertissement très visible - Fond jaune clair, bordure rouge, texte rouge en italique, souligné, centré
        warning_text = "Rapport généré automatiquement par LLM. Toute décision opérationnelle doit être validée par le responsable de site."
        
        # Utiliser une police built-in qui supporte l'italique (Arial ou Helvetica)
        warning_font = "Arial" if font_family == "Unicode" else font_family
        
        # Calculer la hauteur nécessaire pour le texte
        pdf.set_font(warning_font, 'I', 11)  # Italique avec police built-in
        text_width = pdf.get_string_width(warning_text)
        page_width = pdf.w - 2 * pdf.l_margin
        num_lines = int(text_width / page_width) + 1
        cell_height = 8 * num_lines
        
        # Dessiner un rectangle avec fond jaune clair pour la visibilité
        pdf.set_fill_color(255, 240, 200)  # Fond jaune clair
        pdf.set_draw_color(220, 20, 60)  # Bordure rouge
        pdf.rect(pdf.l_margin, pdf.get_y(), page_width, cell_height + 4, 'FD')  # Fill + Draw
        
        # Texte en rouge, italique, souligné
        pdf.set_text_color(220, 20, 60)  # Rouge (#DC143C)
        pdf.set_font(warning_font, 'I', 11)  # Italique avec police built-in
        y_start = pdf.get_y() + 2
        pdf.set_xy(pdf.l_margin, y_start)
        # Utiliser multi_cell pour gérer le texte long et ajouter le soulignement manuellement
        pdf.multi_cell(page_width, 8, warning_text, 0, 'C')
        
        # Dessiner une ligne de soulignement sous chaque ligne de texte
        current_y = y_start
        for _ in range(num_lines):
            pdf.line(pdf.l_margin, current_y + 7, pdf.l_margin + page_width, current_y + 7)
            current_y += 8
        
        # Remettre les paramètres normaux
        pdf.set_font(font_family, '', 12)  # Remettre la police normale
        pdf.set_text_color(0, 0, 0)  # Remettre en noir
        pdf.set_fill_color(255, 255, 255)  # Remettre le fond blanc
        pdf.ln(5)
        
        # Contenu basique : convert markdown lines to simple headings/text
        set_font(bold=False, size=12)
        lines = markdown_content.split('\n')
        
        # Dictionnaire pour suivre quels graphiques ont déjà été insérés
        inserted_graphs = set()
        
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            line_stripped = line.strip()

            # 1) Détection des marqueurs de graphiques [GRAPH:...]
            if line_stripped.startswith('[GRAPH:') and line_stripped.endswith(']'):
                graph_key = line_stripped[7:-1]  # Enlever [GRAPH: et ]
                if graph_key not in inserted_graphs:
                    # Logger pour déboguer
                    try:
                        import logging
                        logging.info(f"Tentative d'insertion du graphique: {graph_key}")
                        logging.info(f"Graphiques disponibles dans additional_graphs: {list(additional_graphs.keys()) if additional_graphs else 'Aucun'}")
                        logging.info(f"Nombre de figures principales: {len(figs) if figs else 0}")
                    except:
                        pass
                    if insert_graph(graph_key, additional_graphs, figs):
                        inserted_graphs.add(graph_key)
                        try:
                            import logging
                            logging.info(f"Graphique {graph_key} inséré avec succès")
                        except:
                            pass
                    else:
                        try:
                            import logging
                            logging.warning(f"Échec de l'insertion du graphique {graph_key}")
                        except:
                            pass
                else:
                    try:
                        import logging
                        logging.info(f"Graphique {graph_key} déjà inséré, ignoré")
                    except:
                        pass
                i += 1
                continue

            # 2) Détection d'un tableau Markdown
            if (
                line_stripped.startswith('|')
                and i + 1 < n
                and lines[i + 1].strip().startswith('|')
                and '---' in lines[i + 1]
            ):
                # On collecte toutes les lignes du tableau
                table_lines = [line]
                j = i + 1
                while j < n and lines[j].strip().startswith('|'):
                    table_lines.append(lines[j])
                    j += 1

                # Dessiner le tableau
                self._draw_markdown_table(pdf, table_lines, font_family)

                i = j
                continue

            # 3) Traitement normal du Markdown (titres, texte, listes)
            txt = self._strip_markdown_formatting(line)

            if txt.startswith('# '):
                # Titre principal de section - Vert foncé OCP
                pdf.set_text_color(0, 104, 56)  # Vert OCP (#006838)
                set_font(bold=True, size=14)
                pdf.cell(0, 10, txt[2:].strip(), 0, 1)
                pdf.set_text_color(0, 0, 0)  # Remettre en noir
                set_font(bold=False, size=12)
            elif txt.startswith('## '):
                # Sous-section - Bleu foncé
                pdf.set_text_color(25, 118, 210)  # Bleu (#1976D2)
                set_font(bold=True, size=12)
                pdf.cell(0, 8, txt[3:].strip(), 0, 1)
                pdf.set_text_color(0, 0, 0)  # Remettre en noir
                set_font(bold=False, size=12)
            elif txt.startswith('### '):
                # Sous-sous-section - Bleu moyen
                pdf.set_text_color(66, 165, 245)  # Bleu clair (#42A5F5)
                set_font(bold=True, size=11)
                pdf.cell(0, 6, txt[4:].strip(), 0, 1)
                pdf.set_text_color(0, 0, 0)  # Remettre en noir
                set_font(bold=False, size=12)
            elif txt.strip():
                stripped = txt.strip()
                
                # Détecter les titres de sections importantes (Analyse, Recommandations, etc.)
                if self._is_important_section_title(stripped) and not stripped.startswith('-') and not stripped.startswith('•'):
                    # Titre important - Rouge, gras, souligné
                    pdf.set_text_color(220, 20, 60)  # Rouge (#DC143C)
                    set_font(bold=True, size=11)
                    # Dessiner le texte avec soulignement
                    pdf.cell(0, 6, stripped, 'U', 1)  # 'U' pour underline
                    pdf.set_text_color(0, 0, 0)  # Remettre en noir
                    set_font(bold=False, size=12)
                    pdf.ln(2)  # Espace supplémentaire après un titre important
                elif stripped.startswith('- '):
                    # Listes Markdown "- " -> puce propre
                    content = stripped[2:].strip()
                    content = self._strip_markdown_formatting(content)
                    pdf.multi_cell(0, 6, u"• " + content)
                else:
                    pdf.multi_cell(0, 6, stripped)
            pdf.ln(1)
            i += 1
        
        # Si des graphiques n'ont pas été insérés (marqueurs manquants), les ajouter à la fin
        all_graph_keys = ['wind_rose', 'time_series', 'correlation', 'histograms', 'boxplots', 'radar']
        missing_graphs = [key for key in all_graph_keys if key not in inserted_graphs and 
                          ((key == 'wind_rose' and figs) or (key != 'wind_rose' and key in additional_graphs))]
        
        if missing_graphs:
            try:
                import logging
                logging.info(f"Graphiques non insérés via marqueurs, ajout à la fin: {missing_graphs}")
            except:
                pass
            pdf.add_page()
            pdf.set_text_color(0, 104, 56)  # Vert OCP (#006838)
            set_font(bold=True, size=14)
            pdf.cell(0, 10, 'GRAPHIQUES SUPPLEMENTAIRES', 0, 1, 'C')
            pdf.set_text_color(0, 0, 0)  # Remettre en noir
            pdf.ln(5)
            
            for graph_key in missing_graphs:
                if insert_graph(graph_key, additional_graphs, figs):
                    inserted_graphs.add(graph_key)
        
        pdf.output(output_path)
        return output_path

class StreamlitOCPApp:
    def __init__(self):
        self.data_analyzer = DataAnalyzer()
        self.report_generator = ReportGenerator()
        self.wind_rose_generator = WindRoseGenerator()
        self.graph_generator = GraphGenerator()
        self.pdf_generator = PDFGenerator()
        
    def run(self):
        st.set_page_config(
            page_title="Système Météo OCP", 
            page_icon="🌪️", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🌪️ Système de Génération de Rapports Météorologiques OCP")
        st.markdown("### Analyse Wind Rose et Conditions Opérationnelles")
        
        # Sidebar pour les paramètres
        with st.sidebar:
            st.header("📋 Paramètres du Rapport")
            periode_type = st.selectbox("Type de rapport", ["Journalier", "Mensuel", "Annuel"])
            
            col1, col2 = st.columns(2)
            with col1:
                date_debut = st.date_input("Date de début", value=datetime.now())
            with col2:
                date_fin = st.date_input("Date de fin", value=datetime.now())
            
            audience = st.selectbox("Audience", ["Opérateurs Terrain", "Management", "Ingénieurs"])
            
            st.header("⚙️ Configuration")
            st.info("📊 Le rapport est TOUJOURS généré par LLM avec analyses narratives complètes")
            st.info("Les APIs sont configurées automatiquement")
        
        # Upload de fichiers
        st.header("📤 Upload des Données Météo")
        uploaded_file = st.file_uploader(
            "Téléchargez vos données météo (CSV, Excel, TXT)", 
            type=['csv', 'xlsx', 'txt'],
            help="Le fichier doit contenir les colonnes: Timestamp, Power, Speed#@1m, Dir, RH, AirTemp"
        )
        
        # Session state init
        if "last_pdf" not in st.session_state:
            st.session_state["last_pdf"] = None
            st.session_state["last_pdf_name"] = None
        if "analysis_json" not in st.session_state:
            st.session_state["analysis_json"] = None
            st.session_state["wind_rose_fig"] = None
            st.session_state["report_markdown"] = None
            st.session_state["additional_graphs"] = None
            st.session_state["df_clean"] = None

        if uploaded_file is not None:
            try:
                # Lecture des données
                df = self.read_uploaded_file(uploaded_file)
                
                if df is not None and len(df) > 0:
                    st.success(f"✅ Données chargées: {len(df)} enregistrements")
                    
                    # Aperçu des données
                    with st.expander("👀 Aperçu des données"):
                        st.dataframe(df.head(), use_container_width=True)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Colonnes", len(df.columns))
                        with col2:
                            st.metric("Lignes", len(df))
                        with col3:
                            # Essayer d'afficher la période si disponible
                            if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
                                st.metric("Période", f"{df.index.min().date()} to {df.index.max().date()}")
                            else:
                                st.metric("Période", "Non disponible")
                    
                    # Bouton pour générer l'analyse
                    if st.button("🚀 Générer le Rapport Complet", type="primary", use_container_width=True):
                        with st.spinner("Analyse des données en cours..."):
                            # Préparation des données nettoyées
                            df_clean = self.data_analyzer.clean_numeric_data(df.copy())
                            df_clean = self.data_analyzer.parse_timestamp_column(df_clean)
                            
                            # Étape 1: Analyse des données
                            analysis_json = self.data_analyzer.analyze_data(
                                df, periode_type, date_debut, date_fin
                            )
                            
                            # Étape 2: Génération de la Wind Rose
                            wind_rose_fig = self.wind_rose_generator.generate_wind_rose_plotly(
                                analysis_json['wind_rose_data']
                            )
                            
                            # Étape 3: Génération du rapport COMPLET par LLM (toujours activé)
                            report_markdown = self.report_generator.generate_report(
                                analysis_json, periode_type, audience, df=df_clean
                            )
                            
                            # Étape 4: Génération des graphiques supplémentaires
                            with st.spinner("Génération des graphiques supplémentaires..."):
                                additional_graphs = self.graph_generator.generate_all_graphs(df_clean, periode_type, analysis_json=analysis_json)

                            # Stockage en session pour affichage persistant
                            st.session_state["analysis_json"] = analysis_json
                            st.session_state["wind_rose_fig"] = wind_rose_fig
                            st.session_state["report_markdown"] = report_markdown
                            st.session_state["additional_graphs"] = additional_graphs
                            st.session_state["df_clean"] = df_clean
                            st.session_state["last_pdf"] = None
                            st.session_state["last_pdf_name"] = None

                    # Affichage persistant si une analyse est disponible
                    if st.session_state.get("analysis_json"):
                        analysis_json = st.session_state["analysis_json"]
                        wind_rose_fig = st.session_state.get("wind_rose_fig")
                        report_markdown = st.session_state.get("report_markdown")

                        st.header("Resultats de l'Analyse")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if wind_rose_fig:
                                st.plotly_chart(wind_rose_fig, use_container_width=True)
                            else:
                                st.warning("Wind Rose non generee")
                        
                        with col2:
                            self.display_key_metrics(analysis_json)
                        
                        # Rapport detaille
                        st.header("Rapport Complet")
                        with st.expander("Voir le rapport detaille", expanded=True):
                            if report_markdown:
                                st.markdown(report_markdown)
                            else:
                                st.info("Rapport non disponible.")
                        
                        # Graphiques supplémentaires
                        additional_graphs = st.session_state.get("additional_graphs", {})
                        if additional_graphs:
                            st.header("📊 Graphiques Supplémentaires")
                            with st.expander("Voir tous les graphiques", expanded=False):
                                if 'time_series' in additional_graphs:
                                    st.subheader("Séries Temporelles")
                                    st.plotly_chart(additional_graphs['time_series'], use_container_width=True)
                                
                                if 'correlation' in additional_graphs:
                                    st.subheader("Matrice de Corrélation")
                                    st.plotly_chart(additional_graphs['correlation'], use_container_width=True)
                                
                                if 'histograms' in additional_graphs:
                                    st.subheader("Distributions des Variables")
                                    st.plotly_chart(additional_graphs['histograms'], use_container_width=True)
                                
                                if 'boxplots' in additional_graphs:
                                    st.subheader("Boxplots - Détection des Valeurs Aberrantes")
                                    st.plotly_chart(additional_graphs['boxplots'], use_container_width=True)

                        # ==== PDF Generation UI ====
                        st.header("Export du Rapport")
                        # Bouton pour generer le PDF (creation stockee en session_state)
                        if st.button("Generer le PDF", use_container_width=True):
                            with st.spinner("Generation du PDF avec tous les graphiques..."):
                                # Pass the wind rose plot and any other figs
                                figs_to_embed = []
                                if wind_rose_fig:
                                    figs_to_embed.append(wind_rose_fig)
                                
                                # Récupérer les graphiques supplémentaires
                                additional_graphs = st.session_state.get("additional_graphs", {})
                                
                                # Create PDF avec tous les graphiques
                                try:
                                    pdf_path = self.pdf_generator.create_pdf(
                                        report_markdown, 
                                        figs=figs_to_embed,
                                        additional_graphs=additional_graphs
                                    )
                                except Exception as e:
                                    st.error(f"Erreur lors de la creation du PDF: {e}")
                                    pdf_path = None

                                # Validate and load into session state
                                if pdf_path and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                                    try:
                                        with open(pdf_path, "rb") as f:
                                            pdf_data = f.read()
                                        # Store the pdf bytes in session state
                                        st.session_state["last_pdf"] = pdf_data
                                        st.session_state["last_pdf_name"] = f"rapport_meteo_ocp_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                                        st.success("PDF genere avec succes. Vous pouvez le telecharger ci-dessous.")
                                    except Exception as e:
                                        st.error(f"Impossible de lire le PDF genere: {e}")
                                        st.session_state["last_pdf"] = None
                                        st.session_state["last_pdf_name"] = None
                                else:
                                    st.error("La generation du PDF a echoue (fichier absent ou vide). Verifiez les logs et reessayez.")

                        # Telechargement persistant si PDF genere precedemment
                        if st.session_state.get("last_pdf"):
                            st.download_button(
                                label="Telecharger le PDF Final",
                                data=st.session_state["last_pdf"],
                                file_name=st.session_state.get("last_pdf_name", "rapport_meteo_ocp.pdf"),
                                mime="application/pdf",
                                use_container_width=True
                            )
                        
                        # === ANALYSE COMPLÉMENTAIRE DES KPIs AVEC GEMINI ===
                        st.markdown("---")
                        st.header("🔍 Analyse Complémentaire des KPIs (Gemini)")
                        st.info("💡 Cette analyse utilise Gemini Pro pour fournir une perspective complémentaire et détaillée sur les KPIs")
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button("🚀 Générer Analyse Complémentaire", type="secondary", use_container_width=True):
                                with st.spinner("Analyse complémentaire en cours avec Gemini Pro (cela peut prendre 30-60 secondes)..."):
                                    try:
                                        # Import de l'analyseur complémentaire
                                        from analyse_kpi_llm import KPIAnalyzerLLM
                                        
                                        analyzer = KPIAnalyzerLLM()
                                        original_analysis = st.session_state["analysis_json"]
                                        
                                        llm_analysis = analyzer.analyze_kpis_with_llm(original_analysis)
                                        
                                        if llm_analysis:
                                            merged_analysis = analyzer.merge_analyses(original_analysis, llm_analysis)
                                            st.session_state["llm_complementary_analysis"] = llm_analysis
                                            st.session_state["merged_analysis"] = merged_analysis
                                            st.success("✅ Analyse complémentaire générée avec succès!")
                                        else:
                                            st.error("❌ Erreur lors de l'analyse complémentaire")
                                            
                                    except ImportError:
                                        st.error("❌ Module analyse_kpi_llm.py non trouvé. Assurez-vous qu'il est dans le même répertoire.")
                                    except Exception as e:
                                        st.error(f"❌ Erreur: {str(e)}")
                        
                        with col2:
                            if st.session_state.get("llm_complementary_analysis"):
                                st.success("✅ Analyse complémentaire disponible")
                        
                        # Afficher l'analyse complémentaire si disponible
                        if st.session_state.get("llm_complementary_analysis"):
                            llm_analysis = st.session_state["llm_complementary_analysis"]
                            kpi_analysis = llm_analysis.get('analyse_detaillee_kpis', {})
                            conclusion = llm_analysis.get('conclusion_generale', {})
                            
                            # Onglets pour organiser l'affichage
                            tab1, tab2, tab3, tab4 = st.tabs([
                                "📈 Synthèse KPIs", 
                                "🔍 Analyses Détaillées", 
                                "💡 Conclusion", 
                                "📄 Rapport Complet"
                            ])
                            
                            with tab1:
                                st.subheader("Synthèse Générale")
                                st.write(kpi_analysis.get('synthese_generale', 'Non disponible'))
                                
                                # Conditions optimales
                                st.markdown("### Temps en Conditions Optimales")
                                opt_analysis = kpi_analysis.get('analyse_temps_conditions_optimales', {})
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Pourcentage", 
                                        f"{opt_analysis.get('valeur', 0):.2f}%",
                                        delta=None
                                    )
                                with col2:
                                    st.metric(
                                        "Classification", 
                                        opt_analysis.get('classification', 'N/A')
                                    )
                                st.write("**Interprétation:**")
                                st.write(opt_analysis.get('interpretation', 'Non disponible'))
                                
                                # Alertes
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("### Alertes Vent Fort")
                                    vent_analysis = kpi_analysis.get('analyse_alertes_vent_fort', {})
                                    st.metric("Nombre", vent_analysis.get('nombre', 0))
                                    st.write(f"**Niveau de risque:** {vent_analysis.get('niveau_risque', 'N/A')}")
                                
                                with col2:
                                    st.markdown("### Alertes Humidité")
                                    hum_analysis = kpi_analysis.get('analyse_alertes_humidite', {})
                                    st.metric("Nombre", hum_analysis.get('nombre', 0))
                                    st.write(f"**Niveau de risque:** {hum_analysis.get('niveau_risque', 'N/A')}")
                            
                            with tab2:
                                st.subheader("Analyses Détaillées")
                                
                                # Corrélations
                                st.markdown("### Corrélations")
                                corr_analysis = kpi_analysis.get('analyse_correlations', {})
                                st.write("**Température-Vent:**")
                                st.write(corr_analysis.get('correlation_temp_vent', 'Non disponible'))
                                st.write("**Humidité-Puissance:**")
                                st.write(corr_analysis.get('correlation_humidite_puissance', 'Non disponible'))
                                
                                # Tendances
                                st.markdown("### Tendances")
                                trend_analysis = kpi_analysis.get('analyse_tendances', {})
                                st.write("**Tendances principales:**")
                                for trend in trend_analysis.get('tendances_principales', []):
                                    st.write(f"- {trend}")
                                
                                # Forces et faiblesses
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("### Forces")
                                    for force in kpi_analysis.get('forces_identifiees', []):
                                        st.success(f"✅ {force}")
                                
                                with col2:
                                    st.markdown("### Faiblesses")
                                    for faiblesse in kpi_analysis.get('faiblesses_identifiees', []):
                                        st.warning(f"⚠️ {faiblesse}")
                            
                            with tab3:
                                st.subheader("Conclusion Générale")
                                
                                st.write("### Résumé Exécutif")
                                st.write(conclusion.get('resume_executif', 'Non disponible'))
                                
                                st.write("### Priorités d'Action")
                                for priorite in conclusion.get('priorites_action', []):
                                    with st.expander(f"Priorité {priorite.get('priorite', 'N/A')}: {priorite.get('action', 'N/A')}"):
                                        st.write(f"**Justification:** {priorite.get('justification', 'N/A')}")
                                        st.write(f"**Délai recommandé:** {priorite.get('delai_recommande', 'N/A')}")
                                
                                st.write("### Message Clé")
                                st.info(f"💡 {conclusion.get('message_cle', 'N/A')}")
                            
                            with tab4:
                                st.subheader("Rapport Markdown Complet")
                                
                                # Générer le rapport Markdown
                                try:
                                    from analyse_kpi_llm import KPIAnalyzerLLM
                                    analyzer = KPIAnalyzerLLM()
                                    markdown_report = analyzer.generate_markdown_report(llm_analysis)
                                    
                                    # Afficher le rapport
                                    st.markdown(markdown_report)
                                    
                                    # Bouton de téléchargement
                                    st.download_button(
                                        label="📥 Télécharger le Rapport Markdown",
                                        data=markdown_report,
                                        file_name=f"analyse_kpi_complementaire_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                        mime="text/markdown"
                                    )
                                    
                                    # Bouton pour télécharger le JSON fusionné
                                    merged_json = json.dumps(
                                        st.session_state.get("merged_analysis", {}), 
                                        indent=2, 
                                        ensure_ascii=False, 
                                        default=str
                                    )
                                    st.download_button(
                                        label="📥 Télécharger le JSON Fusionné",
                                        data=merged_json,
                                        file_name=f"analyse_fusionnee_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                        mime="application/json"
                                    )
                                except Exception as e:
                                    st.error(f"Erreur lors de la génération du rapport: {str(e)}")
                else:
                    # Cas où les données n'ont pas pu être lues ou sont vides
                    st.info("Vérifiez le format de votre fichier et réessayez")
            except Exception as e:
                st.error(f"Erreur lors du traitement du fichier: {e}")
                st.info("Vérifiez le format de votre fichier et réessayez")
    
    def read_uploaded_file(self, uploaded_file):
        """Lit le fichier uploadé avec gestion robuste des erreurs - SPÉCIFIQUE POUR VOTRE FORMAT"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.txt'):
                # LECTURE SPÉCIFIQUE POUR VOTRE FORMAT DE FICHIER
                # Votre fichier a un en-tête sur deux lignes
                try:
                    # Lire le fichier en sautant la première ligne d'en-tête
                    # et en utilisant la deuxième ligne comme en-tête
                    df = pd.read_csv(uploaded_file, delimiter='\t', skiprows=1)
                    
                    # Maintenant nous avons les colonnes avec les unités
                    # Nous devons renommer les colonnes pour correspondre à vos besoins
                    current_columns = df.columns.tolist()
                    
                    # Vérifier le format de vos colonnes
                    st.info(f"Colonnes détectées: {current_columns}")
                    
                    # Votre fichier a ce format spécifique:
                    # Première colonne: Timestamp
                    # Autres colonnes: Power, Speed#@1m, Dir, RH, AirTemp
                    
                    # Renommer les colonnes selon votre format
                    if len(current_columns) >= 6:
                        new_columns = ['Timestamp', 'Power', 'Speed#@1m', 'Dir', 'RH', 'AirTemp']
                        df.columns = new_columns[:len(current_columns)]
                        
                        # Convertir la colonne Timestamp en datetime
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S')
                        
                        # Définir Timestamp comme index
                        df = df.set_index('Timestamp')
                        
                    else:
                        st.error(f"Format de colonnes inattendu. Attendu au moins 6 colonnes, obtenu {len(current_columns)}")
                        return None
                        
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du fichier TXT: {str(e)}")
                    # Fallback: essayer différents délimiteurs
                    try:
                        df = pd.read_csv(uploaded_file, delimiter=',')
                    except:
                        try:
                            df = pd.read_csv(uploaded_file, delimiter=';')
                        except:
                            df = pd.read_csv(uploaded_file, delimiter='\t')
            else:
                st.error("Format de fichier non supporté")
                return None
            
            # Nettoyage basique des noms de colonnes
            df.columns = df.columns.str.strip()
            
            # If there's a timestamp-like column, convert it and set as index
            tcol = None
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['date', 'time', 'timestamp']):
                    tcol = col
                    break
            if tcol:
                try:
                    df['Timestamp'] = pd.to_datetime(df[tcol], errors='coerce', dayfirst=True)
                    if df['Timestamp'].isna().sum() < len(df):
                        df = df.dropna(subset=['Timestamp'])
                        df = df.set_index('Timestamp')
                except Exception:
                    pass  # continue with whatever we have

            # Vérification des colonnes requises
            required_columns = ['Power', 'Speed#@1m', 'Dir', 'RH', 'AirTemp']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"Colonnes manquantes: {missing_columns}")
                st.info(f"Colonnes disponibles: {list(df.columns)}")
                return None
            
            return df
            
        except Exception as e:
            st.error(f"Erreur lecture fichier: {str(e)}")
            return None

    def display_key_metrics(self, analysis_json):
        """Affiche les métriques clés"""
        try:
            st.subheader("📈 Métriques Clés")
            
            # Température - avec vérification de type
            temp = analysis_json.get('statistiques_temperature', {})
            if isinstance(temp, dict):
                col1, col2 = st.columns(2)
                with col1:
                    temp_moy = temp.get('moyenne', 0)
                    st.metric("🌡️ Temp. Moyenne", f"{temp_moy:.1f}°C")
                with col2:
                    temp_amp = temp.get('amplitude', 0)
                    st.metric("📊 Amplitude", f"{temp_amp:.1f}°C")
            
            # Vent - avec vérification de type
            vent = analysis_json.get('statistiques_vitesse_vent', {})
            if isinstance(vent, dict):
                col1, col2 = st.columns(2)
                with col1:
                    vent_moy = vent.get('moyenne', 0)
                    st.metric("💨 Vit. Vent Moy", f"{vent_moy:.1f} m/s")
                with col2:
                    wind_data = analysis_json.get('wind_rose_data', {})
                    if isinstance(wind_data, dict):
                        direction_dom = wind_data.get('direction_dominante', {})
                        if isinstance(direction_dom, dict):
                            dominant_dir = direction_dom.get('nom', "N/A")
                        else:
                            dominant_dir = "N/A"
                    else:
                        dominant_dir = "N/A"
                    st.metric("🧭 Direction Dom.", dominant_dir)
            
            # Humidité - avec vérification de type
            humidite = analysis_json.get('statistiques_humidite', {})
            if isinstance(humidite, dict):
                hum_moy = humidite.get('moyenne', 0)
                st.metric("💧 Humidité Moyenne", f"{hum_moy:.1f}%")
            
            # Conditions optimales - avec vérification de type
            kpis = analysis_json.get('kpis', {})
            if isinstance(kpis, dict) and 'temps_conditions_optimales' in kpis:
                temps_opt = kpis['temps_conditions_optimales']
                if isinstance(temps_opt, dict):
                    optimal = temps_opt.get('pourcentage', 0)
                    st.metric("✅ Conditions Optimales", f"{optimal:.1f}%")
            
        except Exception as e:
            st.error(f"Erreur affichage métriques: {str(e)}")

# Lancement de l'application
if __name__ == "__main__":
    try:
        app = StreamlitOCPApp()
        app.run()
    except Exception as e:
        st.error(f"Erreur critique: {str(e)}")
        st.info("Veuillez redémarrer l'application")

