#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chatbot adapté pour Windy Safi - Utilise les données GP2 et Open Meteo
Architecture RAG basée sur llama.py avec hybrid retrieval
"""

import os
import json
import re
import time
import uuid
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

import numpy as np
import requests
from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration de l'API Cerebras (si disponible)
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "csk-84rhjjx5mjenwtpfxew62tvmj4vwwvy4jdrjnjejcmyw4tkf")
os.environ["CEREBRAS_API_KEY"] = CEREBRAS_API_KEY

# Dossier pour sauvegarder les conversations
CONVERSATIONS_DIR = "conversations_weather"
if not os.path.exists(CONVERSATIONS_DIR):
    os.makedirs(CONVERSATIONS_DIR)

logger = logging.getLogger(__name__)

try:
    from langchain_cerebras import ChatCerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    logger.warning("langchain_cerebras non disponible, utilisation d'un mode dégradé")
    CEREBRAS_AVAILABLE = False

# Configuration des limites de tokens pour Llama 3.1-8B sur Cerebras (identique à llama.py)
TOKEN_LIMITS = {
    "max_requests_per_minute": 30,
    "max_tokens_per_minute": 64000,
    "max_requests_per_hour": 900,
    "max_tokens_per_hour": 1000000,
    "max_requests_per_day": 14400,
    "max_tokens_per_day": 1000000,
    "max_tokens_per_request": 8000,
    "chunk_size": 4000,
    "delay_between_requests": 2,
    "max_retries": 3
}

# Configuration RAG (identique à llama.py)
RAG_CONFIG = {
    "general": {
        "chunk_size": 1200,
        "chunk_overlap": 300,
        "separators": ["\n\n", "\n", " ", ""]
    }
}

# Modèle de re-ranking (cross-encoder)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class UsageStats:
    """Statistiques d'utilisation Cerebras (identique à llama.py)"""
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    requests_this_hour: int = 0
    tokens_this_hour: int = 0
    requests_this_day: int = 0
    tokens_this_day: int = 0
    last_minute_reset: float = 0
    last_hour_reset: float = 0
    last_day_reset: float = 0
    request_times: List[float] = None
    
    def __post_init__(self):
        if self.request_times is None:
            self.request_times = []


class CerebrasTokenManager:
    """Gestionnaire de tokens Cerebras (identique à llama.py)"""
    
    def __init__(self):
        self.stats = UsageStats()
        self.stats.last_minute_reset = time.time()
        self.stats.last_hour_reset = time.time()
        self.stats.last_day_reset = time.time()
        
    def count_tokens(self, text: str) -> int:
        return len(text) // 3
    
    def reset_counters_if_needed(self):
        current_time = time.time()
        
        if current_time - self.stats.last_minute_reset >= 60:
            self.stats.requests_this_minute = 0
            self.stats.tokens_this_minute = 0
            self.stats.last_minute_reset = current_time
            self.stats.request_times = [t for t in self.stats.request_times if current_time - t < 60]
        
        if current_time - self.stats.last_hour_reset >= 3600:
            self.stats.requests_this_hour = 0
            self.stats.tokens_this_hour = 0
            self.stats.last_hour_reset = current_time
        
        if current_time - self.stats.last_day_reset >= 86400:
            self.stats.requests_this_day = 0
            self.stats.tokens_this_day = 0
            self.stats.last_day_reset = current_time
    
    def can_make_request(self, estimated_tokens: int) -> Dict[str, Any]:
        self.reset_counters_if_needed()
        
        checks = {
            "can_proceed": True,
            "blocking_limit": None,
            "wait_time": 0,
            "details": {}
        }
        
        if (self.stats.requests_this_minute + 1) > TOKEN_LIMITS["max_requests_per_minute"]:
            checks["can_proceed"] = False
            checks["blocking_limit"] = "requests_per_minute"
            checks["wait_time"] = 60 - (time.time() - self.stats.last_minute_reset)
        elif (self.stats.tokens_this_minute + estimated_tokens) > TOKEN_LIMITS["max_tokens_per_minute"]:
            checks["can_proceed"] = False
            checks["blocking_limit"] = "tokens_per_minute"
            checks["wait_time"] = 60 - (time.time() - self.stats.last_minute_reset)
        elif (self.stats.requests_this_hour + 1) > TOKEN_LIMITS["max_requests_per_hour"]:
            checks["can_proceed"] = False
            checks["blocking_limit"] = "requests_per_hour"
            checks["wait_time"] = 3600 - (time.time() - self.stats.last_hour_reset)
        elif (self.stats.tokens_this_hour + estimated_tokens) > TOKEN_LIMITS["max_tokens_per_hour"]:
            checks["can_proceed"] = False
            checks["blocking_limit"] = "tokens_per_hour"
            checks["wait_time"] = 3600 - (time.time() - self.stats.last_hour_reset)
        elif (self.stats.requests_this_day + 1) > TOKEN_LIMITS["max_requests_per_day"]:
            checks["can_proceed"] = False
            checks["blocking_limit"] = "requests_per_day"
            checks["wait_time"] = 86400 - (time.time() - self.stats.last_day_reset)
        elif (self.stats.tokens_this_day + estimated_tokens) > TOKEN_LIMITS["max_tokens_per_day"]:
            checks["can_proceed"] = False
            checks["blocking_limit"] = "tokens_per_day"
            checks["wait_time"] = 86400 - (time.time() - self.stats.last_day_reset)
        
        checks["details"] = {
            "current_minute": {
                "requests": self.stats.requests_this_minute,
                "tokens": self.stats.tokens_this_minute
            },
            "current_hour": {
                "requests": self.stats.requests_this_hour,
                "tokens": self.stats.tokens_this_hour
            },
            "current_day": {
                "requests": self.stats.requests_this_day,
                "tokens": self.stats.tokens_this_day
            }
        }
        
        return checks
    
    def wait_if_needed(self, estimated_tokens: int):
        """Attend si nécessaire pour respecter les limites (sans Streamlit)"""
        check_result = self.can_make_request(estimated_tokens)
        
        if not check_result["can_proceed"]:
            wait_time = check_result["wait_time"]
            limit_type = check_result["blocking_limit"]
            
            limit_names = {
                "requests_per_minute": "requêtes par minute",
                "tokens_per_minute": "tokens par minute",
                "requests_per_hour": "requêtes par heure",
                "tokens_per_hour": "tokens par heure",
                "requests_per_day": "requêtes par jour",
                "tokens_per_day": "tokens par jour"
            }
            
            logger.warning(f"⏳ Limite Cerebras atteinte: {limit_names.get(limit_type, limit_type)}")
            logger.info(f"⏰ Attente de {wait_time:.1f} secondes...")
            time.sleep(wait_time)
            self.reset_counters_if_needed()
    
    def record_request(self, tokens_used: int):
        current_time = time.time()
        
        self.stats.requests_this_minute += 1
        self.stats.requests_this_hour += 1
        self.stats.requests_this_day += 1
        
        self.stats.tokens_this_minute += tokens_used
        self.stats.tokens_this_hour += tokens_used
        self.stats.tokens_this_day += tokens_used
        
        self.stats.request_times.append(current_time)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'utilisation (identique à llama.py)"""
        self.reset_counters_if_needed()
        
        return {
            "minute": {
                "requests": f"{self.stats.requests_this_minute}/{TOKEN_LIMITS['max_requests_per_minute']}",
                "tokens": f"{self.stats.tokens_this_minute:,}/{TOKEN_LIMITS['max_tokens_per_minute']:,}",
                "requests_percent": (self.stats.requests_this_minute / TOKEN_LIMITS['max_requests_per_minute']) * 100,
                "tokens_percent": (self.stats.tokens_this_minute / TOKEN_LIMITS['max_tokens_per_minute']) * 100
            },
            "hour": {
                "requests": f"{self.stats.requests_this_hour}/{TOKEN_LIMITS['max_requests_per_hour']}",
                "tokens": f"{self.stats.tokens_this_hour:,}/{TOKEN_LIMITS['max_tokens_per_hour']:,}",
                "requests_percent": (self.stats.requests_this_hour / TOKEN_LIMITS['max_requests_per_hour']) * 100,
                "tokens_percent": (self.stats.tokens_this_hour / TOKEN_LIMITS['max_tokens_per_hour']) * 100
            },
            "day": {
                "requests": f"{self.stats.requests_this_day}/{TOKEN_LIMITS['max_requests_per_day']}",
                "tokens": f"{self.stats.tokens_this_day:,}/{TOKEN_LIMITS['max_tokens_per_day']:,}",
                "requests_percent": (self.stats.requests_this_day / TOKEN_LIMITS['max_requests_per_day']) * 100,
                "tokens_percent": (self.stats.tokens_this_day / TOKEN_LIMITS['max_tokens_per_day']) * 100
            }
        }


# Instance globale du gestionnaire de tokens Cerebras
cerebras_token_manager = CerebrasTokenManager()


def safe_cerebras_call(llm, prompt: str, max_retries: int = TOKEN_LIMITS["max_retries"]) -> str:
    """Appel sécurisé au LLM Cerebras avec gestion complète des erreurs (identique à llama.py)"""
    estimated_tokens = cerebras_token_manager.count_tokens(prompt)
    cerebras_token_manager.wait_if_needed(estimated_tokens)
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(TOKEN_LIMITS["delay_between_requests"])
            
            response = llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            response_tokens = cerebras_token_manager.count_tokens(response_text)
            cerebras_token_manager.record_request(estimated_tokens + response_tokens)
            
            return response_text
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "too many requests" in error_msg:
                wait_time = (attempt + 1) * 30
                logger.warning(f"⚠ Limite Cerebras atteinte. Attente de {wait_time} secondes... (Tentative {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif "token" in error_msg and "limit" in error_msg:
                logger.error("❌ Limite de tokens dépassée. Essayez avec un texte plus court.")
                break
            else:
                if attempt == max_retries - 1:
                    logger.error(f"❌ Erreur Cerebras après {max_retries} tentatives: {str(e)}")
                    break
                else:
                    logger.warning(f"⚠ Erreur Cerebras (tentative {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(5)
    
    return "Désolé, je n'ai pas pu traiter votre demande en raison de limitations techniques avec Cerebras."


def detect_language_simple(text: str) -> str:
    """Détecte la langue d'un texte de manière simple (sans Cerebras)"""
    if not text.strip():
        return "fr"
    
    text_lower = text.lower()
    if any(word in text_lower for word in ["the", "and", "is", "are", "you", "your"]):
        return "en"
    elif any(word in text_lower for word in ["el", "la", "es", "son", "tu", "su"]):
        return "es"
    elif any(word in text_lower for word in ["der", "die", "das", "ist", "sind", "ihr"]):
        return "de"
    elif any(word in text_lower for word in ["il", "la", "è", "sono", "tu", "tuo"]):
        return "it"
    elif any(word in text_lower for word in ["في", "من", "إلى", "هذا", "التي"]):
        return "ar"
    else:
        return "fr"


def generate_smart_title(first_message: str) -> str:
    """Génère un titre intelligent pour la conversation (identique à llama.py)"""
    try:
        title_prompt = f"""Tu es un expert en création de titres courts et descriptifs pour des conversations de chat météorologique.

TÂCHE: Créer un titre court (3-6 mots maximum) qui résume parfaitement le sujet de cette question.

QUESTION DE L'UTILISATEUR: {first_message[:300]}

RÈGLES STRICTES:
1. Maximum 6 mots
2. Pas de guillemets, apostrophes, ou caractères spéciaux, pas de hashtag
3. Pas de ponctuation à la fin
4. Utilise des mots simples et clairs
5. Commence par un verbe d'action ou un nom
6. Évite les articles (le, la, les, un, une)

EXEMPLES DE BONS TITRES:
- Question: "Quelle est la température moyenne cette semaine ?" → Titre: "Température moyenne semaine"
- Question: "Quel est le vent à GP2 ?" → Titre: "Vent station GP2"
- Question: "Prévisions météo demain" → Titre: "Prévisions météo demain"

RÉPONSE (seulement le titre, rien d'autre):"""

        if CEREBRAS_AVAILABLE:
            try:
                llm = ChatCerebras(
                    model="llama3.1-8b",
                    temperature=0.1,
                    max_tokens=50
                )
                response = safe_cerebras_call(llm, title_prompt)
                title = response.strip()
            except:
                title = ""
        else:
            title = ""
        
        # Nettoyer le titre
        title = title.replace('"', '').replace("'", '').replace('`', '')
        title = title.replace('«', '').replace('»', '').replace('"', '').replace('"', '')
        title = title.replace(':', '').replace(';', '').replace('!', '').replace('?', '')
        title = title.replace('\n', ' ').replace('\r', ' ').replace('#', '')
        title = ' '.join(title.split())
        
        if len(title) > 40:
            words = title.split()
            title = ' '.join(words[:5])
        
        if not title or len(title) < 3:
            words = first_message.split()[:10]
            important_words = [w for w in words if len(w) > 3 and w.lower() not in ['comment', 'puis', 'peux', 'veux', 'faire', 'avec', 'pour', 'dans', 'cette', 'quelle', 'quel']]
            title = ' '.join(important_words[:3]) if important_words else first_message[:25]
        
        return title.strip()
    except Exception as e:
        logger.error(f"Erreur lors de la génération du titre: {e}")
        words = first_message.split()[:5]
        return ' '.join(words).replace('?', '').replace('!', '')[:30]


def save_conversation(messages: List[Dict], title: Optional[str] = None) -> Optional[str]:
    """Sauvegarde une conversation dans un fichier JSON (adapté de llama.py)"""
    try:
        if not messages or len(messages) == 0:
            logger.warning("❌ Aucun message à sauvegarder")
            return None
        
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if len(user_messages) == 0:
            logger.warning("❌ Aucun message utilisateur trouvé")
            return None
        
        try:
            if not os.path.exists(CONVERSATIONS_DIR):
                os.makedirs(CONVERSATIONS_DIR)
        except Exception as e:
            logger.error(f"❌ Impossible de créer le dossier: {e}")
            return None
        
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        if not title:
            first_user_message = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
            if first_user_message:
                title = generate_smart_title(first_user_message)
            else:
                title = "Conversation météo"
        
        conversation_data = {
            "id": conversation_id,
            "title": title,
            "mode": "weather_chat",
            "timestamp": timestamp.isoformat(),
            "messages": messages,
            "message_count": len(messages)
        }
        
        filename = f"{CONVERSATIONS_DIR}/conv_{conversation_id}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            if os.path.exists(filename):
                logger.info(f"✅ Conversation sauvegardée: {title}")
                return conversation_id
            else:
                logger.error("❌ Le fichier n'a pas été créé")
                return None
        except PermissionError:
            logger.error("❌ Erreur de permissions - impossible d'écrire le fichier")
            return None
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'écriture: {e}")
            return None
    except Exception as e:
        logger.error(f"❌ Erreur générale lors de la sauvegarde: {e}")
        return None


def load_conversations() -> List[Dict]:
    """Charge toutes les conversations sauvegardées"""
    conversations = []
    if not os.path.exists(CONVERSATIONS_DIR):
        return conversations
    
    for filename in os.listdir(CONVERSATIONS_DIR):
        if filename.startswith("conv_") and filename.endswith(".json"):
            try:
                with open(f"{CONVERSATIONS_DIR}/{filename}", 'r', encoding='utf-8') as f:
                    conv_data = json.load(f)
                conversations.append(conv_data)
            except Exception as e:
                logger.error(f"Erreur lors du chargement de {filename}: {e}")
    
    conversations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return conversations


def load_conversation(conversation_id: str) -> Optional[Dict]:
    """Charge une conversation spécifique"""
    filename = f"{CONVERSATIONS_DIR}/conv_{conversation_id}.json"
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la conversation: {e}")
        return None


def delete_conversation(conversation_id: str) -> bool:
    """Supprime une conversation"""
    filename = f"{CONVERSATIONS_DIR}/conv_{conversation_id}.json"
    try:
        os.remove(filename)
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la suppression: {e}")
        return False


def update_existing_conversation(conversation_id: str, messages: List[Dict]) -> Optional[str]:
    """Met à jour une conversation existante avec les nouveaux messages"""
    filename = f"{CONVERSATIONS_DIR}/conv_{conversation_id}.json"
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            conv_data = json.load(f)
        
        conv_data['messages'] = messages
        conv_data['message_count'] = len(messages)
        conv_data['timestamp'] = datetime.now().isoformat()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conv_data, f, ensure_ascii=False, indent=2)
        
        return conversation_id
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour: {e}")
        return save_conversation(messages)


def get_location_coordinates(location_name: str) -> Optional[Dict[str, Any]]:
    """Retourne les coordonnées (lat, lon) pour un nom de lieu/pays"""
    # Mapping de pays/régions vers leurs coordonnées représentatives
    location_map = {
        # France
        "france": {"lat": 46.2276, "lon": 2.2137, "name": "France (centre)"},
        "paris": {"lat": 48.8566, "lon": 2.3522, "name": "Paris, France"},
        "lyon": {"lat": 45.7640, "lon": 4.8357, "name": "Lyon, France"},
        "marseille": {"lat": 43.2965, "lon": 5.3698, "name": "Marseille, France"},
        "toulouse": {"lat": 43.6047, "lon": 1.4442, "name": "Toulouse, France"},
        "nice": {"lat": 43.7102, "lon": 7.2620, "name": "Nice, France"},
        "bordeaux": {"lat": 44.8378, "lon": -0.5792, "name": "Bordeaux, France"},
        
        # Espagne
        "espagne": {"lat": 40.4637, "lon": -3.7492, "name": "Espagne (centre)"},
        "spain": {"lat": 40.4637, "lon": -3.7492, "name": "Espagne (centre)"},
        "madrid": {"lat": 40.4168, "lon": -3.7038, "name": "Madrid, Espagne"},
        "barcelone": {"lat": 41.3851, "lon": 2.1734, "name": "Barcelone, Espagne"},
        "barcelona": {"lat": 41.3851, "lon": 2.1734, "name": "Barcelone, Espagne"},
        
        # Italie
        "italie": {"lat": 41.8719, "lon": 12.5674, "name": "Italie (centre)"},
        "italy": {"lat": 41.8719, "lon": 12.5674, "name": "Italie (centre)"},
        "rome": {"lat": 41.9028, "lon": 12.4964, "name": "Rome, Italie"},
        "milan": {"lat": 45.4642, "lon": 9.1900, "name": "Milan, Italie"},
        "naples": {"lat": 40.8518, "lon": 14.2681, "name": "Naples, Italie"},
        
        # Allemagne
        "allemagne": {"lat": 51.1657, "lon": 10.4515, "name": "Allemagne (centre)"},
        "germany": {"lat": 51.1657, "lon": 10.4515, "name": "Allemagne (centre)"},
        "berlin": {"lat": 52.5200, "lon": 13.4050, "name": "Berlin, Allemagne"},
        "munich": {"lat": 48.1351, "lon": 11.5820, "name": "Munich, Allemagne"},
        "hamburg": {"lat": 53.5511, "lon": 9.9937, "name": "Hamburg, Allemagne"},
        
        # Royaume-Uni
        "angleterre": {"lat": 52.3555, "lon": -1.1743, "name": "Angleterre (centre)"},
        "england": {"lat": 52.3555, "lon": -1.1743, "name": "Angleterre (centre)"},
        "londres": {"lat": 51.5074, "lon": -0.1278, "name": "Londres, Royaume-Uni"},
        "london": {"lat": 51.5074, "lon": -0.1278, "name": "Londres, Royaume-Uni"},
        "royaume-uni": {"lat": 51.5074, "lon": -0.1278, "name": "Royaume-Uni"},
        "uk": {"lat": 51.5074, "lon": -0.1278, "name": "Royaume-Uni"},
        
        # Belgique
        "belgique": {"lat": 50.5039, "lon": 4.4699, "name": "Belgique (centre)"},
        "belgium": {"lat": 50.5039, "lon": 4.4699, "name": "Belgique (centre)"},
        "bruxelles": {"lat": 50.8503, "lon": 4.3517, "name": "Bruxelles, Belgique"},
        "brussels": {"lat": 50.8503, "lon": 4.3517, "name": "Bruxelles, Belgique"},
        
        # Suisse
        "suisse": {"lat": 46.8182, "lon": 8.2275, "name": "Suisse (centre)"},
        "switzerland": {"lat": 46.8182, "lon": 8.2275, "name": "Suisse (centre)"},
        "zurich": {"lat": 47.3769, "lon": 8.5417, "name": "Zurich, Suisse"},
        "genève": {"lat": 46.2044, "lon": 6.1432, "name": "Genève, Suisse"},
        "geneva": {"lat": 46.2044, "lon": 6.1432, "name": "Genève, Suisse"},
        
        # Portugal
        "portugal": {"lat": 39.3999, "lon": -8.2245, "name": "Portugal (centre)"},
        "lisbonne": {"lat": 38.7223, "lon": -9.1393, "name": "Lisbonne, Portugal"},
        "lisbon": {"lat": 38.7223, "lon": -9.1393, "name": "Lisbonne, Portugal"},
    }
    
    location_lower = location_name.lower().strip()
    return location_map.get(location_lower)


def fetch_global_weather_data(lat: float, lon: float, location_name: str, model: str = "auto") -> Optional[Dict]:
    """Récupère les données météorologiques globales via Open-Meteo pour une coordonnée"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms",
            "forecast_days": 7,
            "timezone": "UTC",
        }
        
        if model and model != "auto":
            params["models"] = model
        
        logger.info(f"Récupération données globales Open-Meteo pour {location_name} ({lat}, {lon})")
        
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout lors de la récupération des données globales pour {location_name}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Erreur de connexion lors de la récupération des données globales pour {location_name}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Erreur de requête lors de la récupération des données globales pour {location_name}: {e}")
            return None
        
        try:
            data = resp.json()
        except ValueError as e:
            logger.warning(f"Erreur de parsing JSON pour {location_name}: {e}")
            return None
        
        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        times = hourly.get("time", [])
        
        if not temps or not times:
            logger.warning(f"Aucune donnée de température trouvée pour {location_name}")
            return None
        
        # Calculer les statistiques
        temps_valid = [t for t in temps if t is not None]
        if not temps_valid:
            logger.warning(f"Aucune température valide trouvée pour {location_name}")
            return None
        
        return {
            "location": location_name,
            "lat": lat,
            "lon": lon,
            "model": model,
            "times": times,
            "temperatures": temps,
            "stats": {
                "mean": float(np.mean(temps_valid)),
                "min": float(np.min(temps_valid)),
                "max": float(np.max(temps_valid)),
                "count": len(temps_valid)
            },
            "available": True
        }
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la récupération des données globales pour {location_name}: {e}", exc_info=True)
        return None


def is_global_question(question: str) -> bool:
    """Détecte si la question est une question globale/générale qui nécessite des données mondiales"""
    question_lower = question.lower()
    
    # Mots-clés indiquant explicitement une question globale/mondiale
    global_keywords_explicit = [
        "du globe", "du monde", "mondiale", "globale", "planétaire",
        "partout", "tous les pays", "tous les continents", "toute la planète",
        "climat mondial", "météo mondiale", "conditions mondiales",
        "température mondiale", "température globale", "température planétaire"
    ]
    
    # Si la question contient explicitement des mots indiquant le globe/monde, c'est global
    if any(keyword in question_lower for keyword in global_keywords_explicit):
        return True
    
    # Sinon, si la question mentionne "zone la plus chaude" ou similaire SANS mentionner "du globe",
    # c'est probablement sur la zone visualisée (pas global)
    if any(phrase in question_lower for phrase in [
        "zone la plus chaude", "zone la plus froide", "endroit le plus chaud", "endroit le plus froid"
    ]):
        # Si pas de mention explicite du globe/monde, considérer comme question locale
        return False
    
    return False


def detect_and_fetch_global_data(question: str) -> Optional[Dict]:
    """Détecte si la question concerne un autre pays/région et récupère les données si nécessaire"""
    # DÉSACTIVÉ : Les appels directs à Open-Meteo entrent en conflit avec le calcul du champ corrigé
    # Les données globales doivent être récupérées via le serveur principal, pas directement ici
    # pour éviter les conflits avec le processus de fusion (Helmholtz) qui utilise aussi Open-Meteo
    logger.info("Récupération des données globales désactivée pour éviter les conflits avec le champ corrigé")
    return None


def detect_geographic_mismatch(question: str, station_data: Dict, forecast_data: Optional[Dict] = None) -> Optional[str]:
    """Détecte si la question concerne une zone géographique différente de celle couverte par les données"""
    question_lower = question.lower()
    
    # Pays et régions couverts par les données (autour de GP2 au Maroc)
    covered_regions = {
        "maroc", "morocco", "safi", "casablanca", "rabat", "marrakech",
        "afrique du nord", "north africa", "maghreb"
    }
    
    # Pays/régions souvent demandés mais non couverts
    other_regions = {
        "france", "paris", "lyon", "marseille", "toulouse", "nice", "bordeaux",
        "espagne", "spain", "madrid", "barcelone", "barcelona",
        "italie", "italy", "rome", "milan", "naples",
        "allemagne", "germany", "berlin", "munich", "hamburg",
        "angleterre", "england", "londres", "london", "royaume-uni", "uk",
        "belgique", "belgium", "bruxelles", "brussels",
        "suisse", "switzerland", "zurich", "genève", "geneva",
        "portugal", "lisbonne", "lisbon",
        "europe", "european", "européen"
    }
    
    # Vérifier si la question mentionne une région non couverte
    for region in other_regions:
        if region in question_lower:
            # Extraire la zone couverte par les données
            covered_area = "la région de Safi (Maroc)"
            if station_data and "station" in station_data:
                st = station_data.get("station", {})
                if st.get("name"):
                    covered_area = f"la station {st.get('name')} (Maroc)"
            
            return f"Les données disponibles concernent uniquement {covered_area} (position: {station_data.get('station', {}).get('lat', 'N/A')}°N, {station_data.get('station', {}).get('lon', 'N/A')}°E). Je ne peux pas fournir d'informations météorologiques pour {region.capitalize()} car ces données ne sont pas disponibles dans le contexte fourni."
    
    return None


class WeatherRAGProcessor:
    """Processeur RAG pour les données météorologiques - basé sur AdvancedRAGProcessor de llama.py"""
    
    @staticmethod
    def create_weather_documents(station_data: Dict, forecast_data: Optional[Dict] = None, global_data: Optional[Dict] = None) -> List[LangchainDocument]:
        """Crée des documents Langchain à partir des données météo (locale + globale si disponible)"""
        documents = []
        
        # Document 1: Données station GP2 temps réel
        if station_data and "station_data" in station_data:
            sd = station_data["station_data"]
            st = station_data.get("station", {})
            
            station_text = f"""DONNÉES STATION GP2 - TEMPS RÉEL

⚠️ IMPORTANT - ZONE GÉOGRAPHIQUE :
Ces données concernent UNIQUEMENT la station GP2 située au Maroc (région de Safi).
Cette station ne fournit PAS de données pour d'autres pays ou régions.

Station: {st.get('name', 'GP2')}
Position: {st.get('lat', 'N/A')}°N, {st.get('lon', 'N/A')}°E
Pays/Région: Maroc (région de Safi)
Timestamp: {station_data.get('station_timestamp', 'N/A')}

Mesures actuelles:
- Vitesse du vent: {sd.get('speed_ms', 'N/A')} m/s
- Direction du vent: {sd.get('dir_deg', 'N/A')}°
- Température de l'air: {sd.get('air_temp_c', 'N/A')}°C
- Humidité relative: {sd.get('rh', 'N/A')}%

Ces données représentent les mesures en temps réel de la station météorologique GP2 située au Maroc."""
            
            documents.append(LangchainDocument(
                page_content=station_text,
                metadata={
                    "source": "station_gp2_realtime",
                    "data_type": "station_realtime",
                    "timestamp": station_data.get('station_timestamp', ''),
                    "station_name": st.get('name', 'GP2')
                }
            ))
        
        # Document 2: Statistiques prévision à la position GP2
        if forecast_data and forecast_data.get("available", True):
            sp = forecast_data.get("station_position", {})
            
            # Statistiques de température
            if "temp_stats_at_station" in forecast_data:
                stats = forecast_data["temp_stats_at_station"]
                
                stats_text = f"""STATISTIQUES PRÉVISION OPEN-METEO - POSITION STATION GP2

⚠️ IMPORTANT - ZONE GÉOGRAPHIQUE :
Ces prévisions concernent UNIQUEMENT la position de la station GP2 au Maroc (région de Safi).
Ces données ne couvrent PAS d'autres pays ou régions comme la France, l'Espagne, etc.

Position station: {sp.get('lat', 'N/A')}°N, {sp.get('lon', 'N/A')}°E
Pays/Région: Maroc (région de Safi)
Modèle: {forecast_data.get('model', 'auto')}
Nombre de points de données: {stats.get('count', 'N/A')}

Températures prévues à la position exacte de la station GP2 (interpolées):
- Température moyenne: {stats.get('mean', 'N/A'):.2f}°C
- Température minimale: {stats.get('min', 'N/A'):.2f}°C
- Température maximale: {stats.get('max', 'N/A'):.2f}°C

Ces statistiques sont calculées à partir des prévisions Open-Meteo interpolées à la position exacte de la station GP2 au Maroc."""
                
                documents.append(LangchainDocument(
                    page_content=stats_text,
                    metadata={
                        "source": "forecast_stats_station",
                        "data_type": "forecast_statistics",
                        "model": forecast_data.get('model', 'auto'),
                        "stat_type": "temperature_at_station"
                    }
                ))
            
            # Statistiques d'humidité relative
            if "rh_stats_at_station" in forecast_data:
                rh_stats = forecast_data["rh_stats_at_station"]
                
                rh_stats_text = f"""STATISTIQUES HUMIDITÉ RELATIVE PRÉVISION OPEN-METEO - POSITION STATION GP2

⚠️ IMPORTANT - ZONE GÉOGRAPHIQUE :
Ces prévisions concernent UNIQUEMENT la position de la station GP2 au Maroc (région de Safi).

Position station: {sp.get('lat', 'N/A')}°N, {sp.get('lon', 'N/A')}°E
Pays/Région: Maroc (région de Safi)
Modèle: {forecast_data.get('model', 'auto')}
Nombre de points de données: {rh_stats.get('count', 'N/A')}

Humidité relative prévue à la position exacte de la station GP2 (interpolée):
- Humidité relative moyenne: {rh_stats.get('mean', 'N/A'):.2f}%
- Humidité relative minimale: {rh_stats.get('min', 'N/A'):.2f}%
- Humidité relative maximale: {rh_stats.get('max', 'N/A'):.2f}%

Ces statistiques sont calculées à partir des prévisions Open-Meteo interpolées à la position exacte de la station GP2 au Maroc."""
                
                documents.append(LangchainDocument(
                    page_content=rh_stats_text,
                    metadata={
                        "source": "forecast_rh_stats_station",
                        "data_type": "forecast_statistics",
                        "model": forecast_data.get('model', 'auto'),
                        "stat_type": "humidity_at_station"
                    }
                ))
            
            # Document 3: Série temporelle détaillée (température et humidité) - TOUTE LA SEMAINE
            if "forecast_timeseries" in forecast_data:
                forecast_series = forecast_data["forecast_timeseries"]
                timeseries_text = "SÉRIE TEMPORELLE PRÉVISION OPEN-METEO - STATION GP2 (TOUTE LA SEMAINE)\n\n"
                timeseries_text += "⚠️ IMPORTANT - ZONE GÉOGRAPHIQUE :\n"
                timeseries_text += "Ces prévisions concernent UNIQUEMENT la position de la station GP2 au Maroc (région de Safi).\n"
                timeseries_text += "Ces données ne couvrent PAS d'autres pays ou régions.\n\n"
                timeseries_text += f"Modèle: {forecast_data.get('model', 'auto')}\n"
                timeseries_text += f"Position station: {forecast_data.get('station_position', {}).get('lat', 'N/A')}°N, {forecast_data.get('station_position', {}).get('lon', 'N/A')}°E\n"
                timeseries_text += "Pays/Région: Maroc (région de Safi)\n"
                timeseries_text += f"Période couverte: TOUTE LA SEMAINE (7 jours) - {len(forecast_series)} échéances disponibles\n\n"
                timeseries_text += "Températures et humidité relative prévues à la position GP2 par échéance (TOUTE LA SEMAINE):\n\n"
                
                # Afficher toutes les échéances de la semaine (pas de limite)
                for fc in forecast_series:
                    h = fc.get("hour", 0)
                    temp_station = fc.get("temp_at_station")
                    temp_mean = fc.get("temp_mean")
                    rh_station = fc.get("rh_at_station")
                    rh_mean = fc.get("rh_mean")
                    
                    if temp_station is not None or rh_station is not None:
                        # Convertir les heures en jours pour plus de clarté
                        days = h // 24
                        hours_remainder = h % 24
                        if days > 0:
                            time_label = f"J+{days} ({h}h)"
                        else:
                            time_label = f"{h}h"
                        
                        timeseries_text += f"+{time_label}:"
                        if temp_station is not None:
                            timeseries_text += f" Temp: {temp_station:.2f}°C (position GP2)"
                            if temp_mean is not None:
                                timeseries_text += f" | Moyenne grille: {temp_mean:.2f}°C"
                        if rh_station is not None:
                            timeseries_text += f" | Humidité: {rh_station:.2f}% (position GP2)"
                            if rh_mean is not None:
                                timeseries_text += f" | Moyenne grille: {rh_mean:.2f}%"
                        timeseries_text += "\n"
                
                timeseries_text += f"\nCes données représentent les prévisions interpolées à la position exacte de la station GP2 au Maroc pour TOUTE LA SEMAINE ({len(forecast_series)} échéances)."
                timeseries_text += "\nLes prévisions couvrent 7 jours complets avec des données de température ET d'humidité relative à chaque échéance."
                
                documents.append(LangchainDocument(
                    page_content=timeseries_text,
                    metadata={
                        "source": "forecast_timeseries_station",
                        "data_type": "forecast_timeseries",
                        "model": forecast_data.get('model', 'auto'),
                        "hours_count": len(forecast_data["forecast_timeseries"])
                    }
                ))
            
            # Document 4: Statistiques sur toute la grille (température et humidité)
            grid = forecast_data.get("grid", {})
            grid_text = f"""STATISTIQUES PRÉVISION OPEN-METEO - GRILLE COMPLÈTE

⚠️ IMPORTANT - ZONE GÉOGRAPHIQUE COUVERTE :
Les données suivantes concernent UNIQUEMENT une petite zone géographique autour de la station GP2 au Maroc (région de Safi).
Cette zone ne couvre PAS d'autres pays ou régions comme la France, l'Espagne, l'Italie, etc.

Domaine géographique:
- Latitude: {grid.get('lat_min', 'N/A')}°N à {grid.get('lat_max', 'N/A')}°N
- Longitude: {grid.get('lon_min', 'N/A')}°E à {grid.get('lon_max', 'N/A')}°E
- Résolution grille: {grid.get('ny', 'N/A')}x{grid.get('nx', 'N/A')} points
- Modèle: {forecast_data.get('model', 'auto')}
- Pays/Région: Maroc (région de Safi)

Statistiques de température sur toute la grille:
"""
            
            if "temp_stats_grid" in forecast_data:
                stats_grid = forecast_data["temp_stats_grid"]
                grid_text += f"""- Température moyenne: {stats_grid.get('mean', 'N/A'):.2f}°C
- Température minimale: {stats_grid.get('min', 'N/A'):.2f}°C
- Température maximale: {stats_grid.get('max', 'N/A'):.2f}°C
"""
            
            if "rh_stats_grid" in forecast_data:
                rh_stats_grid = forecast_data["rh_stats_grid"]
                grid_text += f"""
Statistiques d'humidité relative sur toute la grille:
- Humidité relative moyenne: {rh_stats_grid.get('mean', 'N/A'):.2f}%
- Humidité relative minimale: {rh_stats_grid.get('min', 'N/A'):.2f}%
- Humidité relative maximale: {rh_stats_grid.get('max', 'N/A'):.2f}%
"""
            
            grid_text += "\nCes statistiques représentent les valeurs moyennes, minimales et maximales sur toute la zone géographique couverte par la grille (UNIQUEMENT la région de Safi, Maroc)."
            
            if "temp_stats_grid" in forecast_data or "rh_stats_grid" in forecast_data:
                documents.append(LangchainDocument(
                    page_content=grid_text,
                    metadata={
                        "source": "forecast_stats_grid",
                        "data_type": "forecast_statistics",
                        "model": forecast_data.get('model', 'auto'),
                        "stat_type": "temperature_and_humidity_grid"
                    }
                ))
        
        # Document 5: Données globales (si disponibles)
        if global_data and global_data.get("available", False):
            stats = global_data.get("stats", {})
            location = global_data.get("location", "Localité inconnue")
            
            global_text = f"""DONNÉES MÉTÉOROLOGIQUES GLOBALES - {location.upper()}

⚠️ IMPORTANT - SOURCE GLOBALE :
Ces données proviennent de l'API Open-Meteo et concernent {location}.
Ces données sont différentes des données locales de la station GP2 (Maroc).

Position: {global_data.get('lat', 'N/A')}°N, {global_data.get('lon', 'N/A')}°E
Localité: {location}
Modèle: {global_data.get('model', 'auto')}
Nombre de points de données: {stats.get('count', 'N/A')}

Statistiques de température pour {location} (sur 7 jours):
- Température moyenne: {stats.get('mean', 'N/A'):.2f}°C
- Température minimale: {stats.get('min', 'N/A'):.2f}°C
- Température maximale: {stats.get('max', 'N/A'):.2f}°C

Ces données représentent les prévisions Open-Meteo pour {location} et sont distinctes des données de la station GP2 au Maroc."""
            
            documents.append(LangchainDocument(
                page_content=global_text,
                metadata={
                    "source": "forecast_global",
                    "data_type": "forecast_global",
                    "location": location,
                    "model": global_data.get('model', 'auto'),
                    "stat_type": "temperature_global"
                    }
                ))
        
        # Document 6: Analyse spatiale de la grille temps réel (données visualisées sur l'interface) - OPTIMISÉ
        if station_data and "temp" in station_data and "grid" in station_data:
            try:
                # Conversion optimisée : utiliser directement les listes Python pour les stats simples
                temp_list = station_data["temp"]
                grid_info = station_data["grid"]
                
                # Calculer les statistiques de base sans conversion numpy complète (plus rapide)
                temp_flat = []
                for row in temp_list:
                    if isinstance(row, list):
                        temp_flat.extend([v for v in row if v is not None and not (isinstance(v, float) and np.isnan(v))])
                    else:
                        if row is not None and not (isinstance(row, float) and np.isnan(row)):
                            temp_flat.append(row)
                
                if len(temp_flat) > 0:
                    temp_max_val = float(max(temp_flat))
                    temp_min_val = float(min(temp_flat))
                    temp_mean_val = float(sum(temp_flat) / len(temp_flat))
                    
                    # Calcul simplifié des coordonnées approximatives (sans conversion complète de la grille)
                    # Approximation : centre de la grille pour les extrema
                    max_lat = (grid_info.get("lat_min", 0) + grid_info.get("lat_max", 0)) / 2
                    max_lon = (grid_info.get("lon_min", 0) + grid_info.get("lon_max", 0)) / 2
                    min_lat = max_lat
                    min_lon = max_lon
                    
                    spatial_text = f"""ANALYSE SPATIALE DE LA GRILLE TEMPS RÉEL - DONNÉES VISUALISÉES SUR L'INTERFACE

⚠️ IMPORTANT - DONNÉES VISUALISÉES SUR LA CARTE 2D :
Ces données correspondent EXACTEMENT à ce qui est affiché sur la carte 2D de l'interface.
La grille couvre la zone autour de la station GP2 au Maroc (région de Safi).
Ces valeurs sont celles que l'utilisateur voit lorsqu'il survole la carte avec sa souris.

⚠️ IMPORTANT - CHAMP FUSIONNÉ (OPEN METEO CORRIGÉ PAR LA STATION) :
Ces données ne sont PAS uniquement des données Open-Meteo brutes.
Ces données sont le RÉSULTAT DE LA FUSION entre :
1. Les données Open-Meteo (modèle météorologique)
2. Les mesures réelles de la station GP2

Le processus de fusion corrige les données Open-Meteo en utilisant les mesures de la station GP2
via une équation de Helmholtz, ce qui améliore la précision des données sur toute la grille.
Les valeurs affichées sont donc plus précises que les données Open-Meteo brutes car elles
sont calibrées avec les mesures réelles de la station.

Domaine géographique de la grille affichée sur la carte :
- Latitude: {grid_info.get('lat_min', 'N/A')}°N à {grid_info.get('lat_max', 'N/A')}°N
- Longitude: {grid_info.get('lon_min', 'N/A')}°E à {grid_info.get('lon_max', 'N/A')}°E
- Résolution: {grid_info.get('ny', 'N/A')}x{grid_info.get('nx', 'N/A')} points
- Nombre total de points sur la grille: {grid_info.get('ny', 0) * grid_info.get('nx', 0)}

ANALYSE DE TEMPÉRATURE SUR LA GRILLE VISUALISÉE (CHAMP FUSIONNÉ) :
- Température moyenne sur toute la grille affichée: {temp_mean_val:.2f}°C
- Température minimale sur la grille: {temp_min_val:.2f}°C
- Température maximale sur la grille: {temp_max_val:.2f}°C
- Amplitude thermique (différence max-min): {temp_max_val - temp_min_val:.2f}°C

ZONES EXTREMES VISIBLES SUR LA CARTE :
- Zone la plus chaude visible sur la carte: {temp_max_val:.2f}°C à la position {max_lat:.4f}°N, {max_lon:.4f}°E
- Zone la plus froide visible sur la carte: {temp_min_val:.2f}°C à la position {min_lat:.4f}°N, {min_lon:.4f}°E

Ces valeurs représentent les températures du CHAMP FUSIONNÉ affichées sur la carte 2D de l'interface.
L'utilisateur peut voir ces valeurs en survolant la carte avec sa souris."""
                    
                    # Analyse simplifiée de l'humidité (sans conversion numpy complète)
                    if "rh" in station_data:
                        rh_list = station_data["rh"]
                        rh_flat = []
                        for row in rh_list if isinstance(rh_list, list) else [rh_list]:
                            if isinstance(row, list):
                                rh_flat.extend([v for v in row if v is not None and not (isinstance(v, float) and np.isnan(v))])
                            else:
                                if row is not None and not (isinstance(row, float) and np.isnan(row)):
                                    rh_flat.append(row)
                        
                        if len(rh_flat) > 0:
                            rh_mean = float(sum(rh_flat) / len(rh_flat))
                            rh_min = float(min(rh_flat))
                            rh_max = float(max(rh_flat))
                            spatial_text += f"""

ANALYSE D'HUMIDITÉ RELATIVE SUR LA GRILLE VISUALISÉE (CHAMP FUSIONNÉ) :
- Humidité moyenne sur toute la grille affichée: {rh_mean:.2f}%
- Humidité minimale: {rh_min:.2f}%
- Humidité maximale: {rh_max:.2f}%"""
                    
                    # Analyse simplifiée du vent (sans conversion numpy complète)
                    if "u" in station_data and "v" in station_data:
                        u_list = station_data["u"]
                        v_list = station_data["v"]
                        speed_flat = []
                        u_rows = u_list if isinstance(u_list, list) else [u_list]
                        v_rows = v_list if isinstance(v_list, list) else [v_list]
                        for u_row, v_row in zip(u_rows, v_rows):
                            u_vals = u_row if isinstance(u_row, list) else [u_row]
                            v_vals = v_row if isinstance(v_row, list) else [v_row]
                            for u_val, v_val in zip(u_vals, v_vals):
                                if u_val is not None and v_val is not None:
                                    if not (isinstance(u_val, float) and np.isnan(u_val)) and not (isinstance(v_val, float) and np.isnan(v_val)):
                                        speed = float((u_val**2 + v_val**2)**0.5)  # Plus rapide que np.sqrt
                                        speed_flat.append(speed)
                        
                        if len(speed_flat) > 0:
                            speed_mean = float(sum(speed_flat) / len(speed_flat))
                            speed_max = float(max(speed_flat))
                            speed_min = float(min(speed_flat))
                            spatial_text += f"""

ANALYSE DU VENT SUR LA GRILLE VISUALISÉE (CHAMP FUSIONNÉ) :
- Vitesse moyenne du vent sur la grille affichée: {speed_mean:.2f} m/s ({speed_mean*3.6:.2f} km/h)
- Vitesse minimale: {speed_min:.2f} m/s ({speed_min*3.6:.2f} km/h)
- Vitesse maximale: {speed_max:.2f} m/s ({speed_max*3.6:.2f} km/h)"""
                    
                    documents.append(LangchainDocument(
                        page_content=spatial_text,
                        metadata={
                            "source": "grid_spatial_analysis",
                            "data_type": "spatial_analysis",
                            "grid_type": "realtime",
                            "visualization": "map_2d"
                        }
                    ))
            except Exception as e:
                logger.warning(f"Erreur lors de l'analyse spatiale de la grille: {e}")
        
        # Document 7: Analyse spatiale des prévisions (données visualisables)
        if forecast_data and forecast_data.get("available", True) and "forecast_timeseries" in forecast_data:
            try:
                forecast_series = forecast_data["forecast_timeseries"]
                if forecast_series:
                    all_temps_grid = [f.get("temp_mean") for f in forecast_series if f.get("temp_mean") is not None]
                    all_rh_grid = [f.get("rh_mean") for f in forecast_series if f.get("rh_mean") is not None]
                    
                    forecast_spatial_text = """ANALYSE SPATIALE DES PRÉVISIONS - DONNÉES VISUALISABLES SUR L'INTERFACE

⚠️ IMPORTANT - PRÉVISIONS VISUALISABLES :
Ces données correspondent aux prévisions qui peuvent être affichées sur l'interface en sélectionnant différentes échéances (0h, 24h, 48h, 72h, 96h, 120h, 144h, 168h).
L'utilisateur peut visualiser ces prévisions sur la carte 2D en changeant l'échéance.

"""
                    
                    if all_temps_grid:
                        max_temp_overall = max(all_temps_grid)
                        min_temp_overall = min(all_temps_grid)
                        
                        # Trouver l'échéance avec la température max/min
                        max_hour = None
                        min_hour = None
                        for f in forecast_series:
                            if f.get("temp_mean") == max_temp_overall:
                                max_hour = f.get("hour")
                            if f.get("temp_mean") == min_temp_overall:
                                min_hour = f.get("hour")
                        
                        forecast_spatial_text += f"""ANALYSE TEMPORELLE ET SPATIALE DES PRÉVISIONS - TEMPÉRATURE :
- Température moyenne maximale prévue (sur toute la période): {max_temp_overall:.2f}°C (à +{max_hour}h)
- Température moyenne minimale prévue (sur toute la période): {min_temp_overall:.2f}°C (à +{min_hour}h)
- Amplitude thermique prévue: {max_temp_overall - min_temp_overall:.2f}°C

"""
                    
                    if all_rh_grid:
                        max_rh_overall = max(all_rh_grid)
                        min_rh_overall = min(all_rh_grid)
                        
                        # Trouver l'échéance avec l'humidité max/min
                        max_rh_hour = None
                        min_rh_hour = None
                        for f in forecast_series:
                            if f.get("rh_mean") == max_rh_overall:
                                max_rh_hour = f.get("hour")
                            if f.get("rh_mean") == min_rh_overall:
                                min_rh_hour = f.get("hour")
                        
                        forecast_spatial_text += f"""ANALYSE TEMPORELLE ET SPATIALE DES PRÉVISIONS - HUMIDITÉ RELATIVE :
- Humidité relative moyenne maximale prévue (sur toute la période): {max_rh_overall:.2f}% (à +{max_rh_hour}h)
- Humidité relative moyenne minimale prévue (sur toute la période): {min_rh_overall:.2f}% (à +{min_rh_hour}h)
- Amplitude d'humidité prévue: {max_rh_overall - min_rh_overall:.2f}%

"""
                    
                    forecast_spatial_text += """Ces prévisions peuvent être visualisées sur l'interface en sélectionnant l'échéance correspondante.
L'utilisateur peut voir ces valeurs en survolant la carte avec sa souris après avoir sélectionné l'échéance."""
                    
                    if all_temps_grid or all_rh_grid:
                        documents.append(LangchainDocument(
                            page_content=forecast_spatial_text,
                            metadata={
                                "source": "forecast_spatial_analysis",
                                "data_type": "spatial_analysis",
                                "grid_type": "forecast",
                                "visualization": "map_2d_forecast"
                            }
                        ))
            except Exception as e:
                logger.warning(f"Erreur lors de l'analyse spatiale des prévisions: {e}")
        
        return documents
    
    @staticmethod
    def create_vector_store(documents: List[LangchainDocument], cached_embeddings=None):
        """Crée une base de données vectorielle FAISS avec embeddings (optimisé avec cache)"""
        if not documents:
            return None
        
        try:
            # Réutiliser les embeddings en cache si disponibles
            if cached_embeddings is None:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                )
            else:
                embeddings = cached_embeddings
            
            db = FAISS.from_documents(documents, embeddings)
            logger.info(f"Base vectorielle créée avec {len(documents)} documents")
            return db
        except Exception as e:
            logger.error(f"Erreur création base vectorielle: {e}")
            return None
    
    @staticmethod
    def keyword_search(db, query: str, k: int = 10) -> List[LangchainDocument]:
        """Recherche par mots-clés basée sur TF-IDF (identique à llama.py)"""
        all_docs = []
        if hasattr(db, 'docstore'):
            for doc_id in db.index_to_docstore_id.values():
                all_docs.append(db.docstore.search(doc_id))
        else:
            return []
        
        texts = [doc.page_content for doc in all_docs]
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=2000,
            ngram_range=(1, 2)
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            query_vec = vectorizer.transform([query])
            
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            sorted_indices = np.argsort(similarities)[::-1]
            
            return [all_docs[i] for i in sorted_indices[:k]]
        except:
            return []
    
    @staticmethod
    def deduplicate_documents(documents: List[LangchainDocument]) -> List[LangchainDocument]:
        """Déduplique les documents basés sur le contenu"""
        seen_contents = set()
        unique_docs = []
        
        for doc in documents:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    @staticmethod
    def rerank_documents(query: str, documents: List[LangchainDocument]) -> List[LangchainDocument]:
        """Re-rank les documents par pertinence avec CrossEncoder (optimisé : skip si peu de documents)"""
        # Si peu de documents, skip le re-ranking pour gagner du temps
        if len(documents) <= 3:
            return documents
        
        try:
            cross_encoder = CrossEncoder(RERANKER_MODEL)
            
            pairs = [(query, doc.page_content) for doc in documents]
            scores = cross_encoder.predict(pairs)
            
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in scored_docs]
        except Exception as e:
            logger.warning(f"Re-ranking non disponible: {str(e)}")
            return documents
    
    @staticmethod
    def hybrid_retrieval(db, query: str, k: int = 10) -> List[LangchainDocument]:
        """Recherche hybride: sémantique uniquement (optimisé pour performance maximale)"""
        if db is None:
            return []
        
        # Recherche sémantique uniquement (plus rapide que hybride)
        semantic_results = db.similarity_search(query, k=k)
        
        # Pas de recherche TF-IDF ni de re-ranking pour maximiser la vitesse
        return semantic_results[:k]
    
    @staticmethod
    def validate_retrieval_quality(query: str, retrieved_docs: List[LangchainDocument]) -> List[LangchainDocument]:
        """Filtre les documents non pertinents"""
        relevant_docs = []
        query_keywords = set(query.lower().split())
        
        for doc in retrieved_docs:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_keywords & doc_words)
            relevance_score = overlap / len(query_keywords) if query_keywords else 0
            
            if relevance_score > 0.05:
                relevant_docs.append(doc)
        
        return relevant_docs if relevant_docs else retrieved_docs


class WindyChatbot:
    """Chatbot spécialisé pour répondre aux questions sur les données météo GP2 et Open Meteo avec RAG"""
    
    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.current_conversation_id = None
        self.messages = []
        self._data_hash = None  # Cache pour éviter de reconstruire la base vectorielle
        self._cached_embeddings = None  # Cache des embeddings
        if CEREBRAS_AVAILABLE:
            try:
                self.llm = ChatCerebras(
                    model="llama3.1-8b",
                    temperature=0.7,
                    max_tokens=TOKEN_LIMITS["max_tokens_per_request"]
                )
            except Exception as e:
                logger.error(f"Erreur initialisation Cerebras: {e}")
                self.llm = None
        
        # Initialiser avec un message de bienvenue
        self.messages = [
            {
                "role": "assistant",
                "content": "Bonjour ! Je suis votre assistant météorologique. Posez-moi des questions sur les données de la station GP2 ou les prévisions Open-Meteo."
            }
        ]
    
    def update_data(self, station_data: Dict, forecast_data: Optional[Dict] = None, global_data: Optional[Dict] = None):
        """Met à jour les données et reconstruit la base vectorielle seulement si nécessaire"""
        # Calculer un hash des données pour éviter de reconstruire inutilement
        data_str = json.dumps({
            "station": station_data.get("station_timestamp", "") if station_data else None,
            "forecast": forecast_data.get("hour", "") if forecast_data else None,
            "global": global_data.get("location", "") if global_data else None
        }, sort_keys=True, default=str)
        current_hash = hashlib.md5(data_str.encode()).hexdigest()
        
        # Si les données n'ont pas changé, ne pas reconstruire la base vectorielle
        if current_hash == self._data_hash and self.vector_store is not None:
            logger.debug("Données inchangées, utilisation du cache de la base vectorielle")
            return
        
        # Créer les documents à partir des données météo (locale + globale)
        documents = WeatherRAGProcessor.create_weather_documents(station_data, forecast_data, global_data)
        
        # Créer/mettre à jour la base vectorielle
        if documents:
            self.vector_store = WeatherRAGProcessor.create_vector_store(documents, cached_embeddings=self._cached_embeddings)
            # Mettre en cache les embeddings pour la prochaine fois
            if self.vector_store:
                # Les embeddings sont réutilisés via le paramètre cached_embeddings
                # On garde une référence pour la prochaine fois
                if self._cached_embeddings is None:
                    # Créer les embeddings une seule fois
                    from langchain_huggingface import HuggingFaceEmbeddings
                    self._cached_embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-mpnet-base-v2"
                    )
            self._data_hash = current_hash
        else:
            self.vector_store = None
            self._data_hash = None
    
    def get_rag_prompt(self, context: str, question: str) -> str:
        """Retourne le prompt RAG optimisé avec vérification géographique"""
        return """Tu es un expert analyste météorologique qui doit répondre de manière CLAIRE, SYNTHÉTIQUE et COMPLÈTE.

CONTEXTE FOURNI :
{context}

⚠️ RÈGLE GÉOGRAPHIQUE CRITIQUE :
Les données disponibles peuvent inclure :
1. Des données locales de la station GP2 (Maroc, région de Safi)
2. Des données globales pour d'autres pays/régions spécifiques (si mentionnées dans la question)

⚠️ DONNÉES TEMPS RÉEL - CHAMP FUSIONNÉ :
Les données temps réel (affichées sur la carte 2D) ne sont PAS uniquement des données Open-Meteo.
Ce sont des données FUSIONNÉES qui combinent :
- Les données Open-Meteo (modèle météorologique)
- Les mesures réelles de la station GP2

Le processus de fusion corrige les données Open-Meteo en utilisant les mesures de la station GP2
via une équation de Helmholtz, ce qui améliore la précision des données sur toute la grille.
Les valeurs affichées sont donc plus précises que les données Open-Meteo brutes car elles
sont calibrées avec les mesures réelles de la station.

⚠️ PRÉVISIONS DE LA SEMAINE - TEMPÉRATURE ET HUMIDITÉ :
Les données de prévision couvrent TOUTE LA SEMAINE (7 jours) avec des échéances à 0h, 24h, 48h, 72h, 96h, 120h, 144h, 168h.
Chaque échéance contient :
- Température interpolée à la position exacte de la station GP2
- Température moyenne sur toute la grille
- Humidité relative interpolée à la position exacte de la station GP2
- Humidité relative moyenne sur toute la grille

Tu peux répondre aux questions sur :
- Les prévisions de température et d'humidité pour n'importe quelle échéance de la semaine
- Les statistiques (moyenne, min, max) sur toute la semaine
- Les tendances et évolutions de température et d'humidité au cours de la semaine
- Les comparaisons entre différentes échéances

⚠️ DONNÉES VISUALISÉES SUR L'INTERFACE :
Les données disponibles correspondent EXACTEMENT à ce qui est affiché sur la carte 2D et le globe 3D de l'interface.
Tu peux répondre aux questions sur :
- Les valeurs à n'importe quel point de la grille affichée sur la carte (champ fusionné)
- Les zones les plus chaudes/froides VISIBLES sur la carte (pas du globe entier)
- Les gradients et variations spatiales sur la zone visualisée
- Les comparaisons entre différentes zones de la grille affichée
- Les prévisions à différentes échéances visualisables sur l'interface
- Les zones les plus humides/sèches, les plus ventées sur la carte affichée

IMPORTANT - QUESTIONS GLOBALES :
Si la question concerne une analyse globale/mondiale (comme "zone la plus chaude du globe", "température mondiale", etc.),
réponds IMMÉDIATEMENT et CLAIREMENT que ces données ne sont pas disponibles car elles nécessitent une couverture mondiale complète.
NE JAMAIS essayer de répondre avec les données locales de GP2 pour ce type de questions.

IMPORTANT - QUESTIONS SUR LA ZONE VISUALISÉE :
Si la question concerne "la zone la plus chaude" ou "la zone la plus froide" SANS mentionner "du globe" ou "mondiale",
alors la question porte probablement sur la zone VISUALISÉE sur la carte (région de Safi).
Dans ce cas, utilise les données d'analyse spatiale de la grille pour répondre avec les coordonnées et valeurs exactes.

Si la question mentionne un pays, une région ou une ville différente (comme la France, l'Espagne, Paris, etc.), 
utilise les données globales correspondantes si elles sont disponibles dans le contexte.
Si des données globales sont présentes, elles sont clairement identifiées avec leur localité.
NE JAMAIS utiliser les données de GP2 (Maroc) pour répondre à une question sur un autre pays ou région, 
SAUF si des données globales pour ce pays/région sont présentes dans le contexte.

RÈGLES STRICTES POUR LA RÉPONSE :
1. SYNTHÉTISE : Réponds de manière claire et concise, sans répétitions inutiles
2. COMPLÈTE : Inclus toutes les informations pertinentes du contexte, mais de manière organisée
3. STRUCTURÉE : Utilise des paragraphes courts et des listes à puces si nécessaire pour la clarté
4. PRÉCISE : Utilise les valeurs numériques EXACTES du contexte - ne les modifie pas
5. CONTEXTUELLE : Mentionne la source des données (station GP2, prévisions Open-Meteo) et la zone géographique quand c'est pertinent
6. DIRECTE : Va droit au but, évite les phrases trop longues ou les explications superflues
7. Si l'information n'est pas dans le contexte, dis clairement "Cette information n'est pas disponible dans les données fournies"
8. Si plusieurs sources contiennent des informations complémentaires, intègre-les toutes de manière synthétique
9. Utilise les statistiques pré-calculées (moyennes, min, max) directement - ne les recalcule pas
10. NE TRONQUE JAMAIS ta réponse - fournis une réponse complète même si elle est longue

QUESTION : {question}

RÉPONSE CLAIRE ET SYNTHÉTIQUE BASÉE SUR LE CONTEXTE (sans troncature) :""".format(context=context, question=question)
    
    def generate_response(self, question: str, station_data: Dict, forecast_data: Optional[Dict] = None) -> str:
        """Génère une réponse à une question en utilisant RAG hybride"""
        
        # Vérifier si c'est une question globale qui nécessite des données mondiales complètes
        # Mais permettre les questions sur "la zone la plus chaude" de la région visualisée
        question_lower = question.lower()
        is_about_globe = any(phrase in question_lower for phrase in [
            "du globe", "du monde", "mondiale", "globale", "planétaire", 
            "partout", "tous les pays", "tous les continents", "toute la planète"
        ])
        
        if is_global_question(question) and is_about_globe:
            return "Désolé, je ne peux pas répondre à cette question car elle nécessite des données météorologiques mondiales complètes que je n'ai pas à ma disposition. Les données disponibles concernent uniquement la station GP2 (Maroc, région de Safi) et quelques pays/régions spécifiques sur demande. Pour des questions sur des zones géographiques spécifiques, je peux vous aider si vous mentionnez le pays ou la ville concernée."
        
        # Détecter si la question concerne un autre pays/région et récupérer les données globales
        # Si l'API globale échoue, on continue avec les données locales
        global_data = None
        try:
            global_data = detect_and_fetch_global_data(question)
        except Exception as e:
            logger.warning(f"Erreur lors de la détection/récupération des données globales: {e}. Continuation avec les données locales.")
            global_data = None
        
        # Mettre à jour les données et la base vectorielle (locale + globale si disponible)
        try:
            self.update_data(station_data, forecast_data, global_data)
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des données: {e}", exc_info=True)
            # Essayer sans les données globales
            try:
                self.update_data(station_data, forecast_data, None)
            except Exception as e2:
                logger.error(f"Erreur critique lors de la mise à jour des données: {e2}", exc_info=True)
                return "Désolé, une erreur technique s'est produite lors du traitement de votre question. Veuillez réessayer."
        
        # Si pas de base vectorielle, utiliser le mode dégradé
        if self.vector_store is None:
            return self._fallback_response(question, station_data, forecast_data)
        
        # Récupération sémantique uniquement (optimisé pour vitesse maximale)
        try:
            retrieved_docs = WeatherRAGProcessor.hybrid_retrieval(self.vector_store, question, k=3)  # Réduit à 3 documents
            
            # Validation de la pertinence
            relevant_docs = WeatherRAGProcessor.validate_retrieval_quality(question, retrieved_docs)
            
            if not relevant_docs:
                return "Désolé, je n'ai pas trouvé d'informations pertinentes dans les données disponibles pour répondre à cette question."
            
            # Construire le contexte à partir des documents récupérés
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                source_info = f"Source: {doc.metadata.get('source', 'Données météo')}"
                if 'data_type' in doc.metadata:
                    source_info += f" | Type: {doc.metadata['data_type']}"
                context_parts.append(f"[Document {i+1}] {source_info}\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Créer le prompt RAG
            full_prompt = self.get_rag_prompt(context, question)
            
            # Générer la réponse avec le LLM en utilisant safe_cerebras_call
            if self.llm:
                try:
                    response_text = safe_cerebras_call(self.llm, full_prompt)
                    
                    # Valider la réponse
                    response_text = self.validate_response(response_text, station_data, forecast_data)
                    return response_text
                except Exception as e:
                    logger.error(f"Erreur appel Cerebras: {e}")
                    return self._fallback_response(question, station_data, forecast_data)
            else:
                return self._fallback_response(question, station_data, forecast_data)
                
        except Exception as e:
            logger.error(f"Erreur RAG: {e}")
            return self._fallback_response(question, station_data, forecast_data)
    
    def validate_response(self, response: str, station_data: Dict, forecast_data: Optional[Dict] = None) -> str:
        """Valide et corrige la réponse pour éviter les hallucinations"""
        response_lower = response.lower()
        
        # Vérifier si la réponse mentionne des prévisions non disponibles alors qu'elles le sont
        if forecast_data and forecast_data.get("available", True):
            if any(phrase in response_lower for phrase in [
                "prévisions ne sont pas disponibles",
                "prévisions ne sont pas accessibles",
                "données de prévision ne sont pas disponibles",
                "open-meteo ne sont pas disponibles"
            ]):
                response = response.replace(
                    "les prévisions ne sont pas disponibles",
                    "les prévisions sont disponibles"
                ).replace(
                    "prévisions ne sont pas disponibles",
                    "prévisions sont disponibles"
                )
                logger.warning("Correction appliquée: prévisions mentionnées comme non disponibles")
        
        # Vérifier si la réponse mentionne des données pour un pays/région différent
        # mais utilise quand même les données de GP2 (hallucination géographique)
        other_countries = ["france", "paris", "lyon", "espagne", "spain", "italie", "italy", 
                          "allemagne", "germany", "angleterre", "england", "europe"]
        mentions_other_country = any(country in response_lower for country in other_countries)
        mentions_gp2 = "gp2" in response_lower or "maroc" in response_lower or "safi" in response_lower
        
        if mentions_other_country and mentions_gp2:
            # La réponse mélange des pays différents avec GP2 - c'est suspect
            # Vérifier si c'est juste pour expliquer la différence ou si c'est une hallucination
            if not any(phrase in response_lower for phrase in [
                "ne sont pas disponibles", "non disponible", "pas disponible pour",
                "uniquement", "seulement", "concernent uniquement"
            ]):
                logger.warning("Réponse suspecte: mélange de zones géographiques détecté")
                # Ne pas modifier automatiquement, mais logger l'avertissement
        
        # Ne pas tronquer la réponse - laisser le LLM générer une réponse complète
        return response
    
    def _fallback_response(self, question: str, station_data: Dict, forecast_data: Optional[Dict] = None) -> str:
        """Réponse de secours sans LLM"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["vent", "vitesse", "speed"]):
            if station_data and "station_data" in station_data:
                speed = station_data["station_data"].get("speed_ms", "N/A")
                dir_deg = station_data["station_data"].get("dir_deg", "N/A")
                return f"Selon les données de la station GP2, la vitesse du vent est de {speed} m/s et la direction est de {dir_deg}°."
            return "Les données de vent ne sont pas disponibles actuellement."
        
        elif any(word in question_lower for word in ["température", "temp", "chaud", "froid", "moyenne"]):
            if forecast_data and forecast_data.get("available", True) and "temp_stats_at_station" in forecast_data:
                stats = forecast_data["temp_stats_at_station"]
                return f"Selon les prévisions Open-Meteo interpolées à la position de la station GP2, la température moyenne est de {stats.get('mean', 'N/A'):.2f}°C, avec un minimum de {stats.get('min', 'N/A'):.2f}°C et un maximum de {stats.get('max', 'N/A'):.2f}°C."
            elif station_data and "station_data" in station_data:
                temp = station_data["station_data"].get("air_temp_c", "N/A")
                return f"La température de l'air mesurée par la station GP2 est de {temp}°C."
            return "Les données de température ne sont pas disponibles actuellement."
        
        elif any(word in question_lower for word in ["humidité", "rh", "humidity"]):
            if forecast_data and forecast_data.get("available", True) and "rh_stats_at_station" in forecast_data:
                rh_stats = forecast_data["rh_stats_at_station"]
                return f"Selon les prévisions Open-Meteo interpolées à la position de la station GP2, l'humidité relative moyenne est de {rh_stats.get('mean', 'N/A'):.2f}%, avec un minimum de {rh_stats.get('min', 'N/A'):.2f}% et un maximum de {rh_stats.get('max', 'N/A'):.2f}%."
            elif station_data and "station_data" in station_data:
                rh = station_data["station_data"].get("rh", "N/A")
                return f"L'humidité relative mesurée par la station GP2 est de {rh}%."
            return "Les données d'humidité ne sont pas disponibles actuellement."
        
        else:
            return f"Je peux vous aider à analyser les données météorologiques. Posez-moi une question plus spécifique sur le vent, la température, l'humidité ou les prévisions."
    
    def add_message(self, role: str, content: str):
        """Ajoute un message à la conversation"""
        self.messages.append({"role": role, "content": content})
    
    def get_messages(self) -> List[Dict]:
        """Retourne tous les messages de la conversation"""
        return self.messages
    
    def save_conversation(self, title: Optional[str] = None) -> Optional[str]:
        """Sauvegarde la conversation actuelle"""
        if self.current_conversation_id:
            return update_existing_conversation(self.current_conversation_id, self.messages)
        else:
            conv_id = save_conversation(self.messages, title)
            if conv_id:
                self.current_conversation_id = conv_id
            return conv_id
    
    def load_conversation(self, conversation_id: str) -> bool:
        """Charge une conversation"""
        conv_data = load_conversation(conversation_id)
        if conv_data:
            self.messages = conv_data.get("messages", [])
            self.current_conversation_id = conversation_id
            return True
        return False
    
    def new_conversation(self):
        """Démarre une nouvelle conversation"""
        # Sauvegarder l'ancienne conversation si elle existe
        if self.messages and len(self.messages) > 1:
            self.save_conversation()
        
        self.messages = [
            {
                "role": "assistant",
                "content": "Bonjour ! Je suis votre assistant météorologique. Posez-moi des questions sur les données de la station GP2 ou les prévisions Open-Meteo."
            }
        ]
        self.current_conversation_id = None
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Retourne le résumé d'utilisation Cerebras"""
        return cerebras_token_manager.get_usage_summary()


# Instance globale du chatbot
_chatbot_instance = None

def get_chatbot() -> WindyChatbot:
    """Retourne l'instance singleton du chatbot"""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = WindyChatbot()
    return _chatbot_instance
