import streamlit as st
import os
import tempfile
from typing import List, Dict, Any, Tuple
import pandas as pd
import pypdf
import openpyxl
import re
import time
import json
from datetime import datetime, timedelta
import uuid
import statistics
from dataclasses import dataclass, asdict
from langchain_cerebras import ChatCerebras
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document as LangchainDocument
from sentence_transformers import CrossEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration de la page
st.set_page_config(
    page_title="Assistant IA Multilingue - Cerebras Llama 3.1-8B",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dossier pour sauvegarder les conversations
CONVERSATIONS_DIR = "conversations"
if not os.path.exists(CONVERSATIONS_DIR):
    os.makedirs(CONVERSATIONS_DIR)

# Configuration de l'API Cerebras
CEREBRAS_API_KEY = "csk-mnv698mvc25hmmnvhrcmfj2mvmf3v5nrrvxtwrn8m5r848jy"
os.environ["CEREBRAS_API_KEY"] = CEREBRAS_API_KEY

# Configuration des limites de tokens pour Llama 3.1-8B sur Cerebras
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

# Configuration RAG avancée
RAG_CONFIG = {
    "general": {
        "chunk_size": 1200,  # Augmenté de 1000 à 1200 (+20%)
        "chunk_overlap": 300,  # Augmenté de 200 à 300 (+50%)
        "separators": ["\n\n", "\n", " ", ""]
    },
    "math_heavy": {
        "chunk_size": 960,  # Augmenté de 800 à 960 (+20%)
        "chunk_overlap": 225,  # Augmenté de 150 à 225 (+50%)
        "separators": ["\n\n", "\n$$", "\n$", " ", ""]
    },
    "code_heavy": {
        "chunk_size": 720,  # Augmenté de 600 à 720 (+20%)
        "chunk_overlap": 150,  # Augmenté de 100 à 150 (+50%)
        "separators": ["\n\n", "\n\`\`\`", "\nclass", "\ndef", "\nimport", " "]
    }
}

# Modèle de re-ranking (cross-encoder)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@dataclass
class UsageStats:
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
        check_result = self.can_make_request(estimated_tokens)
        
        if not check_result["can_proceed"]:
            wait_time = check_result["wait_time"]
            limit_type = check_result["blocking_limit"]
            
            limit_names = {
                "requests_per_minute": "requêtes par minute (192)",
                "tokens_per_minute": "tokens par minute (14,400)",
                "requests_per_hour": "requêtes par heure (30)",
                "tokens_per_hour": "tokens par heure (60,000)",
                "requests_per_day": "requêtes par jour (900)",
                "tokens_per_day": "tokens par jour (1,000,000)"
            }
            
            st.warning(f"⏳ Limite Cerebras atteinte: {limit_names.get(limit_type, limit_type)}")
            st.info(f"⏰ Attente de {wait_time:.1f} secondes...")
            
            progress_bar = st.progress(0)
            for i in range(int(wait_time)):
                progress_bar.progress((i + 1) / wait_time)
                time.sleep(1)
            
            progress_bar.empty()
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

class AdvancedRAGProcessor:
    """Processeur RAG avancé avec chunking intelligent et retrieval amélioré"""
    
    @staticmethod
    def detect_content_type(text: str) -> str:
        """Détecte le type de contenu pour optimiser le chunking"""
        math_indicators = len(re.findall(r'\$.*?\$|\\[a-zA-Z]+', text))
        code_indicators = len(re.findall(r'\`\`\`|def |class |import ', text))
        
        if math_indicators > 10:
            return "math_heavy"
        elif code_indicators > 5:
            return "code_heavy"
        else:
            return "general"
    
    @staticmethod
    def extract_section_title(text: str) -> str:
        """Extrait le titre de section d'un texte"""
        section_patterns = [
            r'#+\s+(.+)$',  # Markdown headers
            r'^[A-Z][A-Z\s]{5,}$',  # All caps titles
            r'^[0-9]+\.\s+(.+)$',  # Numbered sections
        ]
        
        lines = text.split('\n')
        for line in lines[:5]:  # Regarder les premières lignes
            for pattern in section_patterns:
                match = re.search(pattern, line.strip())
                if match:
                    title = match.group(1) if len(match.groups()) > 0 else line.strip()
                    return title[:50]  # Limiter la longueur
        
        return "Section sans titre"
    
    @staticmethod
    def smart_chunking(text: str, filename: str, content_type: str) -> List[LangchainDocument]:
        """Chunking intelligent selon le type de contenu avec plus de contexte"""
        config = RAG_CONFIG[content_type]
        
        # Augmenter légèrement la taille des chunks pour plus de contexte
        adjusted_config = config.copy()
        adjusted_config["chunk_size"] = int(config["chunk_size"] * 1.2)  # +20%
        adjusted_config["chunk_overlap"] = int(config["chunk_overlap"] * 1.5)  # +50%
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=adjusted_config["chunk_size"],
            chunk_overlap=adjusted_config["chunk_overlap"],
            separators=config["separators"],
            keep_separator=True
        )
        
        # Diviser le texte
        chunks = text_splitter.split_text(text)
        
        # Créer des documents avec métadonnées enrichies
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": filename,
                "file_type": filename.split('.')[-1],
                "content_type": content_type,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "has_math": AdvancedRAGProcessor.contains_latex(chunk),
                "language": detect_language_with_cerebras(chunk[:500]),
                "section": AdvancedRAGProcessor.extract_section_title(chunk),
                "previous_section": AdvancedRAGProcessor.extract_section_title(chunks[i-1]) if i > 0 else "",
                "next_section": AdvancedRAGProcessor.extract_section_title(chunks[i+1]) if i < len(chunks)-1 else ""
            }
            documents.append(LangchainDocument(page_content=chunk, metadata=metadata))
        
        return documents
    
    @staticmethod
    def contains_latex(text: str) -> bool:
        """Vérifie si un texte contient du LaTeX"""
        latex_patterns = [
            r'\$.*?\$',  # Equations inline
            r'\\[a-zA-Z]+\{',  # Commandes LaTeX
            r'\\begin\{',  # Environnements LaTeX
            r'\\frac\{', r'\\int', r'\\sum', r'\\prod'  # Commandes mathématiques
        ]
        
        for pattern in latex_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    @staticmethod
    def hybrid_retrieval(db, query: str, k: int = 15) -> List[LangchainDocument]:
        """Recherche hybride: sémantique + mots-clés avec plus de résultats"""
        # Recherche sémantique - augmenter le nombre de résultats
        semantic_results = db.similarity_search(query, k=k*3)  # De k*2 à k*3
        
        # Recherche par mots-clés (TF-IDF) - augmenter le nombre de résultats
        keyword_results = AdvancedRAGProcessor.keyword_search(db, query, k=k*2)  # De k à k*2
        
        # Combiner et dédupliquer
        all_results = semantic_results + keyword_results
        unique_results = AdvancedRAGProcessor.deduplicate_documents(all_results)
        
        # Re-ranking avec plus de documents
        reranked_results = AdvancedRAGProcessor.rerank_documents(query, unique_results)
        
        return reranked_results[:k]
    
    @staticmethod
    def keyword_search(db, query: str, k: int = 10) -> List[LangchainDocument]:
        """Recherche par mots-clés basée sur TF-IDF avec améliorations"""
        # Extraire tous les documents
        all_docs = []
        if hasattr(db, 'docstore'):
            for doc_id in db.index_to_docstore_id.values():
                all_docs.append(db.docstore.search(doc_id))
        else:
            # Fallback: utiliser les documents disponibles
            return []
        
        # Préparer les textes pour TF-IDF
        texts = [doc.page_content for doc in all_docs]
        
        # Calculer TF-IDF avec des paramètres améliorés
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=2000,  # Augmenté de 1000 à 2000
            ngram_range=(1, 2)  # Ajouter les bigrammes
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            query_vec = vectorizer.transform([query])
            
            # Calculer les similarités
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            # Trier par similarité
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Retourner les top-k résultats
            return [all_docs[i] for i in sorted_indices[:k]]
        except:
            return []
    
    @staticmethod
    def deduplicate_documents(documents: List[LangchainDocument]) -> List[LangchainDocument]:
        """Déduplique les documents basés sur le contenu"""
        seen_contents = set()
        unique_docs = []
        
        for doc in documents:
            content_hash = hash(doc.page_content[:200])  # Hash du début du contenu
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    @staticmethod
    def rerank_documents(query: str, documents: List[LangchainDocument]) -> List[LangchainDocument]:
        """Re-rank les documents par pertinence"""
        try:
            # Initialiser le cross-encoder pour le re-ranking
            cross_encoder = CrossEncoder(RERANKER_MODEL)
            
            # Préparer les paires (query, document)
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Calculer les scores de pertinence
            scores = cross_encoder.predict(pairs)
            
            # Trier les documents par score
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in scored_docs]
        except Exception as e:
            st.warning(f"Re-ranking non disponible: {str(e)}")
            return documents
    
    @staticmethod
    def validate_retrieval_quality(query: str, retrieved_docs: List[LangchainDocument]) -> List[LangchainDocument]:
        """Filtre les documents non pertinents avec un seuil moins strict"""
        relevant_docs = []
        query_keywords = set(query.lower().split())
        
        for doc in retrieved_docs:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_keywords & doc_words)
            relevance_score = overlap / len(query_keywords) if query_keywords else 0
            
            # Réduire le seuil de pertinence de 0.1 à 0.05
            if relevance_score > 0.05:  # Seuil de pertinence réduit
                relevant_docs.append(doc)
        
        return relevant_docs if relevant_docs else retrieved_docs  # Fallback si tout est filtré

def chunk_text(text: str, chunk_size: int = TOKEN_LIMITS["chunk_size"]) -> List[str]:
    """Divise un texte en chunks optimisés pour Llama 3.1-8B"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) // 3
        if current_size + word_size > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def safe_cerebras_call(llm, prompt: str, max_retries: int = TOKEN_LIMITS["max_retries"]) -> str:
    """Appel sécurisé au LLM Cerebras avec gestion complète des erreurs"""
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
                st.warning(f"⚠ Limite Cerebras atteinte. Attente de {wait_time} secondes... (Tentative {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif "token" in error_msg and "limit" in error_msg:
                st.error("❌ Limite de tokens dépassée. Essayez avec un texte plus court.")
                break
            else:
                if attempt == max_retries - 1:
                    st.error(f"❌ Erreur Cerebras après {max_retries} tentatives: {str(e)}")
                    break
                else:
                    st.warning(f"⚠ Erreur Cerebras (tentative {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(5)
    
    return "Désolé, je n'ai pas pu traiter votre demande en raison de limitations techniques avec Cerebras."

def process_large_text_with_chunking(llm, text: str, system_prompt: str) -> str:
    """Traite un texte volumineux en le divisant en chunks pour Cerebras"""
    chunks = chunk_text(text)
    
    if len(chunks) == 1:
        full_prompt = f"{system_prompt}\n\nTexte: {text}"
        return safe_cerebras_call(llm, full_prompt)
    
    st.info(f"📊 Texte volumineux détecté. Division en {len(chunks)} parties pour Cerebras Llama 3.1-8B...")
    
    responses = []
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(chunks):
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
        st.info(f"📄 Traitement de la partie {i + 1}/{len(chunks)} avec Cerebras...")
        
        chunk_prompt = f"{system_prompt}\n\nPartie {i + 1}/{len(chunks)} du texte:\n{chunk}"
        response = safe_cerebras_call(llm, chunk_prompt)
        responses.append(f"*Partie {i + 1}:*\n{response}")
        
        if i < len(chunks) - 1:
            time.sleep(TOKEN_LIMITS["delay_between_requests"])
    
    progress_bar.progress(1.0)
    st.success(f"✅ Traitement terminé avec Cerebras! {len(chunks)} parties traitées.")
    
    return "\n\n---\n\n".join(responses)

def detect_language_with_cerebras(text: str) -> str:
    """Détecte la langue d'un texte en utilisant Cerebras Llama 3.1-8B"""
    if not text.strip():
        return "fr"
    
    llm = ChatCerebras(
        model="llama3.1-8b",
        temperature=0.1,
        max_tokens=50
    )
    
    prompt = f"""Détectez la langue du texte suivant et répondez UNIQUEMENT par le code de langue (fr, en, es, ar, de, it):

Texte: "{text[:200]}"

Code de langue:"""
    
    try:
        response = safe_cerebras_call(llm, prompt).strip().lower()
        valid_languages = ["fr", "en", "es", "ar", "de", "it"]
        if response in valid_languages:
            return response
        
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
    except Exception:
        return "fr"

def get_language_name(lang_code: str) -> str:
    """Retourne le nom complet de la langue"""
    languages = {
        "fr": "Français",
        "en": "English", 
        "es": "Español",
        "ar": "العربية",
        "de": "Deutsch",
        "it": "Italiano"
    }
    return languages.get(lang_code, "Français")

def get_language_prompt(lang_code: str) -> str:
    """Retourne le prompt système selon la langue détectée pour Llama 3.1-8B"""
    prompts = {
        "fr": "Tu es un assistant IA serviable utilisant Llama 3.1-8B sur Cerebras. Réponds toujours en français de manière claire et détaillée. Si ta réponse contient des mathématiques, utilise la notation LaTeX: équations inline avec $...$ et équations display avec $$...$$",
        "en": "You are a helpful AI assistant using Llama 3.1-8B on Cerebras. Always respond in English in a clear and detailed manner. If your response contains mathematics, use LaTeX notation: inline equations with $...$ and display equations with $$...$$",
        "es": "Eres un asistente de IA útil usando Llama 3.1-8B en Cerebras. Siempre responde en español de manera clara y detallada. Si tu respuesta contiene matemáticas, usa notación LaTeX: ecuaciones inline con $...$ y ecuaciones display con $$...$$",
        "ar": "أنت مساعد ذكي مفيد يستخدم Llama 3.1-8B على Cerebras. أجب دائماً باللغة العربية بطريقة واضحة ومفصلة. إذا كانت إجابتك تحتوي على رياضيات، استخدم تدوين LaTeX",
        "de": "Du bist ein hilfreicher KI-Assistent mit Llama 3.1-8B auf Cerebras. Antworte siempre auf Deutsch in klarer und detaillierter Weise. Wenn deine Antwort Mathematik enthält, verwende LaTeX-Notation",
        "it": "Sei un assistente IA útil que usa Llama 3.1-8B su Cerebras. Rispondi siempre in italiano in modo claro e dettagliato. Se la tua risposta contiene matematica, usa la notazione LaTeX"
    }
    return prompts.get(lang_code, prompts["fr"])

def get_rag_prompt(lang_code: str) -> str:
    """Retourne le prompt RAG optimisé pour des explications détaillées"""
    prompts = {
        "fr": """Tu es un expert analyste qui doit extraire toutes les informations pertinentes du contexte fourni.

CONTEXTE FOURNI :
{context}

RÈGLES STRICTES :
1. Analyse minutieusement TOUT le contexte fourni
2. Extrait toutes les informations pertinentes pour la question, même si elles sont dispersées
3. Si l'information n'est pas dans le contexte, dis clairement "Cette information n'est pas disponible dans les documents fournis"
4. Pour chaque information importante, fournis une explication détaillée
5. Donne des exemples concrets quand c'est possible
6. Structure ta réponse de manière claire avec des sections si nécessaire
7. Utilise la notation LaTeX pour toutes les formules mathématiques
8. Mentionne la source du document et la section quand c'est pertinent
9. Explique le "pourquoi" et le "comment" derrière chaque concept
10. Si plusieurs documents contiennent des informations complémentaires, intègre-les toutes

QUESTION : {question}

RÉPONSE EXHAUSTIVE BASÉE SUR LE CONTEXTE :""",
        
        "en": """You are an expert analyst who must extract all relevant information from the provided context.

PROVIDED CONTEXT:
{context}

STRICT RULES:
1. Thoroughly analyze ALL provided context
2. Extract all relevant information for the question, even if scattered
3. If the information is not in the context, clearly state "This information is not available in the provided documents"
4. For each important information, provide a detailed explanation
5. Give concrete examples when possible
6. Structure your response clearly with sections if needed
7. Use LaTeX notation for all mathematical formulas
8. Mention the document source and section when relevant
9. Explain the "why" and "how" behind each concept
10. If multiple documents contain complementary information, integrate them all

QUESTION: {question}

COMPREHENSIVE CONTEXT-BASED RESPONSE:"""
    }
    return prompts.get(lang_code, prompts["fr"])

def should_start_new_conversation(message: str) -> bool:
    """Détermine si l'utilisateur demande explicitement une nouvelle conversation"""
    new_conversation_keywords = [
        "nouvelle conversation", "nouveau chat", "reset", "effacer", 
        "recommencer", "clear", "new conversation", "start over"
    ]
    
    message_lower = message.lower().strip()
    return any(keyword in message_lower for keyword in new_conversation_keywords)

def auto_save_conversation(messages, mode):
    """Sauvegarde automatique après chaque réponse - mise à jour de la conversation existante"""
    if len(messages) >= 2:
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if len(user_messages) > 0:
            current_conv_id = None
            if mode == "customer_service":
                current_conv_id = st.session_state.get('current_cs_conversation_id')
            else:
                current_conv_id = st.session_state.get('current_general_conversation_id')
            
            if current_conv_id:
                return update_existing_conversation(current_conv_id, messages, mode)
            else:
                return save_conversation(messages, mode)
    return None

def update_existing_conversation(conversation_id, messages, mode):
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
        st.error(f"Erreur lors de la mise à jour: {e}")
        return save_conversation(messages, mode)

def create_new_conversation(mode):
    """Crée une nouvelle conversation"""
    if mode == "customer_service":
        st.session_state.cs_messages = [
            {"role": "assistant", "content": "Bonjour! Je suis votre assistant de service client alimenté par Cerebras Llama 3.1-8B avec rendu LaTeX intégré. Téléchargez vos documents et posez-moi des questions à leur sujet. Si ma réponse contient des mathématiques, elles seront automatiquement rendues!"}
        ]
        st.session_state.current_cs_conversation_id = None
    else:
        st.session_state.general_messages = [
            {"role": "assistant", "content": "Bonjour! Je suis votre assistant IA alimenté par Cerebras Llama 3.1-8B avec rendu LaTeX intégré. Posez-moi n'importe quelle question dans votre langue préférée! Si ma réponse contient des mathématiques, elles seront automatiquement rendues avec une belle présentation."}
        ]
        st.session_state.current_general_conversation_id = None

def rename_conversation(conversation_id, new_title):
    """Renomme une conversation"""
    filename = f"{CONVERSATIONS_DIR}/conv_{conversation_id}.json"
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            conv_data = json.load(f)
        conv_data['title'] = new_title
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conv_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Erreur lors du renommage: {e}")
        return False

def get_conversation_preview(messages):
    """Génère un aperçu de la conversation"""
    if not messages:
        return "Conversation vide"
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    if user_messages:
        first_message = user_messages[0]["content"]
        return first_message[:40] + "..." if len(first_message) > 40 else first_message
    return "Nouvelle conversation"

def show_save_notification(title, message_count, file_size):
    """Affiche une notification toast élégante à droite de l'écran"""
    st.markdown(f"""
    <div id="save-notification" class="save-notification">
        <div class="notification-content">
            <div class="notification-icon">✅</div>
            <div class="notification-text">
                <div class="notification-title">Conversation sauvegardée!</div>
                <div class="notification-details">{title[:30]}...</div>
                <div class="notification-meta">{message_count} messages • {file_size} bytes</div>
            </div>
        </div>
    </div>
    <script>
    setTimeout(function() {{
        const notification = document.getElementById('save-notification');
        if (notification) {{
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(function() {{
                notification.remove();
            }}, 300);
        }}
    }}, 3000);
    </script>
    """, unsafe_allow_html=True)

def generate_smart_title(first_message):
    """Génère un titre intelligent pour la conversation en utilisant Cerebras Llama 3.1-8B"""
    try:
        title_prompt = f"""Tu es un expert en création de titres courts et descriptifs pour des conversations de chat.

TÂCHE: Créer un titre court (3-6 mots maximum) qui résume parfaitement le sujet de cette question.

QUESTION DE L'UTILISATEUR: {first_message[:300]}

RÈGLES STRICTES:
1. Maximum 6 mots
2. Pas de guillemets, apostrophes, ou caractères spéciaux , pas de hashtag
3. Pas de ponctuation à la fin
4. Utilise des mots simples et clairs
5. Commence par un verbe d'action ou un nom
6. Évite les articles (le, la, les, un, une)

EXEMPLES DE BONS TITRES:
- Question: "Comment puis-je configurer mon API ?" → Titre: "Configuration API"
- Question: "J'ai un problème avec Python" → Titre: "Problème Python"
- Question: "Peux-tu m'aider à créer une base de données ?" → Titre: "Création base données"
- Question: "Comment faire du machine learning ?" → Titre: "Machine Learning guide"

RÉPONSE (seulement le titre, rien d'autre):"""

        llm = ChatCerebras(
            model="gpt-oss-120b",
            temperature=0.1,
            max_tokens=1000
        )
        response = safe_cerebras_call(llm, title_prompt)
        title = response.strip()
        
        title = title.replace('"', '').replace("'", '').replace('`', '')
        title = title.replace('«', '').replace('»', '').replace('"', '').replace('"', '')
        title = title.replace(':', '').replace(';', '').replace('!', '').replace('?', '')
        title = title.replace('\n', ' ').replace('\r', ' ').replace('#','')
        title = ' '.join(title.split())
        
        if len(title) > 40:
            words = title.split()
            title = ' '.join(words[:5])
        
        if not title or len(title) < 3:
            words = first_message.split()[:10]
            important_words = [w for w in words if len(w) > 3 and w.lower() not in ['comment', 'puis', 'peux', 'veux', 'faire', 'avec', 'pour', 'dans', 'cette', 'cette']]
            title = ' '.join(important_words[:3]) if important_words else first_message[:25]
        
        return title.strip()
    except Exception as e:
        st.error(f"Erreur lors de la génération du titre: {e}")
        words = first_message.split()[:5]
        return ' '.join(words).replace('?', '').replace('!', '')[:30]

def save_conversation(messages, mode, title=None):
    """Sauvegarde une conversation dans un fichier JSON avec notification toast"""
    try:
        if not messages or len(messages) == 0:
            st.error("❌ Aucun message à sauvegarder")
            return None
        
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if len(user_messages) == 0:
            st.error("❌ Aucun message utilisateur trouvé")
            return None
        
        try:
            if not os.path.exists(CONVERSATIONS_DIR):
                os.makedirs(CONVERSATIONS_DIR)
        except Exception as e:
            st.error(f"❌ Impossible de créer le dossier: {e}")
            return None
        
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        if not title:
            first_user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
            if first_user_message:
                title = generate_smart_title(first_user_message)
            else:
                title = f"Conversation {mode}"
        
        conversation_data = {
            "id": conversation_id,
            "title": title,
            "mode": mode,
            "timestamp": timestamp.isoformat(),
            "messages": messages,
            "message_count": len(messages)
        }
        
        filename = f"{CONVERSATIONS_DIR}/conv_{conversation_id}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                return conversation_id
            else:
                st.error("❌ Le fichier n'a pas été créé")
                return None
        except PermissionError:
            st.error("❌ Erreur de permissions - impossible d'écrire le fichier")
            return None
        except Exception as e:
            st.error(f"❌ Erreur lors de l'écriture: {e}")
            return None
    except Exception as e:
        st.error(f"❌ Erreur générale lors de la sauvegarde: {e}")
        return None

def load_conversations():
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
                st.error(f"Erreur lors du chargement de {filename}: {e}")
    
    conversations.sort(key=lambda x: x["timestamp"], reverse=True)
    return conversations

def load_conversation(conversation_id):
    """Charge une conversation spécifique"""
    filename = f"{CONVERSATIONS_DIR}/conv_{conversation_id}.json"
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Erreur lors du chargement de la conversation: {e}")
        return None

def delete_conversation(conversation_id):
    """Supprime une conversation"""
    filename = f"{CONVERSATIONS_DIR}/conv_{conversation_id}.json"
    try:
        os.remove(filename)
        return True
    except Exception as e:
        st.error(f"Erreur lors de la suppression: {e}")
        return False

def format_conversation_date(timestamp_str):
    """Formate la date pour l'affichage"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%d/%m/%Y %H:%M")
    except:
        return timestamp_str

def display_conversation_item(conv):
    """Affiche un élément de conversation dans la sidebar"""
    col1, col2, col3 = st.columns([7, 1.2, 1.2])
    
    with col1:
        if st.button(
            conv['title'][:35] + "..." if len(conv['title']) > 35 else conv['title'],
            key=f"load_{conv['id']}",
            help=f"{conv['message_count']} messages • {format_conversation_date(conv['timestamp'])}",
            use_container_width=True
        ):
            load_conversation_to_session(conv)
    
    with col2:
        if st.button("✏", key=f"rename_{conv['id']}", help="Renommer", use_container_width=True):
            st.session_state[f"renaming_{conv['id']}"] = True
    
    with col3:
        if st.button("🗑", key=f"delete_{conv['id']}", help="Supprimer", use_container_width=True):
            if delete_conversation(conv['id']):
                st.success("Supprimé!")
                st.rerun()
    
    if st.session_state.get(f"renaming_{conv['id']}", False):
        new_title = st.text_input(
            "Nouveau titre:",
            value=conv['title'],
            key=f"new_title_{conv['id']}"
        )
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("💾", key=f"save_rename_{conv['id']}"):
                if rename_conversation(conv['id'], new_title):
                    st.success("Renommé!")
                    st.session_state[f"renaming_{conv['id']}"] = False
                    st.rerun()
        with col_cancel:
            if st.button("❌", key=f"cancel_rename_{conv['id']}"):
                st.session_state[f"renaming_{conv['id']}"] = False
                st.rerun()

def load_conversation_to_session(conv):
    """Charge une conversation dans la session"""
    loaded_conv = load_conversation(conv['id'])
    if loaded_conv:
        if conv['mode'] == "customer_service":
            st.session_state.cs_messages = loaded_conv['messages']
            st.session_state.current_cs_conversation_id = conv['id']
        else:
            st.session_state.general_messages = loaded_conv['messages']
            st.session_state.current_general_conversation_id = conv['id']
        st.success(f"Conversation '{conv['title'][:30]}...' chargée!")
        st.rerun()

def add_latex_css():
    """Ajoute le CSS et JavaScript pour le rendu LaTeX amélioré"""
    st.markdown("""
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
    window.MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\$$', '\$$']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']],
            processEscapes: true,
            processEnvironments: true,
            macros: {
                RR: "\\mathbb{R}",
                NN: "\\mathbb{N}",
                ZZ: "\\mathbb{Z}",
                QQ: "\\mathbb{Q}",
                CC: "\\mathbb{C}",
                min: "\\operatorname{min}",
                max: "\\operatorname{max}",
                argmin: "\\operatorname{argmin}",
                argmax: "\\operatorname{argmax}",
                grad: "\\nabla",
                div: "\\nabla\\cdot",
                curl: "\\nabla\\times"
            }
        },
        options: {
            ignoreHtmlClass: "tex2jax_ignore",
            processHtmlClass: "tex2jax_process"
        },
        startup: {
            ready: () => {
                MathJax.startup.defaultReady();
                console.log('MathJax is loaded and ready!');
            }
        }
    };
    
    function renderMathJax() {
        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise().catch((err) => console.log(err.message));
        }
    }
    
    const observer = new MutationObserver(function(mutations) {
        let shouldRender = false;
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                for (let node of mutation.addedNodes) {
                    if (node.nodeType === 1 && (node.textContent.includes('$') || node.textContent.includes('\\('))) {
                        shouldRender = true;
                        break;
                    }
                }
            }
        });
        if (shouldRender) {
            setTimeout(renderMathJax, 100);
        }
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    </script>
    
    <style>
    .math-container {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f4fd 100%);
        border: 2px solid #e1e8ed;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        position: relative;
        overflow-x: auto;
    }
    
    .math-container::before {
        content: "📐 Formule Mathématique";
        position: absolute;
        top: -10px;
        left: 15px;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .math-inline {
        background: rgba(102, 126, 234, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .math-error {
        background: #ffe6e6;
        border: 2px solid #ff9999;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        color: #cc0000;
        font-family: monospace;
    }
    
    .math-fallback {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 6px;
        padding: 8px;
        margin: 5px 0;
        font-family: 'Courier New', monospace;
        color: #856404;
    }
    
    .math-rendering {
        opacity: 0.6;
        transition: opacity 0.3s ease;
    }
    
    .math-rendered {
        opacity: 1;
        transition: opacity 0.3s ease;
    }
    
    .MathJax {
        font-size: 1.1em !important;
    }
    
    .MathJax_Display {
        margin: 1em 0 !important;
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Styles CSS modernes
st.markdown("""
<style>
/* Styles CSS modernes */
.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 30%, #16213e 70%, #0f3460 100%);
    color: #ffffff;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    padding: 3rem;
    border-radius: 25px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 25px 50px rgba(102, 126, 234, 0.4);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    position: relative;
    overflow: hidden;
}

.mode-selector {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 2.5rem;
    border-radius: 25px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(79, 172, 254, 0.3);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    position: relative;
}

.chat-container {
    background: linear-gradient(135deg, #2d3748 0%, #4a5568 50%, #2d3748 100%);
    padding: 2.5rem;
    border-radius: 25px;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 20px 40px rgba(45, 55, 72, 0.4);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.feature-card {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 2.5rem;
    border-radius: 20px;
    box-shadow: 0 15px 35px rgba(30, 60, 114, 0.4);
    border: 1px solid rgba(102, 126, 234, 0.3);
    margin-bottom: 2rem;
    color: #e2e8f0;
    backdrop-filter: blur(15px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 25px 50px rgba(30, 60, 114, 0.6);
}

.token-info {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.processing-status {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 15px 30px rgba(17, 153, 142, 0.4);
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 30px;
    padding: 1rem 3rem;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease;
    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 40px rgba(102, 126, 234, 0.6);
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

.language-indicator {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1rem 2rem;
    border-radius: 30px;
    color: white;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px rgba(79, 172, 254, 0.4);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.stChatMessage {
    background: linear-gradient(135deg, rgba(30, 60, 114, 0.9) 0%, rgba(42, 82, 152, 0.9) 100%) !important;
    border-radius: 25px !important;
    padding: 2rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 15px 35px rgba(0,0,0,0.4) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    transition: transform 0.2s ease !important;
}

.stChatMessage:hover {
    transform: translateY(-2px) !important;
}

.stChatMessage[data-testid="chat-message-user"] {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

.stChatMessage[data-testid="chat-message-assistant"] {
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.9) 0%, rgba(0, 242, 254, 0.9) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

.chatbot-signature {
    text-align: right;
    margin-top: 15px;
    padding-top: 15px;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
}

.stChatMessage .stMarkdown {
    color: #ffffff !important;
    font-size: 16px !important;
    line-height: 1.8 !important;
}

.stChatMessage .stMarkdown p {
    color: #ffffff !important;
    margin-bottom: 1rem !important;
}

.stChatMessage .stMarkdown h1, .stChatMessage .stMarkdown h2, .stChatMessage .stMarkdown h3, .stChatMessage .stMarkdown h4, .stChatMessage .stMarkdown h5, .stChatMessage .stMarkdown h6 {
    color: #f093fb !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    margin-bottom: 1rem !important;
}

.stChatInputContainer {
    background: linear-gradient(135deg, rgba(30, 60, 114, 0.9) 0%, rgba(42, 82, 152, 0.9) 100%) !important;
    border-radius: 30px !important;
    border: 2px solid rgba(102, 126, 234, 0.5) !important;
    padding: 1rem !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 15px 30px rgba(0,0,0,0.3) !important;
}

.stChatInputContainer input {
    color: #ffffff !important;
    font-size: 16px !important;
    background: transparent !important;
    font-weight: 500 !important;
}

.stChatInputContainer input::placeholder {
    color: #a0aec0 !important;
    font-style: italic;
}

.css-1d391kg {
    background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    border-right: 1px solid rgba(102, 126, 234, 0.3);
}

.stSidebar .stMarkdown {
    color: #e2e8f0;
}

.stSidebar .stSelectbox label, .stSidebar .stFileUploader label, .stSidebar .stSlider label {
    color: #e2e8f0 !important;
    font-weight: 600;
}

.stSelectbox > div > div {
    background: rgba(30, 60, 114, 0.8) !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(102, 126, 234, 0.4) !important;
    border-radius: 15px !important;
    color: #ffffff !important;
}

.stFileUploader > div {
    background: rgba(30, 60, 114, 0.8) !important;
    backdrop-filter: blur(15px) !important;
    border: 2px dashed rgba(79, 172, 254, 0.6) !important;
    border-radius: 20px !important;
    transition: border-color 0.3s ease;
}

.stFileUploader > div:hover {
    border-color: rgba(79, 172, 254, 0.8) !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    border-radius: 10px !important;
}

.save-notification {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 20px;
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    animation: slideInRight 0.5s ease-out;
    max-width: 380px;
    min-width: 320px;
}

.notification-content {
    display: flex;
    align-items: center;
    padding: 20px;
    color: white;
}

.notification-icon {
    font-size: 28px;
    margin-right: 15px;
    animation: bounce 0.6s ease-in-out;
}

.notification-text {
    flex: 1;
}

.notification-title {
    font-weight: 700;
    font-size: 17px;
    margin-bottom: 5px;
}

.notification-details {
    font-size: 14px;
    opacity: 0.9;
    margin-bottom: 3px;
}

.notification-meta {
    font-size: 12px;
    opacity: 0.7;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(40px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% {
        transform: translateY(0);
    }
    40% {
        transform: translateY(-12px);
    }
    60% {
        transform: translateY(-6px);
    }
}

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(30, 60, 114, 0.5);
    border-radius: 15px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    border: 2px solid rgba(30, 60, 114, 0.5);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

@media (max-width: 768px) {
    .save-notification {
        right: 10px;
        left: 10px;
        max-width: none;
        min-width: auto;
    }
    
    .main-header {
        padding: 2rem;
    }
    
    .mode-selector, .chat-container, .feature-card {
        padding: 1.5rem;
    }
}

.glass-effect {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
    box-shadow: 0 15px 35px rgba(79, 172, 254, 0.4) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
    box-shadow: 0 20px 45px rgba(79, 172, 254, 0.6) !important;
}

.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100") !important;
    opacity: 0.8;
}

.stButton > button[kind="secondary"]:hover {
    opacity: 1;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

class DocumentProcessor:
    """Classe pour traiter différents types de documents"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extrait le texte d'un fichier PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = []
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
                return "\n".join(text)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du PDF: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extrait le texte d'un fichier Word"""
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            
            with zipfile.ZipFile(file_path, 'r') as docx:
                content = docx.read('word/document.xml')
                root = ET.fromstring(content)
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                
                text = []
                for paragraph in root.findall('.//w:p', ns):
                    para_text = []
                    for text_elem in paragraph.findall('.//w:t', ns):
                        if text_elem.text:
                            para_text.append(text_elem.text)
                    if para_text:
                        text.append(''.join(para_text))
                
                return "\n".join(text)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier Word: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_pptx(file_path: str) -> str:
        """Extrait le texte d'un fichier PowerPoint"""
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            
            with zipfile.ZipFile(file_path, 'r') as pptx:
                text = []
                slide_files = [f for f in pptx.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
                
                for slide_file in slide_files:
                    content = pptx.read(slide_file)
                    root = ET.fromstring(content)
                    ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
                    
                    for text_elem in root.findall('.//a:t', ns):
                        if text_elem.text:
                            text.append(text_elem.text)
                
                return "\n".join(text)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier PowerPoint: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_excel(file_path: str) -> str:
        """Extrait le texte d'un fichier Excel"""
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            text = []
            for sheet_name, sheet_df in df.items():
                text.append(f"=== Feuille: {sheet_name} ===")
                text.append(sheet_df.to_string(index=False))
                text.append("")
            return "\n".join(text)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier Excel: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extrait le texte d'un fichier texte"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier texte: {str(e)}")
                return ""
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier texte: {str(e)}")
            return ""

def process_uploaded_files(uploaded_files) -> List[LangchainDocument]:
    """Traite les fichiers téléchargés et retourne une liste de documents"""
    documents = []
    processor = DocumentProcessor()
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                text = processor.extract_text_from_pdf(tmp_file_path)
            elif file_extension in ['docx', 'doc']:
                text = processor.extract_text_from_docx(tmp_file_path)
            elif file_extension in ['pptx', 'ppt']:
                text = processor.extract_text_from_pptx(tmp_file_path)
            elif file_extension in ['xlsx', 'xls']:
                text = processor.extract_text_from_excel(tmp_file_path)
            elif file_extension == 'txt':
                text = processor.extract_text_from_txt(tmp_file_path)
            else:
                st.warning(f"Type de fichier non supporté: {file_extension}")
                continue
            
            if text.strip():
                # Détection du type de contenu
                content_type = AdvancedRAGProcessor.detect_content_type(text)
                
                # Chunking intelligent
                chunked_docs = AdvancedRAGProcessor.smart_chunking(text, uploaded_file.name, content_type)
                documents.extend(chunked_docs)
                
        finally:
            os.unlink(tmp_file_path)
    
    return documents

def create_vector_store_with_chunking(documents: List[LangchainDocument]):
    """Crée une base de données vectorielle avec chunking intelligent"""
    if not documents:
        return None
    
    # Utiliser un embedding plus performant
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Créer la base vectorielle
    st.info("📚 Création de la base de données vectorielle avec embeddings avancés...")
    db = FAISS.from_documents(documents, embeddings)
    st.success("✅ Base de données vectorielle créée avec succès!")
    
    return db

def safe_qa_call(db, query: str, language_prompt: str) -> str:
    """Appel sécurisé à la chaîne QA avec RAG avancé"""
    try:
        # Récupération avancée des documents - augmenter le nombre de documents
        retrieved_docs = AdvancedRAGProcessor.hybrid_retrieval(db, query, k=10)  # De 5 à 10
        
        # Validation de la pertinence avec seuil réduit
        relevant_docs = AdvancedRAGProcessor.validate_retrieval_quality(query, retrieved_docs)
        
        if not relevant_docs:
            return "Désolé, je n'ai pas trouvé d'informations pertinentes dans vos documents pour répondre à cette question."
        
        # Construire le contexte à partir des documents avec plus de métadonnées
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            source_info = f"\nSource: {doc.metadata.get('source', 'Document')}"
            if 'section' in doc.metadata:
                source_info += f" - Section: {doc.metadata['section']}"
            if 'content_type' in doc.metadata:
                source_info += f" - Type: {doc.metadata['content_type']}"
            context_parts.append(f"[Document {i+1}]{source_info}\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Créer le prompt RAG optimisé
        rag_prompt = get_rag_prompt(detect_language_with_cerebras(query))
        full_prompt = rag_prompt.format(context=context, question=query)
        
        # Utiliser safe_cerebras_call avec gestion des tokens
        llm = ChatCerebras(
            model="gpt-oss-120b",
            temperature=0.3,
            max_tokens=TOKEN_LIMITS["max_tokens_per_request"]
        )
        
        response = safe_cerebras_call(llm, full_prompt)
        return response
        
    except Exception as e:
        st.error(f"Erreur lors de la requête QA avec Cerebras: {str(e)}")
        return "Désolé, je n'ai pas pu traiter votre demande en raison d'une erreur technique avec Cerebras."

def initialize_qa_chain_with_limits(db):
    """Initialise la chaîne de question-réponse avec gestion des limites Cerebras"""
    llm = ChatCerebras(
        model="gpt-oss-120b",
        temperature=0.3,
        max_tokens=TOKEN_LIMITS["max_tokens_per_request"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

def initialize_general_llm_with_limits():
    """Initialise le LLM pour le chat général avec gestion des limites Cerebras"""
    return ChatCerebras(
        model="gpt-oss-120b",
        temperature=0.7,
        max_tokens=TOKEN_LIMITS["max_tokens_per_request"]
    )

def contains_latex(text: str) -> bool:
    """Détecte si un texte contient du code LaTeX"""
    latex_indicators = [
        r'\$.*?\$',
        r'\$\$.*?\$\$',
        r'\\[a-zA-Z]+',
        r'\\begin\{',
        r'\\frac\{',
        r'\\int',
        r'\\sum',
        r'\\prod',
        r'\\sqrt',
        r'\^[{]',
        r'_[{]',
    ]
    
    for pattern in latex_indicators:
        if re.search(pattern, text):
            return True
    return False

def extract_latex_equations(text: str):
    """Extrait les équations LaTeX d'un texte"""
    equations = []
    
    # Équations display ($$...$$)
    display_pattern = r'\$\$(.*?)\$\$'
    for match in re.finditer(display_pattern, text, re.DOTALL):
        equations.append({
            'type': 'display',
            'content': match.group(1).strip(),
            'full': match.group(0),
            'start': match.start(),
            'end': match.end()
        })
    
    # Équations inline ($...$) - éviter les doubles $
    text_without_display = re.sub(display_pattern, '', text, flags=re.DOTALL)
    inline_pattern = r'\$([^$]+)\$'
    for match in re.finditer(inline_pattern, text_without_display):
        equations.append({
            'type': 'inline',
            'content': match.group(1).strip(),
            'full': f"${match.group(1)}$",
            'start': match.start(),
            'end': match.end()
        })
    
    return equations

def render_latex_content(text: str) -> str:
    """Rendu LaTeX simplifié et efficace"""
    if not text:
        return ""
    
    try:
        # Remplacer les crochets par des dollars pour les expressions mathématiques
        text = re.sub(r'\[\s*(\\begin\{aligned\}.*?\\end\{aligned\})\s*\]', r'$$\1$$', text, flags=re.DOTALL)
        text = re.sub(r'\[\s*([^\[\]]+)\s*\]', r'$$\1$$', text)
        
        # Corriger les accolades dans \in{...} vers \in \{...\}
        text = re.sub(r'\\in\{([^}]+)\}', r'\\in \{\1\}', text)
        
        # Détecter et traiter le LaTeX
        has_display_math = '$$' in text
        has_inline_math = bool(re.search(r'(?<!\$)\$[^$]+\$(?!\$)', text))
        
        if has_display_math or has_inline_math:
            def clean_display_math(match):
                formula = match.group(1)
                return f'<div class="math-container">$$\\displaystyle {formula}$$</div>'
            
            def clean_inline_math(match):
                formula = match.group(1)
                return f'<span class="math-inline">${formula}$</span>'
            
            text = re.sub(r'\$\$([^$]+?)\$\$', clean_display_math, text, flags=re.DOTALL)
            text = re.sub(r'(?<!\$)\$([^$]+?)\$(?!\$)', clean_inline_math, text)
            
            return f'<div class="latex-content">{text}</div>'
        
        return text
        
    except Exception as e:
        return text

def display_ai_response_with_latex(response: str, token_manager):
    """Affiche la réponse de l'IA avec rendu LaTeX simplifié"""
    try:
        if not response:
            st.warning("Réponse vide reçée")
            return
        
        tokens_used = token_manager.count_tokens(response)
        rendered_content = render_latex_content(response)
        
        st.markdown(rendered_content, unsafe_allow_html=True)
        
        st.markdown("""
        <script>
        setTimeout(function() {
            if (window.MathJax && window.MathJax.typesetPromise) {
                window.MathJax.typesetPromise().catch((err) => console.log('MathJax error:', err.message));
            }
        }, 100);
        </script>
        """, unsafe_allow_html=True)
        
        with st.expander("ℹ️ Informations sur la réponse Cerebras", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tokens utilisés", tokens_used)
            with col2:
                st.metric("Longueur", len(response))
        
    except Exception as e:
        st.error(f"Erreur lors de l'affichage: {str(e)}")
        st.text(response)

def get_cerebras_usage():
    """Récupère les statistiques d'utilisation de Cerebras"""
    try:
        usage = cerebras_token_manager.get_usage_summary()
        return usage
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données d'utilisation: {e}")
        return None

def display_cerebras_usage_dashboard():
    """Affiche le tableau de bord d'utilisation de Cerebras"""
    st.subheader("📊 Utilisation Cerebras")
    
    usage = get_cerebras_usage()
    if not usage:
        st.error("Impossible de récupérer les données d'utilisation")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Requêtes/minute", 
            usage["minute"]["requests"],
            delta=f"{usage['minute']['requests_percent']:.1f}% utilisé"
        )
    with col2:
        st.metric(
            "Tokens/minute", 
            usage["minute"]["tokens"],
            delta=f"{usage['minute']['tokens_percent']:.1f}% utilisé"
        )
    
    st.progress(usage["minute"]["requests_percent"] / 100)
    st.progress(usage["minute"]["tokens_percent"] / 100)
    
    with st.expander("📈 Statistiques détaillées"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Req/heure", usage["hour"]["requests"])
        with col2:
            st.metric("Tokens/heure", usage["hour"]["tokens"])
        with col3:
            st.metric("Req/jour", usage["day"]["requests"])
        with col4:
            st.metric("Tokens/jour", usage["day"]["tokens"])

def main():
    add_latex_css()
    
    # En-tête principal
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Assistant IA Multilingue - Cerebras Llama 3.1-8B</h1>
        <p>Votre assistant intelligent avec Cerebras et rendu mathématique parfait</p>
        <p><strong>🚀 Modèle:</strong> Llama 3.1-8B | <strong>⚡ Plateforme:</strong> Cerebras</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard d'utilisation Cerebras
    display_cerebras_usage_dashboard()
    
    with st.sidebar:
        st.header("🧪 Test LaTeX avec Cerebras")
        if st.button("Tester le rendu LaTeX"):
            test_formulas = [
                "Formule simple: $f(x) = x^2$",
                "Optimisation: $$\\min_{x \\in \\mathbb{R}^n} f(x) \\text{ sous } g_i(x) \\leq 0$$",
                "Gradient: $$\\nabla f(x^*) = 0$$",
                "Matrice: $$\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$$",
                "Intégrale: $$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$"
            ]
            
            st.markdown("### Exemples de rendu LaTeX avec Cerebras:")
            for formula in test_formulas:
                rendered = render_latex_content(formula)
                st.markdown(rendered, unsafe_allow_html=True)
    
    # Sélecteur de mode
    st.markdown("""
    <div class="mode-selector">
        <h3 style="color: white; text-align: center; margin-bottom: 1rem;">🎯 Choisissez votre mode Cerebras</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📚 Service Client", help="Chat basé sur vos documents avec Cerebras Llama 3.1-8B", use_container_width=True):
            st.session_state.chat_mode = "customer_service"
    
    with col2:
        if st.button("💬 Chat Général", help="Conversation libre avec Cerebras Llama 3.1-8B", use_container_width=True):
            st.session_state.chat_mode = "general_chat"
    
    # Initialiser le mode par défaut
    if 'chat_mode' not in st.session_state:
        st.session_state.chat_mode = "customer_service"
    
    # Affichage du mode actuel
    mode_text = "📚 Service Client" if st.session_state.chat_mode == "customer_service" else "💬 Chat Général"
    st.markdown(f"""
    <div class="chat-container">
        <h3>Mode actuel: {mode_text} (Cerebras Llama 3.1-8B avec rendu LaTeX)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Interface selon le mode
    if st.session_state.chat_mode == "customer_service":
        # Mode Service Client avec Cerebras
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); border-radius: 10px; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0;">🤖 Assistant Cerebras</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("➕ Nouvelle Conversation", use_container_width=True, type="primary"):
                create_new_conversation(st.session_state.chat_mode)
                st.rerun()
            
            st.markdown("---")
            
            st.subheader("📚 Historique")
            conversations = load_conversations()
            current_mode_conversations = [conv for conv in conversations if conv["mode"] == st.session_state.chat_mode]
            
            if current_mode_conversations:
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)
                week_ago = today - timedelta(days=7)
                
                today_convs = []
                yesterday_convs = []
                week_convs = []
                older_convs = []
                
                for conv in current_mode_conversations:
                    conv_date = datetime.fromisoformat(conv["timestamp"]).date()
                    if conv_date == today:
                        today_convs.append(conv)
                    elif conv_date == yesterday:
                        yesterday_convs.append(conv)
                    elif conv_date > week_ago:
                        week_convs.append(conv)
                    else:
                        older_convs.append(conv)
                
                if today_convs:
                    st.markdown("*Aujourd'hui*")
                    for conv in today_convs[:5]:
                        display_conversation_item(conv)
                
                if yesterday_convs:
                    st.markdown("*Hier*")
                    for conv in yesterday_convs[:5]:
                        display_conversation_item(conv)
                
                if week_convs:
                    st.markdown("*Cette semaine*")
                    for conv in week_convs[:5]:
                        display_conversation_item(conv)
                
                if older_convs:
                    st.markdown("*Plus ancien*")
                    for conv in older_convs[:5]:
                        display_conversation_item(conv)
            else:
                st.info("Aucune conversation sauvegardée")
        
        st.markdown("""
        <div class="feature-card">
            <h4>📚 Service Client Intelligent avec Cerebras Llama 3.1-8B</h4>
            <p>Téléchargez vos documents et posez des questions. L'IA Cerebras analysera vos fichiers pour vous donner des réponses précises avec rendu mathématique automatique.</p>
            <p><strong>Formats supportés:</strong> PDF, Word, PowerPoint, Excel, Texte</p>
            <p><strong>🧠 Modèle:</strong> Llama 3.1-8B sur Cerebras | <strong>🔍 Nouveau:</strong> RAG avancé avec chunking intelligent!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload de fichiers
        uploaded_files = st.file_uploader(
            "📁 Téléchargez vos documents",
            type=['pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls', 'txt'],
            accept_multiple_files=True,
            help="Formats supportés: PDF, Word, PowerPoint, Excel, Texte"
        )
        
        # Ajout d'un slider pour contrôler le nombre de résultats
        num_results = st.slider("Nombre de résultats à retourner", min_value=5, max_value=20, value=10, help="Augmentez pour obtenir plus de résultats du dataset")
        
        if uploaded_files:
            if st.button("📄 Traiter les Documents avec Cerebras", type="primary"):
                with st.spinner("Traitement des documents avec Cerebras en cours..."):
                    documents = process_uploaded_files(uploaded_files)
                    if documents:
                        db = create_vector_store_with_chunking(documents)
                        if db:
                            st.session_state.db = db
                            st.success(f"✅ {len(documents)} documents traités avec RAG avancé!")
                        else:
                            st.error("❌ Erreur lors de la création de la base de données vectorielle")
                    else:
                        st.error("❌ Aucun document valide trouvé")
        
        # Messages pour Service Client
        if "cs_messages" not in st.session_state:
            st.session_state.cs_messages = [
                {"role": "assistant", "content": "Bonjour! Je suis votre assistant de service client alimenté par Cerebras Llama 3.1-8B avec RAG avancé. Téléchargez vos documents et posez-moi des questions à leur sujet."}
            ]
        
        for message in st.session_state.cs_messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    display_ai_response_with_latex(message["content"], cerebras_token_manager)
                else:
                    st.markdown(message["content"])
        
        if prompt := st.chat_input("Posez votre question sur vos documents..."):
            if should_start_new_conversation(prompt):
                create_new_conversation(st.session_state.chat_mode)
                st.success("🔄 Nouvelle conversation démarrée!")
                st.rerun()
            
            if "db" not in st.session_state:
                st.warning("⚠ Veuillez d'abord télécharger et traiter des documents.")
            else:
                detected_lang = detect_language_with_cerebras(prompt)
                st.markdown(f"""
                <div class="language-indicator">
                    🌍 Langue détectée: {get_language_name(detected_lang)} (Cerebras) | 🔍 RAG Avancé: Activé
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.cs_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Recherche intelligente dans vos documents..."):
                        try:
                            # Utiliser le nombre de résultats sélectionné
                            st.session_state.num_results = num_results
                            response = safe_qa_call(st.session_state.db, prompt, get_language_prompt(detected_lang))
                            
                            display_ai_response_with_latex(response, cerebras_token_manager)
                            st.session_state.cs_messages.append({"role": "assistant", "content": response})
                            
                            # Sauvegarde automatique
                            if len(st.session_state.cs_messages) >= 2:
                                conv_id = auto_save_conversation(st.session_state.cs_messages, "customer_service")
                                if conv_id and not st.session_state.get('current_cs_conversation_id'):
                                    st.session_state.current_cs_conversation_id = conv_id
                        except Exception as e:
                            error_msg = f"Désolé, une erreur s'est produite avec Cerebras: {str(e)}"
                            st.error(error_msg)
                            st.session_state.cs_messages.append({"role": "assistant", "content": error_msg})
        
        if st.session_state.cs_messages and len(st.session_state.cs_messages) > 1:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑 Effacer l'historique", type="secondary"):
                    delete_conversation(st.session_state.current_cs_conversation_id)
                    create_new_conversation(st.session_state.chat_mode)
                    st.rerun()
            
            with col2:
                if st.button("💾 Sauvegarder manuellement", type="primary"):
                    if 'cs_messages' in st.session_state:
                        messages = st.session_state.cs_messages
                        user_msgs = [msg for msg in messages if msg["role"] == "user"]
                        if len(user_msgs) > 0:
                            conv_id = save_conversation(messages, "customer_service")
                            if conv_id:
                                show_save_notification("Sauvegarde manuelle", len(messages), 0)
                                st.balloons()
                                st.rerun()
                        else:
                            st.warning("⚠ Aucun message utilisateur à sauvegarder")
                    else:
                        st.error("❌ Aucune session trouvée")
    
    else:
        # Mode Chat Général avec Cerebras
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); border-radius: 10px; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0;">💬 Chat Cerebras</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("➕ Nouvelle Conversation", use_container_width=True, type="primary"):
                create_new_conversation(st.session_state.chat_mode)
                st.rerun()
            
            st.markdown("---")
            
            st.subheader("💬 Historique")
            conversations = load_conversations()
            general_conversations = [conv for conv in conversations if conv["mode"] == "general_chat"]
            
            if general_conversations:
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)
                week_ago = today - timedelta(days=7)
                
                today_convs = []
                yesterday_convs = []
                week_convs = []
                older_convs = []
                
                for conv in general_conversations:
                    conv_date = datetime.fromisoformat(conv["timestamp"]).date()
                    if conv_date == today:
                        today_convs.append(conv)
                    elif conv_date == yesterday:
                        yesterday_convs.append(conv)
                    elif conv_date > week_ago:
                        week_convs.append(conv)
                    else:
                        older_convs.append(conv)
                
                if today_convs:
                    st.markdown("*Aujourd'hui*")
                    for conv in today_convs[:5]:
                        display_conversation_item(conv)
                
                if yesterday_convs:
                    st.markdown("*Hier*")
                    for conv in yesterday_convs[:5]:
                        display_conversation_item(conv)
                
                if week_convs:
                    st.markdown("*Cette semaine*")
                    for conv in week_convs[:5]:
                        display_conversation_item(conv)
                
                if older_convs:
                    st.markdown("*Plus ancien*")
                    for conv in older_convs[:5]:
                        display_conversation_item(conv)
            else:
                st.info("Aucune conversation sauvegardée")
        
        st.markdown("""
        <div class="feature-card">
            <h4>💬 Chat Général avec Cerebras Llama 3.1-8B</h4>
            <p>Discutez librement avec l'IA Cerebras sur n'importe quel sujet. L'assistant détectera automatiquement votre langue et rendra les équations mathématiques en LaTeX.</p>
            <p><strong>Langues supportées:</strong> Français, English, Español, العربية, Deutsch, Italiano</p>
            <p><strong>🧠 Modèle:</strong> Llama 3.1-8B sur Cerebras | <strong>🔍 Nouveau:</strong> Rendu automatique des équations LaTeX!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'general_llm' not in st.session_state:
            st.session_state.general_llm = initialize_general_llm_with_limits()
        
        if "general_messages" not in st.session_state:
            st.session_state.general_messages = [
                {"role": "assistant", "content": "Bonjour! Je suis votre assistant IA alimenté par Cerebras Llama 3.1-8B avec rendu LaTeX intégré. Posez-moi n'importe quelle question dans votre langue préférée!"}
            ]
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.general_messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    display_ai_response_with_latex(message["content"], cerebras_token_manager)
                else:
                    st.markdown(message["content"])
        
        if prompt := st.chat_input("Posez votre question dans n'importe quelle langue..."):
            if should_start_new_conversation(prompt):
                create_new_conversation(st.session_state.chat_mode)
                st.success("🔄 Nouvelle conversation démarrée!")
                st.rerun()
            
            detected_lang = detect_language_with_cerebras(prompt)
            st.markdown(f"""
            <div class="language-indicator">
                🌍 Langue détectée: {get_language_name(detected_lang)} (Cerebras) | 🔍 Rendu LaTeX: Activé
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.general_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Génération de la réponse avec Cerebras..."):
                    try:
                        enhanced_prompt = f"{get_language_prompt(detected_lang)}\n\nQuestion: {prompt}\n\nRéponds de manière détaillée et complète:"
                        response = safe_cerebras_call(st.session_state.general_llm, enhanced_prompt)
                        
                        display_ai_response_with_latex(response, cerebras_token_manager)
                        st.session_state.general_messages.append({"role": "assistant", "content": response})
                        
                        if len(st.session_state.general_messages) >= 2:
                                conv_id = auto_save_conversation(st.session_state.general_messages, "general_chat")
                                if conv_id and not st.session_state.get('current_general_conversation_id'):
                                    st.session_state.current_general_conversation_id = conv_id
                    except Exception as e:
                        error_msg = f"Désolé, une erreur s'est produite avec Cerebras: {str(e)}"
                        st.error(error_msg)
                        st.session_state.general_messages.append({"role": "assistant", "content": error_msg})
        
        if st.session_state.general_messages and len(st.session_state.general_messages) > 1:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑 Effacer l'historique", type="secondary"):
                    delete_conversation(st.session_state.current_general_conversation_id)
                    create_new_conversation(st.session_state.chat_mode)
                    st.rerun()
            
            with col2:
                if st.button("💾 Sauvegarder manuellement", type="primary"):
                    if 'general_messages' in st.session_state:
                        messages = st.session_state.general_messages
                        user_msgs = [msg for msg in messages if msg["role"] == "user"]
                        if len(user_msgs) > 0:
                            conv_id = save_conversation(messages, "general_chat")
                            if conv_id:
                                show_save_notification("Sauvegarde manuelle", len(messages), 0)
                                st.balloons()
                                st.rerun()
                        else:
                            st.warning("⚠ Aucun message utilisateur à sauvegarder")
                    else:
                        st.error("❌ Aucune session trouvée")

if __name__ == "__main__":
    main()