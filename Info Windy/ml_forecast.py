#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module pour charger et utiliser les modèles ML de prédiction météorologique.
Utilise les données Open-Meteo corrigées (comme pour le temps réel) pour prédire
les données au point de la station GP2.

Prépare les features exactement comme lors de l'entraînement :
- Features temporelles (hour, dayofweek, month, etc.)
- Features cycliques (hour_sin, hour_cos, month_sin, month_cos)
- Lags (si données historiques disponibles)
- Rolling features (si données historiques disponibles)
- Valeurs actuelles des variables météo
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
import joblib
import re
import io

# Supprimer les warnings XGBoost
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# Chemin vers le dossier Models
MODELS_DIR = Path(__file__).parent / "Models"

# Liste des modèles disponibles
MODEL_FILES = {
    "xgb": "xgb_best.pkl",
    "lgbm": "lgbm_best.pkl",
    "hgbr": "hgbr_best.pkl",
    "bundle": "model_bundle.pkl"
}

# Cache pour les modèles chargés
_loaded_models = {}
_model_bundle = None

# Cache pour les données historiques
_historical_data_cache = None
_historical_data_timestamp = None


# Fonction helper pour charger les modèles avec gestion du BitGenerator
def _load_model_with_bitgenerator_fix(model_path: Path):
    """
    Charge un modèle en gérant les problèmes de compatibilité BitGenerator.
    Utilise plusieurs méthodes pour contourner le problème PCG64.
    """
    # Essayer d'abord avec joblib (peut parfois contourner le problème)
    try:
        model = joblib.load(model_path, mmap_mode=None)
        # Réinitialiser le random_state si présent
        if hasattr(model, 'random_state'):
            model.random_state = np.random.RandomState()
        return model
    except Exception:
        pass
    
    # Si joblib échoue, essayer avec pickle et patch BitGenerator
    try:
        # Patcher numpy.random._pickle pour gérer PCG64
        import numpy.random._pickle as np_pickle
        original_ctor = getattr(np_pickle, '__bit_generator_ctor', None)
        
        def patched_ctor(bit_generator):
            # Si c'est PCG64 ou un BitGenerator incompatible, utiliser MT19937
            bg_str = str(bit_generator)
            if 'PCG64' in bg_str or (isinstance(bit_generator, type) and 'PCG64' in bit_generator.__name__):
                return np.random.MT19937()
            # Sinon, utiliser le constructeur original
            if original_ctor:
                return original_ctor(bit_generator)
            return np.random.MT19937()
        
        np_pickle.__bit_generator_ctor = patched_ctor
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Réinitialiser le random_state si présent
            if hasattr(model, 'random_state'):
                model.random_state = np.random.RandomState()
            
            return model
        finally:
            # Restaurer le constructeur original
            if original_ctor:
                np_pickle.__bit_generator_ctor = original_ctor
    except Exception as e:
        # Si tout échoue, lever l'exception
        raise Exception(f"Impossible de charger le modèle avec gestion BitGenerator: {e}")


def load_model_bundle() -> Optional[Dict[str, Any]]:
    """Charge le bundle de modèles qui contient les scalers et métadonnées."""
    global _model_bundle
    
    if _model_bundle is not None:
        return _model_bundle
    
    bundle_path = MODELS_DIR / MODEL_FILES["bundle"]
    if not bundle_path.exists():
        logger.warning(f"Bundle de modèles non trouvé: {bundle_path}")
        return None
    
    try:
        _model_bundle = joblib.load(bundle_path)
        logger.info("✓ Bundle de modèles chargé")
        return _model_bundle
    except Exception as e:
        logger.error(f"Erreur lors du chargement du bundle: {e}")
        return None


def load_ml_models() -> Dict[str, Any]:
    """
    Charge tous les modèles ML depuis le dossier Models.
    Retourne un dictionnaire avec les modèles chargés.
    Gère gracieusement les erreurs de dépendances manquantes.
    """
    global _loaded_models
    
    if _loaded_models:
        return _loaded_models
    
    models = {}
    
    # Vérifier les dépendances disponibles
    has_lightgbm = False
    try:
        import lightgbm
        has_lightgbm = True
    except ImportError:
        logger.warning("LightGBM non disponible - le modèle lgbm ne sera pas chargé")
    
    for model_name, filename in MODEL_FILES.items():
        if model_name == "bundle":
            continue  # Le bundle est chargé séparément
        
        # Vérifier les dépendances avant d'essayer de charger
        if model_name == "lgbm" and not has_lightgbm:
            logger.warning(f"Modèle {model_name} ignoré : LightGBM non disponible")
            continue
        
        model_path = MODELS_DIR / filename
        
        if not model_path.exists():
            logger.warning(f"Modèle {model_name} non trouvé: {model_path}")
            continue
        
        try:
            if filename.endswith('.pkl'):
                # Pour HGBR, utiliser la fonction helper qui gère le BitGenerator
                if model_name == "hgbr":
                    try:
                        models[model_name] = _load_model_with_bitgenerator_fix(model_path)
                        logger.info(f"✓ Modèle {model_name} chargé depuis {filename} (avec gestion BitGenerator)")
                    except Exception as hgbr_error:
                        logger.error(f"Erreur lors du chargement de {model_name}: {hgbr_error}")
                        logger.warning(f"💡 Le modèle {model_name} a un problème de compatibilité numpy BitGenerator.")
                        logger.warning(f"💡 Solutions:")
                        logger.warning(f"   1. Installer numpy compatible: pip install numpy==1.23.5")
                        logger.warning(f"   2. Ré-entraîner le modèle HGBR avec la version actuelle de numpy")
                        logger.warning(f"   3. Continuer sans HGBR (xgb et lgbm fonctionnent)")
                        continue
                else:
                    # Essayer d'abord avec joblib (meilleure compatibilité)
                    try:
                        models[model_name] = joblib.load(model_path)
                        logger.info(f"✓ Modèle {model_name} chargé depuis {filename} (joblib)")
                    except Exception as joblib_error:
                        logger.debug(f"Joblib a échoué pour {model_name}: {joblib_error}, tentative avec pickle")
                        # Fallback sur pickle si joblib échoue
                        try:
                            with open(model_path, 'rb') as f:
                                models[model_name] = pickle.load(f)
                            logger.info(f"✓ Modèle {model_name} chargé depuis {filename} (pickle)")
                        except Exception as pickle_error:
                            logger.debug(f"Pickle standard a échoué pour {model_name}: {pickle_error}, tentative avec compatibilité numpy")
                            # Si les deux échouent, essayer avec pickle et un contexte de compatibilité numpy
                            try:
                                import numpy as np
                                # Sauvegarder l'état du random state
                                old_state = np.random.get_state()
                                with open(model_path, 'rb') as f:
                                    models[model_name] = pickle.load(f)
                                # Restaurer l'état
                                np.random.set_state(old_state)
                                logger.info(f"✓ Modèle {model_name} chargé depuis {filename} (pickle avec compatibilité numpy)")
                            except Exception as numpy_compat_error:
                                # Dernière tentative : utiliser Unpickler personnalisé pour gérer les BitGenerators incompatibles
                                try:
                                    import numpy as np
                                    import warnings
                                    
                                    # Solution pour BitGenerator incompatible : utiliser joblib avec allow_pickle=True
                                    # ou charger avec pickle en gérant l'erreur et en réinitialisant le random_state
                                    
                                    # Supprimer temporairement les warnings
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        
                                        # Sauvegarder l'état du random state
                                        old_state = np.random.get_state()
                                        
                                        # Essayer de charger avec joblib en mode permissif
                                        try:
                                            # Joblib peut parfois contourner les problèmes de BitGenerator
                                            models[model_name] = joblib.load(model_path, mmap_mode=None)
                                            # Réinitialiser le random_state du modèle si possible
                                            if hasattr(models[model_name], 'random_state'):
                                                models[model_name].random_state = np.random.RandomState()
                                            logger.info(f"✓ Modèle {model_name} chargé depuis {filename} (joblib avec gestion BitGenerator)")
                                        except Exception as joblib_bg_error:
                                            # Si joblib échoue aussi, essayer avec pickle en gérant l'erreur
                                            logger.debug(f"Joblib a échoué pour {model_name}: {joblib_bg_error}, tentative pickle avec gestion d'erreurs")
                                            try:
                                                # Charger avec pickle et capturer l'erreur de BitGenerator
                                                with open(model_path, 'rb') as f:
                                                    try:
                                                        models[model_name] = pickle.load(f)
                                                        # Si le chargement réussit, réinitialiser le random_state
                                                        if hasattr(models[model_name], 'random_state'):
                                                            models[model_name].random_state = np.random.RandomState()
                                                        logger.info(f"✓ Modèle {model_name} chargé depuis {filename} (pickle avec réinitialisation random_state)")
                                                    except (ValueError, TypeError) as bg_error:
                                                        if 'BitGenerator' in str(bg_error) or 'PCG64' in str(bg_error):
                                                            # L'erreur vient du BitGenerator
                                                            # Essayer de charger en ignorant cette erreur spécifique
                                                            logger.warning(f"Erreur BitGenerator pour {model_name}, tentative de contournement...")
                                                            
                                                            # Méthode 1 : Réessayer avec seed fixe
                                                            try:
                                                                np.random.seed(42)  # Seed fixe pour reproductibilité
                                                                f.seek(0)
                                                                models[model_name] = pickle.load(f)
                                                                # Réinitialiser le random_state du modèle
                                                                if hasattr(models[model_name], 'random_state'):
                                                                    models[model_name].random_state = np.random.RandomState()
                                                                logger.info(f"✓ Modèle {model_name} chargé depuis {filename} (pickle avec seed fixe et réinitialisation)")
                                                            except Exception:
                                                                # Méthode 2 : Essayer de patcher numpy.random._pickle
                                                                try:
                                                                    import numpy.random._pickle as np_pickle
                                                                    original_ctor = getattr(np_pickle, '__bit_generator_ctor', None)
                                                                    
                                                                    def patched_ctor(bit_generator):
                                                                        if 'PCG64' in str(bit_generator) or isinstance(bit_generator, type) and 'PCG64' in bit_generator.__name__:
                                                                            return np.random.MT19937()
                                                                        if original_ctor:
                                                                            return original_ctor(bit_generator)
                                                                        return np.random.MT19937()
                                                                    
                                                                    np_pickle.__bit_generator_ctor = patched_ctor
                                                                    f.seek(0)
                                                                    models[model_name] = pickle.load(f)
                                                                    if original_ctor:
                                                                        np_pickle.__bit_generator_ctor = original_ctor
                                                                    
                                                                    # Réinitialiser le random_state du modèle
                                                                    if hasattr(models[model_name], 'random_state'):
                                                                        models[model_name].random_state = np.random.RandomState()
                                                                    logger.info(f"✓ Modèle {model_name} chargé depuis {filename} (pickle avec patch BitGenerator)")
                                                                except Exception as e2:
                                                                    logger.error(f"Toutes les méthodes de contournement ont échoué: {e2}")
                                                                    raise bg_error
                                                        else:
                                                            raise bg_error
                                            except Exception as final_pickle_error:
                                                logger.error(f"Impossible de charger {model_name}: {final_pickle_error}")
                                                logger.warning(f"💡 Le modèle {model_name} a été sauvegardé avec une version incompatible de numpy.")
                                                logger.warning(f"💡 Solutions possibles:")
                                                logger.warning(f"   1. Mettre à jour numpy: pip install --upgrade numpy")
                                                logger.warning(f"   2. Ré-entraîner et re-sauvegarder le modèle {model_name}")
                                                raise final_pickle_error
                                        
                                        # Restaurer l'état
                                        np.random.set_state(old_state)
                                except Exception as final_error:
                                    logger.error(f"Impossible de charger {model_name} avec toutes les méthodes: {final_error}")
                                    logger.error(f"Détails: joblib={joblib_error}, pickle={pickle_error}, numpy_compat={numpy_compat_error}, final={final_error}")
                                    logger.warning(f"💡 Suggestion: Le modèle {model_name} a peut-être été sauvegardé avec une version différente de numpy. Essayez de mettre à jour numpy ou de re-sauvegarder le modèle.")
                                    # Ne pas lever l'exception, continuer avec les autres modèles
                                    continue
            else:
                logger.warning(f"Format de fichier non supporté: {filename}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_name}: {e}")
            logger.debug(f"Détails de l'erreur pour {model_name}:", exc_info=True)
            continue
    
    _loaded_models = models
    
    if not models:
        logger.warning("Aucun modèle ML n'a pu être chargé. Vérifiez les dépendances et les fichiers de modèles.")
    else:
        logger.info(f"✓ {len(models)} modèle(s) ML chargé(s): {', '.join(models.keys())}")
    
    return models


def create_temporal_features(dt: datetime) -> Dict[str, float]:
    """Crée les features temporelles comme dans le script d'entraînement."""
    return {
        'hour': float(dt.hour),
        'minute': float(dt.minute),
        'dayofweek': float(dt.weekday()),
        'month': float(dt.month),
        'quarter': float((dt.month - 1) // 3 + 1),
        'dayofyear': float(dt.timetuple().tm_yday),
        'weekofyear': float(dt.isocalendar()[1]),
        # Features cycliques
        'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
        'month_sin': np.sin(2 * np.pi * dt.month / 12),
        'month_cos': np.cos(2 * np.pi * dt.month / 12),
    }


def load_historical_gp2_data(data_dir: Optional[Path] = None, max_files: int = 1) -> Optional[pd.DataFrame]:
    """
    Charge le dernier fichier GP2 généré (qui contient toute la data historique)
    et le transforme en DataFrame historique pour les features de lag et rolling.
    
    Args:
        data_dir: Dossier contenant les fichiers GP2 (par défaut: ./data)
        max_files: Ignoré (toujours 1 = dernier fichier uniquement)
    
    Returns:
        DataFrame avec les colonnes: Tmp(°C), Vit(m/s), Dir(°), RH(%), Rad. (W/m²), Precipitation
        Indexé par datetime, trié chronologiquement
    """
    global _historical_data_cache, _historical_data_timestamp
    
    # Vérifier le cache (valide pendant 5 minutes)
    if _historical_data_cache is not None and _historical_data_timestamp is not None:
        if (datetime.now() - _historical_data_timestamp).total_seconds() < 300:
            logger.debug("Utilisation du cache des données historiques")
            return _historical_data_cache
    
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    
    if not data_dir.exists():
        logger.warning(f"Dossier de données non trouvé: {data_dir}")
        return None
    
    # Pattern pour extraire le timestamp du nom de fichier
    fname_regex = re.compile(r"GP2_(\d{2}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.txt$")
    
    def extract_ts_from_filename(fname: str) -> Optional[datetime]:
        m = fname_regex.search(fname)
        if not m:
            return None
        date_str, time_str = m.group(1), m.group(2)
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%d-%m-%y_%H-%M-%S")
        except:
            return None
    
    # Trouver tous les fichiers GP2 et trouver le plus récent
    files_with_ts = []
    for f in data_dir.glob("GP2_*.txt"):
        if not f.is_file():
            continue
        ts = extract_ts_from_filename(f.name)
        if ts is not None:
            files_with_ts.append((ts, f))
    
    if not files_with_ts:
        logger.warning(f"Aucun fichier GP2 trouvé dans {data_dir}")
        return None
    
    # Trier par timestamp et prendre UNIQUEMENT le dernier fichier (le plus récent)
    files_with_ts.sort(key=lambda x: x[0], reverse=True)
    latest_file = files_with_ts[0]  # Le dernier fichier généré
    
    logger.info(f"Chargement du dernier fichier GP2: {latest_file[1].name} (contient toute la data historique)")
    
    # Charger uniquement le dernier fichier
    ts, filepath = latest_file
    try:
        # Lire le fichier
        txt = filepath.read_text(encoding='utf-8', errors='ignore')
        
        # Parser avec pandas (détecter séparateur et décimal)
        # Le fichier a le format: Date+Heure Power Speed#@1m Dir RH AirTemp
        # L'en-tête est "Unnamed: 0 Power Speed#@1m Dir RH AirTemp"
        # Mais pandas peut séparer "Unnamed: 0" en deux colonnes, donc on doit gérer ça
        df = None
        for sep in ['\t', r'\s+']:
            try:
                df = pd.read_csv(
                    io.StringIO(txt),
                    sep=sep,
                    engine='python',
                    decimal=',',
                    header=0,
                    skiprows=0
                )
                # Vérifier qu'on a au moins 6 colonnes (Date+Heure, Power, Speed, Dir, RH, AirTemp)
                # Mais si "Unnamed: 0" est séparé, on aura 7 colonnes
                if df.shape[1] >= 6:
                    # Si on a 7 colonnes, c'est que "Unnamed: 0" a été séparé en "Unnamed:" et "0"
                    # Dans ce cas, la date est dans "Unnamed:" et l'heure dans "0"
                    if df.shape[1] == 7 and 'Unnamed:' in df.columns and '0' in df.columns:
                        # Combiner "Unnamed:" et "0" en une seule colonne datetime
                        df['DateTime'] = df['Unnamed:'].astype(str) + ' ' + df['0'].astype(str)
                        # Supprimer les colonnes séparées
                        df = df.drop(columns=['Unnamed:', '0'])
                        # Renommer DateTime en première position
                        cols = ['DateTime'] + [c for c in df.columns if c != 'DateTime']
                        df = df[cols]
                    break
            except Exception as e:
                logger.debug(f"Erreur parsing avec sep='{sep}': {e}")
                continue
        
        if df is None or df.empty:
            logger.warning(f"Fichier {filepath.name} vide ou non parsable")
            return None
        
        logger.debug(f"Fichier parsé: {df.shape[0]} lignes, {df.shape[1]} colonnes. Colonnes: {list(df.columns)}")
            
        # Mapper les colonnes vers les noms attendus
        # Colonnes possibles: Power, Speed#@1m, Speed, Dir, RH, AirTemp, Temp, etc.
        col_mapping = {}
        
        # Chercher les colonnes par nom (insensible à la casse)
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        logger.debug(f"Colonnes disponibles (lowercase): {list(df_cols_lower.keys())}")
        
        # Température - chercher AirTemp en priorité
        if 'airtemp' in df_cols_lower:
            col_mapping['Tmp(°C)'] = df_cols_lower['airtemp']
            logger.debug(f"✓ Température mappée: AirTemp -> Tmp(°C)")
        elif 'temp' in df_cols_lower:
            col_mapping['Tmp(°C)'] = df_cols_lower['temp']
            logger.debug(f"✓ Température mappée: {df_cols_lower['temp']} -> Tmp(°C)")
        else:
            logger.warning("⚠️ Colonne température (AirTemp) non trouvée")
        
        # Vitesse du vent - chercher Speed#@1m en priorité
        if 'speed#@1m' in df_cols_lower:
            col_mapping['Vit(m/s)'] = df_cols_lower['speed#@1m']
            logger.debug(f"✓ Vitesse mappée: Speed#@1m -> Vit(m/s)")
        elif 'speed' in df_cols_lower:
            col_mapping['Vit(m/s)'] = df_cols_lower['speed']
            logger.debug(f"✓ Vitesse mappée: {df_cols_lower['speed']} -> Vit(m/s)")
        else:
            logger.warning("⚠️ Colonne vitesse (Speed#@1m) non trouvée")
        
        # Direction du vent
        if 'dir' in df_cols_lower:
            col_mapping['Dir(°)'] = df_cols_lower['dir']
            logger.debug(f"✓ Direction mappée: {df_cols_lower['dir']} -> Dir(°)")
        else:
            logger.warning("⚠️ Colonne direction (Dir) non trouvée")
        
        # Humidité relative
        if 'rh' in df_cols_lower:
            col_mapping['RH(%)'] = df_cols_lower['rh']
            logger.debug(f"✓ Humidité mappée: RH -> RH(%)")
        else:
            logger.warning("⚠️ Colonne humidité (RH) non trouvée")
        
        # Power (pour Rad. (W/m²) si disponible)
        if 'power' in df_cols_lower:
            col_mapping['Rad. (W/m²)'] = df_cols_lower['power']
            logger.debug(f"✓ Power mappée: Power -> Rad. (W/m²)")
        else:
            logger.warning("⚠️ Colonne Power non trouvée")
        
        # Extraire le timestamp : la première colonne contient Date+Heure combinés (format "dd/mm/yyyy hh:mm:ss")
        # Ou si "Unnamed: 0" a été séparé, on a déjà créé "DateTime"
        datetime_col = None
        if 'DateTime' in df.columns:
            # On a déjà créé la colonne DateTime
            try:
                datetime_col = pd.to_datetime(df['DateTime'], dayfirst=True, format='%d/%m/%Y %H:%M:%S', errors='coerce')
            except Exception as e:
                logger.warning(f"Erreur parsing DateTime: {e}")
        elif len(df.columns) >= 1:
            # La première colonne contient déjà Date+Heure combinés
            try:
                first_col = df.columns[0]
                # Essayer de parser directement la première colonne comme datetime
                datetime_col = pd.to_datetime(df[first_col], dayfirst=True, format='%d/%m/%Y %H:%M:%S', errors='coerce')
                if datetime_col.notna().sum() == 0:
                    # Si ça ne marche pas, essayer sans format explicite
                    datetime_col = pd.to_datetime(df[first_col], dayfirst=True, errors='coerce')
            except Exception as e:
                logger.warning(f"Erreur parsing datetime depuis première colonne: {e}")
                pass
        
        if datetime_col is None or datetime_col.notna().sum() == 0:
            # Fallback: essayer de parser la première colonne seule avec différents formats
            try:
                datetime_col = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors='coerce')
            except:
                pass
        
        if datetime_col is None or datetime_col.notna().sum() == 0:
            # Dernier recours: utiliser le timestamp du nom de fichier pour toutes les lignes
            logger.warning(f"Impossible de parser les dates, utilisation du timestamp du fichier: {ts}")
            datetime_col = pd.Series([ts] * len(df))
        
        # Vérifier que le parsing a fonctionné
        if datetime_col.notna().sum() == 0:
            logger.error(f"⚠️ Aucune date valide parsée dans {filepath.name}")
        else:
            logger.debug(f"✓ {datetime_col.notna().sum()}/{len(datetime_col)} dates parsées avec succès")
        
        # Créer un DataFrame avec les colonnes mappées
        # IMPORTANT: Créer d'abord un DataFrame temporaire avec les valeurs, puis assigner l'index
        temp_data = {}
        
        for target_col, source_col in col_mapping.items():
            # Convertir en numérique (gérer les virgules comme décimales)
            if source_col in df.columns:
                # Vérifier le type actuel de la colonne
                col_data = df[source_col]
                logger.debug(f"Colonne {source_col}: dtype={col_data.dtype}, premier valeur={col_data.iloc[0] if len(col_data) > 0 else 'N/A'}")
                
                # Si c'est déjà numérique, utiliser directement
                if pd.api.types.is_numeric_dtype(col_data):
                    values = col_data.values  # Utiliser .values pour obtenir un array numpy sans index
                else:
                    # Convertir en string, remplacer virgules par points, puis en numérique
                    values = pd.to_numeric(
                        col_data.astype(str).str.replace(',', '.'), 
                        errors='coerce'
                    ).values  # Utiliser .values pour obtenir un array numpy sans index
                
                temp_data[target_col] = values
                
                # Vérifier que la conversion a fonctionné
                if pd.isna(values).all():
                    logger.warning(f"⚠️ Toutes les valeurs de {source_col} sont NaN après conversion. Type original: {col_data.dtype}, échantillon: {col_data.iloc[0] if len(col_data) > 0 else 'N/A'}")
                else:
                    logger.debug(f"✓ {target_col}: {pd.notna(values).sum()}/{len(values)} valeurs valides")
            else:
                logger.warning(f"Colonne source '{source_col}' non trouvée dans le DataFrame. Colonnes disponibles: {list(df.columns)}")
                temp_data[target_col] = np.full(len(df), np.nan)
        
        # Créer le DataFrame avec les données et l'index datetime
        result_df = pd.DataFrame(temp_data, index=datetime_col)
        
        # Ajouter les colonnes manquantes avec NaN
        required_cols = ['Tmp(°C)', 'Vit(m/s)', 'Dir(°)', 'RH(%)', 'Rad. (W/m²)', 'Precipitation']
        for col in required_cols:
            if col not in result_df.columns:
                result_df[col] = np.nan
        
        # Ajouter 'precip' (alias pour Precipitation)
        if 'Precipitation' not in result_df.columns or result_df['Precipitation'].isna().all():
            result_df['precip'] = 0.0
        else:
            result_df['precip'] = result_df['Precipitation'].fillna(0.0)
        
        # Utiliser directement le DataFrame du dernier fichier (qui contient toute la data)
        combined_df = result_df
        
    except Exception as e:
        logger.warning(f"Erreur lors du chargement de {filepath.name}: {e}")
        return None
    
    # Trier par datetime
    combined_df = combined_df.sort_index()
    
    # Supprimer les doublons (garder le dernier)
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
    
    # S'assurer que les colonnes sont dans le bon ordre
    required_cols = ['Tmp(°C)', 'Vit(m/s)', 'Dir(°)', 'RH(%)', 'Rad. (W/m²)', 'Precipitation', 'precip']
    for col in required_cols:
        if col not in combined_df.columns:
            combined_df[col] = 0.0 if col in ['Precipitation', 'precip', 'Rad. (W/m²)'] else np.nan
    
    # Sélectionner uniquement les colonnes requises
    available_cols = [col for col in required_cols if col in combined_df.columns]
    combined_df = combined_df[available_cols]
    
    logger.info(f"✓ Données historiques chargées: {len(combined_df)} points, colonnes: {list(combined_df.columns)}")
    
    # Vérifier la dernière ligne pour debug
    if len(combined_df) > 0:
        last_row = combined_df.iloc[-1]
        logger.info(f"✓ Dernière ligne (index={last_row.name}): Temp={last_row.get('Tmp(°C)', 'N/A')}°C, RH={last_row.get('RH(%)', 'N/A')}%, Vit={last_row.get('Vit(m/s)', 'N/A')}m/s, Dir={last_row.get('Dir(°)', 'N/A')}°")
    
    # Mettre en cache
    _historical_data_cache = combined_df
    _historical_data_timestamp = datetime.now()
    
    return combined_df


def prepare_features_from_current_data(
    temp: float,
    rh: float,
    wind_speed: float,
    wind_dir: float,
    dt: datetime,
    historical_data: Optional[pd.DataFrame] = None
) -> Optional[np.ndarray]:
    """
    Prépare les features pour les modèles ML à partir des données actuelles.
    
    Args:
        temp: Température actuelle (°C)
        rh: Humidité relative actuelle (%)
        wind_speed: Vitesse du vent actuelle (m/s)
        wind_dir: Direction du vent actuelle (degrés)
        dt: Datetime actuel
        historical_data: DataFrame avec données historiques (optionnel, pour lags/rolling)
    
    Returns:
        Tableau numpy des features (None si erreur)
    """
    try:
        bundle = load_model_bundle()
        if bundle is None:
            logger.error("Impossible de charger le bundle de modèles")
            return None
        
        meta = bundle.get('meta', {})
        feature_cols = meta.get('feature_cols', [])
        
        if not feature_cols:
            logger.warning("Aucune information sur les colonnes de features dans le bundle")
            # Utiliser des colonnes par défaut basées sur le script d'entraînement
            feature_cols = []
        
        # Créer un DataFrame avec les valeurs actuelles
        # IMPORTANT : Utiliser les mêmes noms de colonnes que dans le script d'entraînement
        # Note: Le bundle utilise 'precip' et non 'Precipitation'
        data = {
            'Tmp(°C)': temp,
            'Vit(m/s)': wind_speed,
            'Dir(°)': wind_dir,
            'RH(%)': rh,
            'Rad. (W/m²)': 0.0,  # Valeur par défaut (non disponible dans les données fusionnées)
            'precip': 0.0,  # Valeur par défaut (nom utilisé dans le bundle, pas 'Precipitation')
        }
        
        # Ajouter les features temporelles (exactement comme dans le script d'entraînement)
        temporal_features = create_temporal_features(dt)
        data.update(temporal_features)
        
        # TOUJOURS générer les lags et rolling features (même sans données historiques)
        # Le modèle a été entraîné avec ces features, elles doivent donc être présentes
        # Colonnes météo pour les lags (exactement comme dans le script d'entraînement)
        lags = [1, 4, 8, 12, 24, 48]
        meteo_cols_for_lags = ['Tmp(°C)', 'Vit(m/s)', 'Dir(°)', 'Rad. (W/m²)', 'precip']
        # Note: RH(%) n'est PAS dans les lags selon le script d'entraînement
        # Note: Le bundle utilise 'precip' et non 'Precipitation'
        
        if historical_data is not None and len(historical_data) > 0:
            # Utiliser les données historiques si disponibles
            # historical_data est indexé par datetime, on doit trouver les valeurs aux timestamps appropriés
            for col in meteo_cols_for_lags:
                if col in historical_data.columns:
                    for lag in lags:
                        # Calculer le timestamp cible (lag heures avant maintenant)
                        target_time = dt - timedelta(hours=lag)
                        
                        # Trouver la valeur la plus proche dans l'historique
                        # Utiliser iloc pour accéder par position (les données sont triées chronologiquement)
                        if len(historical_data) >= lag:
                            # Prendre la valeur à la position -lag (lag positions avant la fin)
                            try:
                                lag_value = float(historical_data[col].iloc[-lag])
                                if not np.isnan(lag_value):
                                    data[f"{col}_lag_{lag}"] = lag_value
                                else:
                                    data[f"{col}_lag_{lag}"] = data.get(col, 0.0)
                            except (IndexError, KeyError):
                                data[f"{col}_lag_{lag}"] = data.get(col, 0.0)
                        else:
                            # Si pas assez de données, utiliser la valeur actuelle
                            data[f"{col}_lag_{lag}"] = data.get(col, 0.0)
                else:
                    # Si la colonne n'existe pas dans l'historique, utiliser la valeur actuelle
                    current_val = data.get(col, 0.0)
                    for lag in lags:
                        data[f"{col}_lag_{lag}"] = current_val
        else:
            # Sans données historiques, utiliser la valeur actuelle pour tous les lags
            for col in meteo_cols_for_lags:
                current_val = data.get(col, 0.0)
                for lag in lags:
                    data[f"{col}_lag_{lag}"] = current_val
        
        # TOUJOURS générer les rolling features
        # Colonnes pour rolling (exactement comme dans le script d'entraînement)
        windows = [4, 12, 24, 48, 96]
        rolling_cols = ['Tmp(°C)', 'Vit(m/s)', 'Rad. (W/m²)']
        # Note: RH(%) n'est PAS dans les rolling features selon le script d'entraînement
        
        if historical_data is not None and len(historical_data) > 0:
            # Utiliser les données historiques si disponibles
            for col in rolling_cols:
                if col in historical_data.columns:
                    for window in windows:
                        if len(historical_data) >= window:
                            window_data = historical_data[col].iloc[-window:]
                            data[f"{col}_rolling_mean_{window}"] = float(window_data.mean())
                            data[f"{col}_rolling_std_{window}"] = float(window_data.std() if len(window_data) > 1 else 0.0)
                            data[f"{col}_rolling_min_{window}"] = float(window_data.min())
                            data[f"{col}_rolling_max_{window}"] = float(window_data.max())
                        else:
                            # Si pas assez de données, utiliser la valeur actuelle
                            current_val = data.get(col, 0.0)
                            data[f"{col}_rolling_mean_{window}"] = current_val
                            data[f"{col}_rolling_std_{window}"] = 0.0
                            data[f"{col}_rolling_min_{window}"] = current_val
                            data[f"{col}_rolling_max_{window}"] = current_val
                else:
                    # Si la colonne n'existe pas, utiliser la valeur actuelle
                    current_val = data.get(col, 0.0)
                    for window in windows:
                        data[f"{col}_rolling_mean_{window}"] = current_val
                        data[f"{col}_rolling_std_{window}"] = 0.0
                        data[f"{col}_rolling_min_{window}"] = current_val
                        data[f"{col}_rolling_max_{window}"] = current_val
        else:
            # Sans données historiques, utiliser la valeur actuelle pour tous les rolling features
            for col in rolling_cols:
                current_val = data.get(col, 0.0)
                for window in windows:
                    data[f"{col}_rolling_mean_{window}"] = current_val
                    data[f"{col}_rolling_std_{window}"] = 0.0
                    data[f"{col}_rolling_min_{window}"] = current_val
                    data[f"{col}_rolling_max_{window}"] = current_val
        
        # Créer un DataFrame avec toutes les features
        df = pd.DataFrame([data])
        
        logger.debug(f"Features générées avant sélection: {len(df.columns)} colonnes")
        logger.debug(f"Colonnes générées: {list(df.columns)[:20]}...")  # Afficher les 20 premières
        
        # Sélectionner uniquement les colonnes qui sont dans feature_cols
        if feature_cols:
            logger.debug(f"feature_cols du bundle: {len(feature_cols)} colonnes attendues")
            
            # Ajouter les colonnes manquantes avec NaN
            missing_cols = []
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = np.nan
                    missing_cols.append(col)
            
            if missing_cols:
                logger.warning(f"Colonnes manquantes ajoutées avec NaN: {len(missing_cols)} colonnes")
                logger.debug(f"Premières colonnes manquantes: {missing_cols[:10]}")
            
            # Sélectionner dans l'ordre exact
            df_features = df[feature_cols]
            logger.debug(f"Features après sélection: {len(df_features.columns)} colonnes")
        else:
            # Si pas de feature_cols, utiliser toutes les colonnes disponibles
            logger.warning("Aucune feature_cols dans le bundle - utilisation de toutes les colonnes générées")
            df_features = df
        
        # Remplacer NaN par 0 (ou interpolation si nécessaire)
        df_features = df_features.fillna(0)
        
        # Convertir en numpy array
        features = df_features.values.astype(np.float32)
        
        logger.info(f"Features préparées: shape={features.shape}, n_features={features.shape[1] if features.ndim >= 2 else len(features)}")
        
        # Appliquer le scaler X si disponible
        X_scaler = bundle.get('scalers', {}).get('X_scaler')
        if X_scaler is not None:
            features = X_scaler.transform(features)
            logger.debug(f"Features après scaling: shape={features.shape}")
        else:
            logger.warning("X_scaler non disponible dans le bundle")
        
        return features
        
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des features: {e}", exc_info=True)
        return None


def prepare_features_from_fused_data(
    fused_data: Dict[str, Any],
    station_lat: float,
    station_lon: float,
    grid_config: Any,
    hours_ahead: int = 0,
    historical_data: Optional[pd.DataFrame] = None
) -> Optional[np.ndarray]:
    """
    Prépare les features pour les modèles ML à partir des données fusionnées.
    
    Args:
        fused_data: Dictionnaire contenant les champs fusionnés (temp_corr, rh_corr, u_corr, v_corr)
        station_lat: Latitude de la station GP2
        station_lon: Longitude de la station GP2
        grid_config: Configuration de la grille (GRID_CFG)
        hours_ahead: Nombre d'heures à prédire (0 = maintenant)
        historical_data: DataFrame avec données historiques (optionnel)
    
    Returns:
        Tableau numpy des features (None si erreur)
    """
    try:
        import Windy_Open_Meteo as core
        
        # Récupérer les champs fusionnés
        temp_corr = fused_data.get("temp_corr")
        rh_corr = fused_data.get("rh_corr")
        u_corr = fused_data.get("u_corr")
        v_corr = fused_data.get("v_corr")
        
        if temp_corr is None or rh_corr is None or u_corr is None or v_corr is None:
            logger.error("Données fusionnées incomplètes")
            return None
        
        # Convertir en numpy arrays si nécessaire
        if not isinstance(temp_corr, np.ndarray):
            temp_corr = np.array(temp_corr)
        if not isinstance(rh_corr, np.ndarray):
            rh_corr = np.array(rh_corr)
        if not isinstance(u_corr, np.ndarray):
            u_corr = np.array(u_corr)
        if not isinstance(v_corr, np.ndarray):
            v_corr = np.array(v_corr)
        
        # Interpoler les valeurs au point de la station
        i_f, j_f = core.latlon_to_ij_float(station_lat, station_lon, grid_config)
        
        temp_at_station = core.bilinear_interp_indices(temp_corr, i_f, j_f)
        rh_at_station = core.bilinear_interp_indices(rh_corr, i_f, j_f)
        u_at_station = core.bilinear_interp_indices(u_corr, i_f, j_f)
        v_at_station = core.bilinear_interp_indices(v_corr, i_f, j_f)
        
        # Calculer la vitesse et direction du vent
        wind_speed = np.sqrt(u_at_station**2 + v_at_station**2)
        wind_dir = np.arctan2(u_at_station, v_at_station) * 180 / np.pi
        if wind_dir < 0:
            wind_dir += 360
        
        # Datetime pour les features temporelles
        dt = datetime.utcnow() + timedelta(hours=hours_ahead)
        
        # Préparer les features
        features = prepare_features_from_current_data(
            temp=float(temp_at_station),
            rh=float(rh_at_station),
            wind_speed=float(wind_speed),
            wind_dir=float(wind_dir),
            dt=dt,
            historical_data=historical_data
        )
        
        return features
        
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des features: {e}", exc_info=True)
        return None


def predict_with_models(
    fused_data: Dict[str, Any],
    station_lat: float,
    station_lon: float,
    grid_config: Any,
    hours_ahead: int = 0,
    model_names: Optional[List[str]] = None,
    historical_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Prédit les données météorologiques au point de la station GP2 en utilisant les modèles ML.
    
    Args:
        fused_data: Dictionnaire contenant les champs fusionnés
        station_lat: Latitude de la station GP2
        station_lon: Longitude de la station GP2
        grid_config: Configuration de la grille
        hours_ahead: Nombre d'heures à prédire (0 = maintenant)
        model_names: Liste des modèles à utiliser (None = tous)
        historical_data: DataFrame avec données historiques (optionnel)
    
    Returns:
        Dictionnaire avec les prédictions de chaque modèle
    """
    models = load_ml_models()
    bundle = load_model_bundle()
    
    if not models:
        logger.error("Aucun modèle ML disponible")
        return {}
    
    if model_names is None:
        model_names = list(models.keys())
    
    if bundle is None:
        logger.error("Bundle de modèles non disponible - impossible de préparer les features")
        return {}
    
    # Préparer les features
    features = prepare_features_from_fused_data(
        fused_data, station_lat, station_lon, grid_config, hours_ahead, historical_data
    )
    
    if features is None:
        logger.error("Impossible de préparer les features")
        return {}
    
    # S'assurer que features est 2D (batch, features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Sauvegarder les features originales
    features_original = features.copy()
    
    # Vérifier le nombre de features
    n_features_single = features.shape[1] if features.ndim >= 2 else len(features)
    logger.info(f"Features pour prédiction: shape={features.shape}, n_features={n_features_single}")
    
    # IMPORTANT: Les modèles classiques (XGB, LGBM, HGBR) ont été entraînés avec des séquences aplaties
    # Le script d'entraînement fait: X_train_2d = X_train.reshape(X_train.shape[0], -1)
    # Donc (batch, seq_length=24, n_features=96) -> (batch, 24*96=2304)
    # Il faut créer une séquence de 24 timesteps et l'aplatir pour les modèles classiques
    seq_length = bundle.get('meta', {}).get('seq_length', 24) if bundle else 24
    expected_features_flattened = n_features_single * seq_length  # 96 * 24 = 2304
    
    # Vérifier chaque modèle individuellement pour déterminer s'il a besoin de flattening
    models_to_check = model_names if model_names else list(models.keys())
    model_needs_flattening = {}  # Dictionnaire pour chaque modèle
    
    if models:
        # Vérifier les modèles classiques
        for model_name in models_to_check:
            
            if model_name not in models:
                continue
            
            model = models[model_name]
            needs_flatten = False
            
            # Pour XGBoost/LightGBM, vérifier n_features_in_
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                logger.info(f"Modèle {model_name} attend {expected_features} features (n_features_in_)")
                if expected_features == expected_features_flattened:
                    needs_flatten = True
                    logger.info(f"✓ Le modèle {model_name} attend des séquences aplaties ({expected_features} = {n_features_single} × {seq_length})")
                elif expected_features == n_features_single:
                    logger.info(f"✓ Le modèle {model_name} attend des features simples ({expected_features})")
                    needs_flatten = False
                else:
                    logger.warning(f"⚠️ Le modèle {model_name} attend {expected_features} features, mais on a {n_features_single} (simple) ou {expected_features_flattened} (aplati)")
            elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                # Pour MultiOutputRegressor, vérifier le premier estimateur
                first_estimator = model.estimators_[0]
                if hasattr(first_estimator, 'n_features_in_'):
                    expected_features = first_estimator.n_features_in_
                    logger.info(f"Modèle {model_name} (MultiOutput) attend {expected_features} features")
                    if expected_features == expected_features_flattened:
                        needs_flatten = True
                        logger.info(f"✓ Le modèle {model_name} attend des séquences aplaties ({expected_features} = {n_features_single} × {seq_length})")
                    elif expected_features == n_features_single:
                        logger.info(f"✓ Le modèle {model_name} attend des features simples ({expected_features})")
                        needs_flatten = False
                    else:
                        logger.warning(f"⚠️ Le modèle {model_name} attend {expected_features} features, mais on a {n_features_single} (simple) ou {expected_features_flattened} (aplati)")
            elif hasattr(model, 'feature_names_in_'):
                # Pour certains modèles scikit-learn (comme HGBR), vérifier feature_names_in_
                # Mais cela ne donne pas le nombre de features, donc on essaie les deux formats
                logger.info(f"Modèle {model_name} a feature_names_in_ mais pas n_features_in_, test des deux formats")
                needs_flatten = True  # Par défaut, essayer avec flattening
            else:
                # Si on ne peut pas déterminer, essayer avec flattening par défaut pour les modèles classiques
                logger.warning(f"⚠️ Impossible de déterminer le nombre de features attendu pour {model_name} (type: {type(model)}), tentative avec flattening")
                needs_flatten = True
            
            model_needs_flattening[model_name] = needs_flatten
    
    # Préparer les features pour chaque modèle
    features_for_models = {}
    
    # Créer la séquence si nécessaire
    if any(model_needs_flattening.values()):
        # Créer une séquence de seq_length timesteps
        if historical_data is not None and len(historical_data) >= seq_length:
            logger.info(f"Création de séquence à partir de {len(historical_data)} points historiques")
            features_seq = np.tile(features_original, (seq_length, 1))
        else:
            logger.info(f"Création de séquence en répétant les features actuelles {seq_length} fois")
            features_seq = np.tile(features_original, (seq_length, 1))
        
        # Aplatir pour les modèles classiques qui en ont besoin
        features_flattened = features_seq.reshape(1, -1)
        logger.info(f"Features aplaties: shape={features_flattened.shape}, n_features={features_flattened.shape[1]}")
        
        # Assigner les features selon les besoins de chaque modèle
        for model_name in models_to_check:
            if model_needs_flattening.get(model_name, False):
                features_for_models[model_name] = features_flattened
            else:
                features_for_models[model_name] = features_original
    else:
        # Aucun modèle n'a besoin de flattening, utiliser les features originales
        for model_name in models_to_check:
            features_for_models[model_name] = features_original
    
    # S'assurer que tous les modèles à prédire ont leurs features préparées
    models_to_predict = model_names if model_names else list(models.keys())
    for model_name in models_to_predict:
        if model_name not in features_for_models:
            # Si le modèle n'a pas été préparé, utiliser les features originales
            features_for_models[model_name] = features_original
    
    predictions = {}
    y_scaler = bundle.get('scalers', {}).get('y_scaler') if bundle else None
    target_cols = bundle.get('meta', {}).get('target_cols', []) if bundle else []
    
    for model_name in models_to_predict:
        if model_name not in models:
            logger.warning(f"Modèle {model_name} non disponible")
            continue
        
        try:
            model = models[model_name]
            
            # Utiliser les features préparées pour ce modèle spécifique
            model_features = features_for_models.get(model_name, features_original)
            
            logger.info(f"Prédiction avec {model_name}: features shape={model_features.shape}")
            
            # Modèles classiques (XGB, LGBM, HGBR)
            try:
                logger.info(f"Tentative prédiction {model_name} avec features shape={model_features.shape}")
                pred = model.predict(model_features)
                logger.info(f"✓ Prédiction {model_name} réussie, shape={pred.shape if isinstance(pred, np.ndarray) else type(pred)}")
            except Exception as e:
                    logger.error(f"Erreur prédiction {model_name} avec features shape={model_features.shape}: {e}", exc_info=True)
                    # Essayer avec les features originales si l'erreur vient du format
                    if "Feature shape mismatch" in str(e) or "feature_names" in str(e).lower() or "feature" in str(e).lower():
                        logger.info(f"Tentative avec features originales (shape={features_original.shape}) pour {model_name}")
                        try:
                            pred = model.predict(features_original)
                            logger.info(f"✓ Prédiction {model_name} réussie avec features originales, shape={pred.shape if isinstance(pred, np.ndarray) else type(pred)}")
                        except Exception as e2:
                            logger.error(f"Erreur même avec features originales pour {model_name}: {e2}", exc_info=True)
                            # Essayer aussi avec features aplaties si on n'a pas déjà essayé
                            if model_features.shape != features_flattened.shape and 'features_flattened' in locals():
                                logger.info(f"Tentative avec features aplaties (shape={features_flattened.shape}) pour {model_name}")
                                try:
                                    pred = model.predict(features_flattened)
                                    logger.info(f"✓ Prédiction {model_name} réussie avec features aplaties")
                                except Exception as e3:
                                    logger.error(f"Erreur avec toutes les tentatives pour {model_name}: {e3}")
                                    predictions[model_name] = {"error": f"Erreur format: {str(e)} (tentatives: originales={str(e2)}, aplaties={str(e3)})"}
                                    continue
                            else:
                                predictions[model_name] = {"error": f"Erreur format: {str(e)} (tentative originale: {str(e2)})"}
                                continue
                    else:
                        predictions[model_name] = {"error": str(e)}
                        continue
            
            # S'assurer que pred est 2D
            if isinstance(pred, np.ndarray):
                if pred.ndim == 1:
                    pred = pred.reshape(1, -1)
            
            # Dénormaliser si scaler disponible
            if y_scaler is not None:
                try:
                    pred = y_scaler.inverse_transform(pred)
                except Exception as e:
                    logger.warning(f"Erreur lors de la dénormalisation: {e}")
            
            # Extraire les valeurs selon l'ordre des target_cols
            # Format attendu: Tmp(°C), Vit(m/s), Dir(°), RH(%)
            if isinstance(pred, np.ndarray) and pred.ndim == 2 and pred.shape[0] > 0:
                pred_dict = {}
                
                logger.debug(f"Prédiction {model_name}: shape={pred.shape}, target_cols={target_cols}")
                
                # Mapper selon target_cols si disponible
                if target_cols and len(target_cols) > 0:
                    logger.debug(f"Mapping pour {model_name}: target_cols={target_cols}, pred.shape={pred.shape}")
                    for i, col in enumerate(target_cols):
                        if i < pred.shape[1]:
                            val = float(pred[0, i])
                            col_lower = col.lower()
                            # Mapping par nom exact en priorité
                            if col == 'Tmp(°C)' or col == 'Tmp':
                                pred_dict['temp'] = val
                            elif col == 'Vit(m/s)' or col == 'Vit':
                                pred_dict['wind_speed'] = val
                            elif col == 'Dir(°)' or col == 'Dir':
                                pred_dict['wind_dir'] = val
                            elif col == 'RH(%)' or col == 'RH':
                                pred_dict['rh'] = val
                                logger.debug(f"Humidité trouvée à l'index {i} (col='{col}') pour {model_name}: {val}")
                            # Mapping flexible si pas encore assigné
                            elif 'tmp' in col_lower or 'temp' in col_lower or 'temperature' in col_lower:
                                if 'temp' not in pred_dict:
                                    pred_dict['temp'] = val
                            elif 'vit' in col_lower or 'speed' in col_lower or 'wind_speed' in col_lower:
                                if 'wind_speed' not in pred_dict:
                                    pred_dict['wind_speed'] = val
                            elif 'dir' in col_lower or 'direction' in col_lower or 'wind_dir' in col_lower:
                                if 'wind_dir' not in pred_dict:
                                    pred_dict['wind_dir'] = val
                            elif 'rh' in col_lower or 'humid' in col_lower or 'relative_humidity' in col_lower:
                                if 'rh' not in pred_dict:
                                    pred_dict['rh'] = val
                                    logger.debug(f"Humidité trouvée à l'index {i} (col='{col}') pour {model_name}: {val}")
                else:
                    # Fallback: ordre par défaut (Tmp, Vit, Dir, RH)
                    logger.warning(f"target_cols non disponible pour {model_name}, utilisation de l'ordre par défaut")
                    if pred.shape[1] >= 1:
                        pred_dict['temp'] = float(pred[0, 0])
                    if pred.shape[1] >= 2:
                        pred_dict['wind_speed'] = float(pred[0, 1])
                    if pred.shape[1] >= 3:
                        pred_dict['wind_dir'] = float(pred[0, 2])
                    if pred.shape[1] >= 4:
                        pred_dict['rh'] = float(pred[0, 3])
                
                # Vérifier que l'humidité est présente, sinon essayer de la trouver
                if 'rh' not in pred_dict and pred.shape[1] >= 4:
                    # Essayer de trouver RH dans les colonnes restantes
                    for i in range(4, min(pred.shape[1], len(target_cols) if target_cols else 10)):
                        if target_cols and i < len(target_cols):
                            col = target_cols[i]
                            if 'rh' in col.lower() or 'humid' in col.lower():
                                pred_dict['rh'] = float(pred[0, i])
                                logger.info(f"Humidité trouvée à l'index {i} pour {model_name}: {pred_dict['rh']}")
                                break
                
                # Log pour debug
                if 'rh' not in pred_dict:
                    logger.warning(f"⚠️ Humidité non trouvée dans les prédictions de {model_name}. target_cols={target_cols}, pred.shape={pred.shape}")
                    # Fallback: utiliser l'humidité de la dernière ligne GP2 si disponible dans historical_data
                    if historical_data is not None and len(historical_data) > 0 and 'RH(%)' in historical_data.columns:
                        last_rh = float(historical_data['RH(%)'].iloc[-1])
                        if not np.isnan(last_rh):
                            pred_dict['rh'] = last_rh
                            logger.info(f"Utilisation de l'humidité de la dernière ligne GP2 pour {model_name}: {last_rh}%")
                else:
                    logger.debug(f"✓ Prédiction {model_name}: temp={pred_dict.get('temp', 'N/A')}, rh={pred_dict.get('rh', 'N/A')}, wind_speed={pred_dict.get('wind_speed', 'N/A')}, wind_dir={pred_dict.get('wind_dir', 'N/A')}")
                
                # Si on a wind_speed et wind_dir, calculer u et v
                if 'wind_speed' in pred_dict and 'wind_dir' in pred_dict:
                    ws = pred_dict['wind_speed']
                    wd = pred_dict['wind_dir']
                    wd_rad = np.deg2rad(wd)
                    u = -ws * np.sin(wd_rad)  # Convention météo
                    v = -ws * np.cos(wd_rad)
                    pred_dict['u'] = float(u)
                    pred_dict['v'] = float(v)
                
                predictions[model_name] = pred_dict
            else:
                logger.warning(f"Format de prédiction inattendu pour {model_name}: shape={pred.shape if isinstance(pred, np.ndarray) else type(pred)}")
                predictions[model_name] = {"error": f"Format inattendu: {type(pred)}"}
                
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction avec {model_name}: {e}", exc_info=True)
            predictions[model_name] = {"error": str(e)}
    
    return predictions


def predict_timeseries(
    fused_data: Dict[str, Any],
    station_lat: float,
    station_lon: float,
    grid_config: Any,
    hours: List[int],
    model_names: Optional[List[str]] = None,
    historical_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Prédit une série temporelle pour plusieurs heures à l'avance.
    
    Args:
        fused_data: Dictionnaire contenant les champs fusionnés
        station_lat: Latitude de la station GP2
        station_lon: Longitude de la station GP2
        grid_config: Configuration de la grille
        hours: Liste des heures à prédire (ex: [0, 1, 2, ..., 168])
        model_names: Liste des modèles à utiliser (None = tous)
        historical_data: DataFrame avec données historiques (optionnel)
    
    Returns:
        Dictionnaire avec les prédictions pour chaque heure et chaque modèle
    """
    timeseries = {}
    
    for h in hours:
        predictions = predict_with_models(
            fused_data, station_lat, station_lon, grid_config,
            hours_ahead=h, model_names=model_names, historical_data=historical_data
        )
        timeseries[h] = predictions
    
    return timeseries
