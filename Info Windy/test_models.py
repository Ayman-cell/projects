#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour charger et vérifier tous les modèles ML.
"""

import pickle
import joblib
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import traceback

# Tentative d'import TensorFlow avec gestion d'erreur
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"⚠ TensorFlow non disponible : {e}")
    print("   Le modèle LSTM ne pourra pas être chargé.")
    TENSORFLOW_AVAILABLE = False
    load_model = None

# ----- chemins vers les fichiers de modèles -----
MODELS_DIR = Path("Models")
PATH_BUNDLE = MODELS_DIR / "model_bundle.pkl"
PATH_LSTM   = MODELS_DIR / "LSTM_best.keras"
PATH_LGBM   = MODELS_DIR / "lgbm_best.pkl"
PATH_XGB    = MODELS_DIR / "xgb_best.pkl"
PATH_HGBR   = MODELS_DIR / "hgbr_best.pkl"


# ================== FONCTIONS DE CHARGEMENT ==================

def load_model_bundle(path: Path):
    """Charge le bundle (pipeline / préprocesseur + modèle(s))."""
    print(f"\n>>> Chargement du model_bundle depuis {path}")
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {path}")
    try:
        bundle = joblib.load(path)
        print("✅ model_bundle chargé avec joblib.")
        return bundle
    except Exception as e:
        print(f"joblib.load a échoué ({e}), tentative avec pickle...")
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        print("✅ model_bundle chargé avec pickle.")
        return bundle


def load_lgbm_model(path: Path):
    """Charge le modèle LightGBM."""
    print(f"\n>>> Chargement du modèle LightGBM depuis {path}")
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {path}")
    try:
        model = joblib.load(path)
        print("✅ LightGBM chargé avec joblib (API sklearn).")
        return model
    except Exception as e:
        print(f"joblib.load a échoué ({e}), tentative Booster natif...")
        booster = lgb.Booster(model_file=str(path))
        print("✅ LightGBM chargé comme Booster natif.")
        return booster


def load_xgb_model(path: Path):
    """Charge le modèle XGBoost."""
    print(f"\n>>> Chargement du modèle XGBoost depuis {path}")
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {path}")
    try:
        model = joblib.load(path)
        print("✅ XGBoost chargé avec joblib (XGBRegressor / XGBClassifier).")
        return model
    except Exception as e:
        print(f"joblib.load a échoué ({e}), tentative Booster natif...")
        booster = xgb.Booster()
        booster.load_model(str(path))
        print("✅ XGBoost chargé comme Booster natif.")
        return booster


def load_hgbr_model(path: Path):
    """Charge le modèle HistogramGradientBoostingRegressor (sklearn)."""
    print(f"\n>>> Chargement du modèle HGBR depuis {path}")
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {path}")
    model = joblib.load(path)
    print("✅ HGBR (sklearn) chargé via joblib.")
    return model


def load_lstm_model(path: Path):
    """Charge le modèle LSTM Keras."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow n'est pas disponible. Installez-le avec : pip install tensorflow")
    print(f"\n>>> Chargement du modèle LSTM Keras depuis {path}")
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {path}")
    model = load_model(str(path))
    print("✅ Modèle LSTM Keras chargé.")
    return model


# ================== FONCTION DE VÉRIFICATION ==================

def check_model(name: str, obj) -> bool:
    """
    Vérifie que l'objet modèle n'est pas None
    et qu'il possède au moins une méthode 'predict' ou similaire.
    """
    print(f"\n=== Vérification du modèle : {name} ===")
    if obj is None:
        print("❌ Le modèle est None -> échec du chargement.")
        return False

    print(f"Type du modèle : {type(obj)}")

    has_predict = hasattr(obj, "predict")
    has_inplace_predict = hasattr(obj, "inplace_predict")  # pour Booster XGBoost
    has_call = hasattr(obj, "__call__")  # pour les modèles Keras

    print(f"Possède 'predict'         : {'✅' if has_predict else '❌'}")
    print(f"Possède 'inplace_predict': {'✅' if has_inplace_predict else '❌'}")
    print(f"Possède '__call__'        : {'✅' if has_call else '❌'}")

    # Si aucun des deux, on considère que ce n'est pas normal pour un modèle
    if not (has_predict or has_inplace_predict or has_call):
        print("⚠ Le modèle ne semble pas avoir de méthode de prédiction standard.")
        return False

    print("✅ Vérification basique OK.")
    return True


# ================== MAIN ==================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DE CHARGEMENT DES MODÈLES ML")
    print("=" * 60)
    
    # Vérifier que le dossier Models existe
    if not MODELS_DIR.exists():
        print(f"\n❌ Le dossier '{MODELS_DIR}' n'existe pas !")
        print("   Assurez-vous que les modèles sont dans le dossier Models/")
        exit(1)
    
    status = {}

    # Bundle
    try:
        bundle = load_model_bundle(PATH_BUNDLE)
        status["model_bundle"] = check_model("model_bundle", bundle)
    except Exception as e:
        print(f"\n❌ Erreur lors du chargement de model_bundle :")
        traceback.print_exc()
        status["model_bundle"] = False

    # LSTM
    if TENSORFLOW_AVAILABLE:
        try:
            lstm_model = load_lstm_model(PATH_LSTM)
            status["LSTM_best"] = check_model("LSTM_best", lstm_model)
        except Exception as e:
            print(f"\n❌ Erreur lors du chargement de LSTM_best :")
            traceback.print_exc()
            status["LSTM_best"] = False
    else:
        print(f"\n⚠ LSTM non testé : TensorFlow non disponible")
        status["LSTM_best"] = None

    # LightGBM
    try:
        lgbm_model = load_lgbm_model(PATH_LGBM)
        status["lgbm_best"] = check_model("lgbm_best", lgbm_model)
    except Exception as e:
        print(f"\n❌ Erreur lors du chargement de lgbm_best :")
        traceback.print_exc()
        status["lgbm_best"] = False

    # XGBoost
    try:
        xgb_model = load_xgb_model(PATH_XGB)
        status["xgb_best"] = check_model("xgb_best", xgb_model)
    except Exception as e:
        print(f"\n❌ Erreur lors du chargement de xgb_best :")
        traceback.print_exc()
        status["xgb_best"] = False

    # HGBR
    try:
        hgbr_model = load_hgbr_model(PATH_HGBR)
        status["hgbr_best"] = check_model("hgbr_best", hgbr_model)
    except Exception as e:
        print(f"\n❌ Erreur lors du chargement de hgbr_best :")
        traceback.print_exc()
        status["hgbr_best"] = False

    # Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ DU CHARGEMENT DES MODÈLES :")
    print("=" * 60)
    for name, ok in status.items():
        if ok is None:
            print(f"- {name:15s} : ⚠ NON TESTÉ (TensorFlow manquant)")
        elif ok:
            print(f"- {name:15s} : ✅ OK")
        else:
            print(f"- {name:15s} : ❌ PROBLÈME")

    successful = [k for k, v in status.items() if v is True]
    failed = [k for k, v in status.items() if v is False]
    skipped = [k for k, v in status.items() if v is None]

    print(f"\n✅ Modèles chargés avec succès : {len(successful)}/{len(status)}")
    if failed:
        print(f"❌ Modèles en échec : {len(failed)}")
    if skipped:
        print(f"⚠ Modèles non testés : {len(skipped)}")

    if all(v is True or v is None for v in status.values()):
        print("\n🎉 Tous les modèles disponibles ont été chargés et vérifiés avec succès.")
    elif successful:
        print(f"\n⚠ {len(failed)} modèle(s) n'ont pas été chargés correctement. Regardez les erreurs ci-dessus.")
    else:
        print("\n❌ Aucun modèle n'a pu être chargé. Vérifiez les chemins et les dépendances.")
