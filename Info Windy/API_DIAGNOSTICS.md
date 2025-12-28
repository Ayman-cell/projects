# 🔍 Guide de Diagnostic des API

Ce document explique comment vérifier que les API fonctionnent correctement après le démarrage du serveur.

## 🚀 Démarrage du serveur

Lancez le serveur avec :
```bash
python Windy_Server.py
```

Vous verrez dans la console les endpoints disponibles et les logs des appels API.

## 📊 Endpoints de Diagnostic

### 1. **Page de Diagnostics Web** (Recommandé)
Ouvrez dans votre navigateur :
```
http://127.0.0.1:5000/diagnostics
```

Cette page affiche :
- ✅ Test de santé du serveur
- ✅ Test de tous les modèles Open-Meteo (ECMWF, GFS, etc.)
- ✅ Test des prévisions avec différents modèles
- ✅ Diagnostics complets du système

### 2. **API de Santé**
```bash
curl http://127.0.0.1:5000/api/health
```

Retourne le statut du serveur.

### 3. **Test API Open-Meteo**
```bash
curl http://127.0.0.1:5000/api/test/openmeteo
```

Teste tous les modèles météorologiques disponibles :
- `auto` (meilleur disponible)
- `ecmwf_ifs` (ECMWF IFS)
- `gfs` (GFS/NOAA)
- `gem` (CMC GEM)
- `icon` (DWD ICON)
- `metno_nordic` (MET Norway)
- `jma_seam` (JMA)

Pour chaque modèle, vous obtenez :
- Statut (ok/error/no_data)
- Temps de réponse
- Nombre de points de données
- Exemple de température

### 4. **Test des Prévisions**
```bash
curl http://127.0.0.1:5000/api/test/forecast
```

Teste les prévisions avec différents modèles et échéances :
- Hour 0 avec modèle auto
- Hour 12 avec ECMWF
- Hour 24 avec GFS

### 5. **Diagnostics Complets**
```bash
curl http://127.0.0.1:5000/api/diagnostics
```

Retourne toutes les informations du système :
- Configuration de la grille
- Configuration de la station
- Tests API de base

## 📝 Logs dans la Console

Quand le serveur tourne, vous verrez dans la console :

```
2024-XX-XX XX:XX:XX - INFO - Appel API Open-Meteo: lat=32.23, lon=-9.25, model=auto
2024-XX-XX XX:XX:XX - INFO - ✓ API Open-Meteo: 72 points de données reçus
```

En cas d'erreur :
```
2024-XX-XX XX:XX:XX - ERROR - ✗ Erreur API Open-Meteo: [détails de l'erreur]
```

## ✅ Vérifications à Faire

1. **Serveur démarré** : Vérifiez `http://127.0.0.1:5000/api/health`
2. **API Open-Meteo accessible** : Vérifiez que tous les modèles retournent `status: "ok"`
3. **Prévisions fonctionnelles** : Vérifiez que les tests de prévision passent
4. **Pas d'erreurs dans les logs** : Surveillez la console pour les messages d'erreur

## 🐛 En cas de Problème

### Erreur de connexion
- Vérifiez votre connexion Internet
- Vérifiez que l'API Open-Meteo est accessible : `https://api.open-meteo.com/v1/forecast`

### Modèle non disponible
- Certains modèles peuvent ne pas être disponibles pour toutes les régions
- Essayez un autre modèle ou utilisez `auto`

### Timeout
- Les appels API ont un timeout de 10 secondes
- Si c'est trop long, vérifiez votre connexion

## 📌 Exemples d'Utilisation

### Test rapide avec curl
```bash
# Santé
curl http://127.0.0.1:5000/api/health

# Test Open-Meteo
curl http://127.0.0.1:5000/api/test/openmeteo | python -m json.tool

# Diagnostics
curl http://127.0.0.1:5000/api/diagnostics | python -m json.tool
```

### Test dans Python
```python
import requests

# Test santé
r = requests.get('http://127.0.0.1:5000/api/health')
print(r.json())

# Test Open-Meteo
r = requests.get('http://127.0.0.1:5000/api/test/openmeteo')
data = r.json()
for model, info in data['models'].items():
    print(f"{model}: {info['status']}")
```

## 🎯 Résultat Attendu

Un système fonctionnel devrait montrer :
- ✅ Tous les modèles avec `status: "ok"`
- ✅ Temps de réponse < 5 secondes
- ✅ Données de température valides
- ✅ Prévisions générées correctement

