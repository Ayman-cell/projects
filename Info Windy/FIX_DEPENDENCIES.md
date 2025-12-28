# Correction des conflits de dépendances TensorFlow/Keras

## Problème identifié

Vous avez :
- **TensorFlow 2.11.0** (tensorflow-intel)
- **Keras 3.12.0** (trop récent, incompatible)
- **Protobuf 3.20.3** (trop récent, incompatible)

TensorFlow 2.11.0 nécessite :
- Keras >= 2.11.0 et < 2.12.0
- Protobuf >= 3.9.2 et < 3.20.0

## Solution automatique (recommandée)

```bash
python fix_tensorflow_dependencies.py
```

Ce script va :
1. Désinstaller Keras 3.12.0
2. Installer Keras 2.11.0 (compatible)
3. Corriger Protobuf si nécessaire

## Solution manuelle

Si vous préférez faire manuellement :

```bash
# Désinstaller Keras incompatible
pip uninstall keras

# Installer la bonne version de Keras
pip install keras==2.11.0

# Corriger Protobuf
pip install "protobuf<3.20,>=3.9.2"
```

## Vérification

Après correction, testez :

```bash
python test_tensorflow.py
python test_ml_models.py
```

## Alternative : Mettre à jour TensorFlow

Si vous préférez utiliser une version plus récente de TensorFlow qui supporte Keras 3.x :

```bash
# Désinstaller l'ancienne version
pip uninstall tensorflow tensorflow-intel

# Installer TensorFlow 2.15+ (supporte Keras 3.x)
pip install tensorflow>=2.15.0
```

**Note** : Cette option nécessite Python 3.9-3.11 et peut avoir d'autres dépendances.

## Recommandation

Pour TensorFlow 2.11.0, la solution la plus simple est d'installer Keras 2.11.0 (compatible).

