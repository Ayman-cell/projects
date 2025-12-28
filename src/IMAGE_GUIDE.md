# 📸 Guide d'Ajout d'Images Personnelles

## Comment ajouter vos propres images à l'application Airboard

### Méthode 1 : Coller directement dans le chat ✨ (RECOMMANDÉ)

Vous pouvez **coller vos images directement dans le chat Figma Make** et je les intégrerai automatiquement dans votre application.

**Étapes :**
1. Copiez l'image depuis votre ordinateur (Ctrl+C / Cmd+C)
2. Collez-la directement dans le chat (Ctrl+V / Cmd+V)
3. Dites-moi où vous voulez l'utiliser (ex: "Utilise cette image pour Ayman dans la page À propos")
4. Je l'intégrerai automatiquement !

### Méthode 2 : Utiliser des URLs d'images

Si vos images sont déjà hébergées quelque part (Google Drive, Dropbox, serveur web, etc.), vous pouvez me donner les URLs et je les intégrerai.

**Format accepté :**
```
https://votre-domaine.com/image.jpg
https://votre-domaine.com/image.png
```

---

## 🎯 Images à remplacer dans votre application

### Page "À propos" (`/components/pages/AboutUsPage.tsx`)

**Lignes 15-36** - Photos des membres de l'équipe :

```typescript
const teamMembers = [
  {
    name: 'Ayman Amasrour',
    image: 'URL_DE_VOTRE_IMAGE_ICI',  // ← Remplacer cette URL
  },
  {
    name: 'Rihab Essafi',
    image: 'URL_DE_VOTRE_IMAGE_ICI',  // ← Remplacer cette URL
  },
  {
    name: 'Jad Lasiri',
    image: 'URL_DE_VOTRE_IMAGE_ICI',  // ← Remplacer cette URL
  },
]
```

### Page "Comment ça fonctionne" (`/components/pages/HowItWorksPage.tsx`)

**Lignes 13-86** - Images techniques pour chaque étape :

```typescript
const steps = [
  {
    title: 'Collecte des Données',
    image: 'URL_IMAGE_CAPTEURS',  // ← Photo de vos capteurs industriels
  },
  {
    title: 'Ingestion & Stockage',
    image: 'URL_IMAGE_ARCHITECTURE',  // ← Schéma de votre architecture de données
  },
  {
    title: 'Prévisions par Machine Learning',
    image: 'URL_IMAGE_ML',  // ← Visualisation de vos modèles ML
  },
  {
    title: 'Génération de Scénarios',
    image: 'URL_IMAGE_SCENARIOS',  // ← Capture d'écran de vos scénarios
  },
  {
    title: 'Alertes & Notifications',
    image: 'URL_IMAGE_ALERTES',  // ← Interface de votre système d'alertes
  },
  {
    title: 'Analyse & Reporting',
    image: 'URL_IMAGE_ANALYTICS',  // ← Capture de vos tableaux de bord
  },
]
```

---

## 💡 Conseils pour de bonnes images

### Photos d'équipe :
- Format : JPG ou PNG
- Résolution recommandée : 800x800px minimum
- Fond : Neutre de préférence
- Éclairage : Bon éclairage naturel ou professionnel

### Images techniques :
- Format : JPG, PNG ou SVG
- Résolution recommandée : 1200x800px minimum
- Qualité : Nette et lisible
- Contenu : Représentatif de la fonctionnalité décrite

---

## 🚀 Exemples d'utilisation

### Exemple 1 : Remplacer la photo d'Ayman
```
"Voici la photo d'Ayman, remplace l'image actuelle dans la page À propos"
[Collez l'image]
```

### Exemple 2 : Ajouter plusieurs images techniques
```
"Voici 3 images pour la page Comment ça fonctionne :
- Image 1 : Pour 'Collecte des Données'
- Image 2 : Pour 'Ingestion & Stockage'  
- Image 3 : Pour 'Prévisions ML'"
[Collez les images]
```

### Exemple 3 : Utiliser des URLs
```
"Remplace les images de l'équipe avec ces URLs :
- Ayman : https://mon-site.com/ayman.jpg
- Rihab : https://mon-site.com/rihab.jpg
- Jad : https://mon-site.com/jad.jpg"
```

---

## ❓ Questions fréquentes

**Q : Les images doivent-elles être hébergées quelque part ?**
R : Non ! Vous pouvez simplement les coller directement dans le chat.

**Q : Quelle taille maximum pour les images ?**
R : Il n'y a pas de limite stricte, mais pour de meilleures performances, gardez-les sous 5MB.

**Q : Puis-je utiliser des GIFs animés ?**
R : Oui, les GIFs sont supportés !

**Q : Les images sont-elles automatiquement optimisées ?**
R : Oui, le composant `ImageWithFallback` gère automatiquement le chargement et les erreurs.

---

**Prêt à ajouter vos images ? Collez-les simplement dans le chat ! 🎨**
