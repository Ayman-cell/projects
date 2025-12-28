// ============================================================================
// GLOBE 3D WINDY-LIKE - CesiumJS
// python -m http.server 8000
// ============================================================================

// Variables globales
let viewer;
let temperatureLayer;
let temperatureCanvas = null;
let temperatureCtx = null;
let particleCollection;
let particles = [];
let time = 0;

// Paramètres
let windEnabled = true;
let particleCount = 10000; // Nombre total de particules (peut être grand)
let windSpeed = 1.0;
let tempOpacity = 0.6;

// Paramètres de rendu (limite pour éviter les erreurs)
let maxRenderedParticles = 5000; // Maximum de particules rendues à l'écran (non utilisé maintenant, on rend toutes les particules)

// Fonction pour ajuster dynamiquement le nombre de particules rendues
function updateMaxRenderedParticles() {
    // Ajuster automatiquement selon le nombre total de particules
    if (particleCount <= 5000) {
        maxRenderedParticles = particleCount; // Rendre toutes les particules
    } else if (particleCount <= 10000) {
        maxRenderedParticles = 5000; // Rendre 50%
    } else if (particleCount <= 20000) {
        maxRenderedParticles = 6000; // Rendre 30%
    } else {
        maxRenderedParticles = 7000; // Maximum absolu
    }
}

// ============================================================================
// INITIALISATION CESIUM
// ============================================================================

function initCesium() {
    // Désactiver complètement Ion AVANT de créer le viewer
    if (typeof Cesium !== 'undefined' && Cesium.Ion) {
        Cesium.Ion.defaultAccessToken = '';
        // Désactiver complètement Ion
        if (Cesium.Ion.defaultServer) {
            Cesium.Ion.defaultServer = '';
        }
    }

    // Créer le viewer SANS provider par défaut pour éviter Ion
    viewer = new Cesium.Viewer('cesiumContainer', {
        timeline: false,
        animation: false,
        baseLayerPicker: false,
        fullscreenButton: false,
        vrButton: false,
        geocoder: false,
        homeButton: false,
        infoBox: false,
        sceneModePicker: false,
        selectionIndicator: false,
        navigationHelpButton: false,
        terrainProvider: new Cesium.EllipsoidTerrainProvider(),
        imageryProvider: false // CRUCIAL: Pas de provider par défaut (évite Ion)
    });

    // Supprimer TOUTES les couches par défaut immédiatement
    if (viewer.imageryLayers) {
        viewer.imageryLayers.removeAll();
    }

    // Ajouter OpenStreetMap (style Google Maps)
    try {
        const osmProvider = new Cesium.OpenStreetMapImageryProvider({
            url: 'https://a.tile.openstreetmap.org/',
            credit: '© OpenStreetMap contributors'
        });
        viewer.imageryLayers.addImageryProvider(osmProvider);
        console.log('✅ Carte OSM ajoutée');
    } catch (error) {
        console.error('❌ Erreur OSM:', error);
        // Fallback: utiliser un autre provider
        try {
            const esriProvider = new Cesium.ArcGisMapServerImageryProvider({
                url: 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer'
            });
            viewer.imageryLayers.addImageryProvider(esriProvider);
            console.log('✅ Carte ArcGIS ajoutée (fallback)');
        } catch (fallbackError) {
            console.error('❌ Erreur fallback:', fallbackError);
        }
    }

    // Vue globale
    viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(0, 0, 20000000)
    });

    // Collection de particules
    particleCollection = new Cesium.PointPrimitiveCollection();
    viewer.scene.primitives.add(particleCollection);

    console.log('✅ Cesium initialisé');
}

// ============================================================================
// GÉNÉRATION TEXTURE TEMPÉRATURE
// ============================================================================

function generateTemperatureTexture() {
    const width = 1024;
    const height = 512;
    
    // Créer le canvas une seule fois et le réutiliser
    if (!temperatureCanvas) {
        temperatureCanvas = document.createElement('canvas');
        temperatureCanvas.width = width;
        temperatureCanvas.height = height;
        temperatureCtx = temperatureCanvas.getContext('2d');
    }
    
    const imageData = temperatureCtx.createImageData(width, height);
    const data = imageData.data;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const u = x / width;
            const v = y / height;
            
            // Latitude et longitude
            const lat = (v - 0.5) * Math.PI;
            const lon = u * 2 * Math.PI;

            // Gradient équatorial (plus chaud à l'équateur)
            const gradient = Math.cos(lat);
            
            // Bruit simple pour variation (animation lente et fluide)
            const noise = Math.sin(lon * 5 + time * 0.05) * Math.cos(lat * 3) * 0.3;
            
            // Température normalisée (0-1)
            let temp = (gradient + noise + 1) / 2;
            temp = Math.max(0, Math.min(1, temp));

            // Palette de couleurs: bleu → cyan → jaune → orange → rouge
            let r, g, b;
            if (temp < 0.25) {
                const t = temp / 0.25;
                r = 0;
                g = t * 255;
                b = 255;
            } else if (temp < 0.5) {
                const t = (temp - 0.25) / 0.25;
                r = t * 255;
                g = 255;
                b = (1 - t) * 255;
            } else if (temp < 0.75) {
                const t = (temp - 0.5) / 0.25;
                r = 255;
                g = 255 - t * 128;
                b = 0;
            } else {
                const t = (temp - 0.75) / 0.25;
                r = 255;
                g = 127 - t * 127;
                b = 0;
            }

            const index = (y * width + x) * 4;
            data[index] = r;
            data[index + 1] = g;
            data[index + 2] = b;
            data[index + 3] = Math.floor(tempOpacity * 255);
        }
    }

    // Mettre à jour le canvas directement (pas de recréation)
    temperatureCtx.putImageData(imageData, 0, 0);
    return temperatureCanvas.toDataURL('image/png');
}

function addTemperatureLayer() {
    if (!viewer || !viewer.imageryLayers) return;

    try {
        // Si la couche n'existe pas encore, la créer une seule fois
        if (!temperatureLayer) {
            // Générer la texture initiale
            const dataUrl = generateTemperatureTexture();
            
            const provider = new Cesium.SingleTileImageryProvider({
                url: dataUrl,
                rectangle: Cesium.Rectangle.MAX_VALUE
            });
            temperatureLayer = viewer.imageryLayers.addImageryProvider(provider);
            console.log('✅ Couche température créée');
        }
        
        // Toujours mettre à jour l'opacité
        if (temperatureLayer) {
            temperatureLayer.alpha = tempOpacity;
        }
    } catch (error) {
        console.error('❌ Erreur température:', error);
    }
}

// Mettre à jour la texture de température de manière fluide (sans recréer la couche)
let temperatureUpdatePending = false;
function updateTemperatureTexture() {
    if (!temperatureLayer || !temperatureCanvas || temperatureUpdatePending) return;
    
    // Marquer comme en cours pour éviter les mises à jour simultanées
    temperatureUpdatePending = true;
    
    try {
        // Mettre à jour le canvas directement
        generateTemperatureTexture();
        
        // Utiliser requestAnimationFrame pour synchroniser avec le rendu
        requestAnimationFrame(() => {
            try {
                // Forcer la mise à jour du provider avec un nouveau blob URL
                const dataUrl = temperatureCanvas.toDataURL('image/png');
                const blob = dataURLtoBlob(dataUrl);
                const blobUrl = URL.createObjectURL(blob);
                
                // Remplacer le provider de manière transparente
                const layerIndex = viewer.imageryLayers.indexOf(temperatureLayer);
                if (layerIndex >= 0) {
                    const oldProvider = temperatureLayer.imageryProvider;
                    viewer.imageryLayers.remove(temperatureLayer, false);
                    
                    const newProvider = new Cesium.SingleTileImageryProvider({
                        url: blobUrl,
                        rectangle: Cesium.Rectangle.MAX_VALUE
                    });
                    temperatureLayer = viewer.imageryLayers.addImageryProvider(newProvider, layerIndex);
                    temperatureLayer.alpha = tempOpacity;
                    
                    // Nettoyer l'ancien blob après un court délai
                    setTimeout(() => {
                        if (oldProvider && oldProvider._url && oldProvider._url.startsWith('blob:')) {
                            URL.revokeObjectURL(oldProvider._url);
                        }
                        if (blobUrl) {
                            // Ne pas révoquer immédiatement, laisser Cesium l'utiliser
                            setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
                        }
                    }, 100);
                }
            } catch (error) {
                console.error('❌ Erreur mise à jour texture:', error);
            } finally {
                temperatureUpdatePending = false;
            }
        });
    } catch (error) {
        console.error('❌ Erreur mise à jour texture:', error);
        temperatureUpdatePending = false;
    }
}

// Fonction utilitaire pour convertir dataURL en Blob
function dataURLtoBlob(dataurl) {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
}

// ============================================================================
// PARTICULES VENT (STYLE WINDY)
// ============================================================================

function generateParticles(count) {
    // Valider le count pour éviter les erreurs
    const safeCount = Math.max(1, Math.min(count, 10000));
    const points = [];
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    const earthRadius = 6378137;
    const altitude = 10000;

    for (let i = 0; i < safeCount; i++) {
        const y = 1 - (i / Math.max(1, safeCount - 1)) * 2;
        const radius = Math.sqrt(Math.max(0, 1 - y * y));
        const theta = goldenAngle * i;
        const x = Math.cos(theta) * radius;
        const z = Math.sin(theta) * radius;

        const pos = new Cesium.Cartesian3(
            x * (earthRadius + altitude),
            y * (earthRadius + altitude),
            z * (earthRadius + altitude)
        );

        points.push({
            position: pos,
            previousPos: pos.clone(),
            trail: [], // Historique simplifié
            age: Math.random() * 100,
            maxAge: 100 + Math.random() * 50,
            speed: 0.5 + Math.random() * 0.5,
            velocity: new Cesium.Cartesian3(0, 0, 0)
        });
    }

    return points;
}

function computeWind(cartesian, t) {
    const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
    const lat = cartographic.latitude;
    const lon = cartographic.longitude;

    // Vecteurs tangents
    const radial = Cesium.Cartesian3.normalize(cartesian, new Cesium.Cartesian3());
    const north = new Cesium.Cartesian3(0, 0, 1);
    
    const ePhi = Cesium.Cartesian3.normalize(
        Cesium.Cartesian3.cross(
            Cesium.Cartesian3.cross(radial, north, new Cesium.Cartesian3()),
            radial,
            new Cesium.Cartesian3()
        ),
        new Cesium.Cartesian3()
    );
    
    const eTheta = Cesium.Cartesian3.normalize(
        Cesium.Cartesian3.cross(radial, ePhi, new Cesium.Cartesian3()),
        new Cesium.Cartesian3()
    );

    // Champ de vent analytique
    const band = Math.sin(2 * lat + 0.8 * Math.sin(0.1 * t)) * Math.cos(3 * lon);
    const jet = Math.sin(5 * lon + 0.2 * t) * 0.5 + 0.5;

    const vPhi = band * 0.3;
    const vTheta = jet * 0.5;

    const wind = new Cesium.Cartesian3();
    Cesium.Cartesian3.multiplyByScalar(ePhi, vPhi, wind);
    Cesium.Cartesian3.add(
        wind,
        Cesium.Cartesian3.multiplyByScalar(eTheta, vTheta, new Cesium.Cartesian3()),
        wind
    );

    return wind;
}

function updateParticles(dt) {
    if (!windEnabled) return;

    const earthRadius = 6378137;
    const altitude = 10000;
    const clampedDt = Math.min(dt, 0.1); // Limiter le delta time

    particles.forEach(particle => {
        // Calculer vent
        const wind = computeWind(particle.position, time);
        
        // Calculer vélocité avec interpolation pour mouvement fluide
        const targetVelocity = Cesium.Cartesian3.multiplyByScalar(
            wind,
            windSpeed * particle.speed * 50000,
            new Cesium.Cartesian3()
        );
        
        // Interpolation de la vélocité pour mouvement plus fluide
        Cesium.Cartesian3.lerp(
            particle.velocity,
            targetVelocity,
            0.1,
            particle.velocity
        );
        
        // Déplacer particule
        const movement = Cesium.Cartesian3.multiplyByScalar(
            particle.velocity,
            clampedDt,
            new Cesium.Cartesian3()
        );
        
        Cesium.Cartesian3.add(particle.position, movement, particle.position);

        // Reprojection sphérique
        const targetDist = earthRadius + altitude;
        Cesium.Cartesian3.multiplyByScalar(
            Cesium.Cartesian3.normalize(particle.position, new Cesium.Cartesian3()),
            targetDist,
            particle.position
        );

        // Traînée simplifiée (optionnel, peut être désactivé pour grandes quantités)
        // Ne pas stocker de traînée si trop de particules pour économiser la mémoire
        if (particles.length < 10000) {
            if (!particle.trail) {
                particle.trail = [];
            }
            const newPos = particle.position.clone();
            if (newPos) {
                particle.trail.push(newPos);
                // Limiter à 2 positions max
                if (particle.trail.length > 2) {
                    particle.trail.shift();
                }
            }
        } else {
            // Pour grandes quantités, pas de traînée pour économiser la mémoire
            particle.trail = null;
        }

        // Vieillissement et respawn
        particle.age += clampedDt;
        const shouldRespawn = particle.age > particle.maxAge || 
                             (particle.trail !== null && particle.trail.length === 0);
        if (shouldRespawn) {
            const newP = generateParticles(1)[0];
            particle.position = newP.position;
            particle.previousPos = newP.previousPos.clone();
            // Réinitialiser la traînée seulement si elle est activée
            if (particles.length < 10000) {
                particle.trail = [particle.position.clone()];
            } else {
                particle.trail = null;
            }
            particle.velocity = new Cesium.Cartesian3(0, 0, 0);
            particle.age = 0;
            particle.maxAge = 100 + Math.random() * 50;
            particle.speed = 0.5 + Math.random() * 0.5;
        }
    });
}

function renderParticles() {
    if (!particleCollection || !viewer || !viewer.scene) return;
    
    if (!windEnabled || !particles) return;
    
    // Vérifier que particles est un tableau valide
    if (!Array.isArray(particles) || particles.length === 0) return;

    // TOUJOURS nettoyer la collection en premier pour éviter l'accumulation
    try {
        if (particleCollection.length > 0) {
            particleCollection.removeAll();
        }
    } catch (error) {
        console.error('Erreur removeAll:', error);
        // Recréer la collection si nécessaire
        try {
            if (viewer.scene.primitives) {
                viewer.scene.primitives.remove(particleCollection);
            }
            particleCollection = new Cesium.PointPrimitiveCollection();
            if (viewer.scene.primitives) {
                viewer.scene.primitives.add(particleCollection);
            }
        } catch (recreateError) {
            console.error('Erreur recréation:', recreateError);
            return;
        }
    }

    // Rendre TOUTES les particules (pas de throttling, pas de LOD)
    // Si trop de particules, on les rend quand même mais avec un échantillonnage intelligent
    const totalParticles = particles.length;
    let pointsAdded = 0;
    const maxPoints = 20000; // Limite de sécurité augmentée

    // Si trop de particules, utiliser un échantillonnage uniforme
    const step = totalParticles > maxPoints ? Math.ceil(totalParticles / maxPoints) : 1;

    // Rendre les particules
    for (let i = 0; i < totalParticles && pointsAdded < maxPoints; i += step) {
        const particle = particles[i];
        
        if (!particle || !particle.position) continue;
        
        try {
            const pos = particle.position;
            if (!pos) continue;
            
            particleCollection.add({
                position: pos,
                pixelSize: 2.5,
                color: Cesium.Color.CYAN.withAlpha(0.9),
                outlineColor: Cesium.Color.WHITE,
                outlineWidth: 1
            });
            
            pointsAdded++;
        } catch (error) {
            // Ignorer les erreurs individuelles
            continue;
        }
    }
}

// ============================================================================
// ANIMATION
// ============================================================================

let lastTime = performance.now();
let lastTempUpdate = 0;

function animate() {
    try {
        const now = performance.now();
        const dt = Math.min((now - lastTime) / 1000, 0.1);
        lastTime = now;

        if (!viewer || !viewer.scene) {
            return;
        }

        time += dt;

        // Mettre à jour particules (chaque frame pour animation fluide)
        try {
            updateParticles(dt);
        } catch (updateError) {
            console.error('Erreur updateParticles:', updateError);
        }

        // Rendre les particules avec gestion d'erreur
        try {
            renderParticles();
        } catch (renderError) {
            console.error('Erreur renderParticles:', renderError);
            // Réinitialiser la collection en cas d'erreur
            if (particleCollection) {
                try {
                    viewer.scene.primitives.remove(particleCollection);
                    particleCollection = new Cesium.PointPrimitiveCollection();
                    viewer.scene.primitives.add(particleCollection);
                } catch (resetError) {
                    console.error('Erreur reset collection:', resetError);
                }
            }
        }

        // Mettre à jour température de manière fluide
        // Intervalle plus long pour éviter le clignotement, surtout avec beaucoup de particules
        // L'animation dans generateTemperatureTexture utilise time, donc elle reste fluide même avec moins de mises à jour
        const updateInterval = particles.length > 5000 ? 2.0 : 1.0; // Plus long si beaucoup de particules
        if (time - lastTempUpdate > updateInterval) {
            try {
                if (temperatureLayer) {
                    updateTemperatureTexture();
                } else {
                    addTemperatureLayer();
                }
            } catch (tempError) {
                console.error('Erreur température:', tempError);
            }
            lastTempUpdate = time;
        }

        requestAnimationFrame(animate);
    } catch (error) {
        console.error('Erreur dans animate:', error);
        // Continuer l'animation même en cas d'erreur
        requestAnimationFrame(animate);
    }
}

// ============================================================================
// CONTROLES UI
// ============================================================================

function setupControls() {
    document.getElementById('windToggle').addEventListener('change', (e) => {
        windEnabled = e.target.checked;
        renderParticles();
    });

    const particleSlider = document.getElementById('particleCount');
    const particleValue = document.getElementById('particleCountValue');
    particleSlider.addEventListener('input', (e) => {
        particleCount = parseInt(e.target.value);
        particleValue.textContent = particleCount;
        updateMaxRenderedParticles(); // Ajuster le rendu selon le nombre
        particles = generateParticles(particleCount);
        renderParticles();
    });

    const speedSlider = document.getElementById('windSpeed');
    const speedValue = document.getElementById('windSpeedValue');
    speedSlider.addEventListener('input', (e) => {
        windSpeed = parseFloat(e.target.value);
        speedValue.textContent = windSpeed.toFixed(1);
    });

    const tempSlider = document.getElementById('tempOpacity');
    const tempValue = document.getElementById('tempOpacityValue');
    tempSlider.addEventListener('input', (e) => {
        tempOpacity = parseFloat(e.target.value);
        tempValue.textContent = tempOpacity.toFixed(1);
        if (temperatureLayer) {
            temperatureLayer.alpha = tempOpacity;
            addTemperatureLayer();
        }
    });
}

// ============================================================================
// INITIALISATION
// ============================================================================

function init() {
    if (typeof Cesium === 'undefined') {
        alert('Erreur: CesiumJS non chargé');
        return;
    }

    console.log('🚀 Initialisation...');
    
    initCesium();
    
    setTimeout(() => {
        addTemperatureLayer();
        updateMaxRenderedParticles(); // Initialiser le rendu
        particles = generateParticles(particleCount);
        renderParticles();
        setupControls();
        animate();
        console.log('✅ Globe prêt!');
        console.log(`📊 ${particleCount} particules simulées, ${maxRenderedParticles} rendues`);
    }, 500);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
