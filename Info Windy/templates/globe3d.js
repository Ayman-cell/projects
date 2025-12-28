/**
 * Globe3D.js - Module pour afficher un globe 3D météo interactif
 * Compatible Three.js r128 (sans OrbitControls)
 * 
 * Fonctionnalités :
 * - Globe texturé avec données météo (temp/humidité)
 * - Particules 3D animées selon champs de vent (u, v)
 * - Contrôles manuels (rotation, zoom)
 * - Raycasting pour hover info
 * - Support timeline de prévisions
 */

// ========================================================
// CLASSE PRINCIPALE : Globe3D
// ========================================================

class Globe3D {
  constructor(containerId, apiEndpoints) {
    this.containerId = containerId;
    this.apiEndpoints = apiEndpoints || {
      fields: '/api/fields',
      forecast: '/api/forecast'
    };

    // Three.js objets
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.globe = null;
    this.globeTexture = null;
    this.textureCanvas = null;
    this.textureCtx = null;

    // Système de particules
    this.particleSystem = null;
    this.particleCount = 3000; // Ajusté selon performance

    // Contrôles
    this.controls = null;

    // Données météo
    this.weatherData = null;
    this.gridConfig = null;
    this.currentLayer = 'temp'; // 'temp' | 'rh'
    this.isForecastMode = false;
    this.forecastHour = 0;
    this.forecastModel = 'auto';

    // État animation
    this.isAnimating = false;
    this.animationId = null;

    // Raycaster pour hover
    this.raycaster = null;
    this.mouse = new THREE.Vector2();
    this.hoverInfo = null;
    this.hoverTimeout = null;

    // Dimensions conteneur
    this.width = 0;
    this.height = 0;

    console.log('[Globe3D] Instance créée');
  }

  /**
   * Initialise la scène Three.js, caméra, renderer, lumières
   */
  init() {
    // Vérifier que Three.js est chargé
    if (typeof THREE === 'undefined') {
      console.error('[Globe3D] Three.js n\'est pas chargé !');
      alert('Erreur : Three.js n\'est pas chargé. Vérifiez votre connexion internet.');
      return;
    }

    const container = document.getElementById(this.containerId);
    if (!container) {
      console.error('[Globe3D] Conteneur non trouvé:', this.containerId);
      return;
    }

    // S'assurer que le conteneur est visible pour obtenir ses dimensions
    const wasHidden = container.style.display === 'none' || !container.classList.contains('active');
    if (wasHidden) {
      container.style.display = 'block';
      container.classList.add('active');
    }

    // Utiliser requestAnimationFrame pour s'assurer que le DOM est prêt
    requestAnimationFrame(() => {
      this.width = container.clientWidth || window.innerWidth;
      this.height = container.clientHeight || window.innerHeight;

      if (this.width === 0 || this.height === 0) {
        console.warn('[Globe3D] Conteneur a une taille 0, utilisation des dimensions de la fenêtre');
        this.width = window.innerWidth;
        this.height = window.innerHeight;
      }

      console.log('[Globe3D] Dimensions conteneur:', this.width, 'x', this.height);

      // Scène
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0x05060a);

      // Caméra (PerspectiveCamera)
      this.camera = new THREE.PerspectiveCamera(
        45, // FOV
        this.width / this.height,
        0.1, // near
        1000 // far
      );
      this.camera.position.set(0, 0, 3.5); // Distance initiale du globe
      this.camera.lookAt(0, 0, 0);

      // Renderer
      this.renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true 
      });
      this.renderer.setSize(this.width, this.height);
      this.renderer.setPixelRatio(window.devicePixelRatio);
      container.appendChild(this.renderer.domElement);

      // Lumières
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      this.scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(5, 5, 5);
      this.scene.add(directionalLight);

      // Créer le globe
      this.createGlobe();

      // Créer le système de particules
      this.createParticles();

      // Setup contrôles
      this.setupControls();

      // Setup raycaster
      this.setupRaycaster();

      // Gestion redimensionnement
      window.addEventListener('resize', () => this.resize());

      // Rendu initial pour s'assurer que quelque chose s'affiche
      this.renderer.render(this.scene, this.camera);

      console.log('[Globe3D] Initialisation terminée');
    });
  }

  /**
   * Crée la sphère terrestre avec géométrie et matériau
   */
  createGlobe() {
    const geometry = new THREE.SphereGeometry(1, 64, 64);
    
    // Canvas pour texture dynamique (512x256 ratio 2:1)
    this.textureCanvas = document.createElement('canvas');
    this.textureCanvas.width = 512;
    this.textureCanvas.height = 256;
    this.textureCtx = this.textureCanvas.getContext('2d');

    // Texture initiale (dégradé bleu-vert pour visibilité)
    const gradient = this.textureCtx.createLinearGradient(0, 0, 512, 256);
    gradient.addColorStop(0, '#001a33');
    gradient.addColorStop(0.5, '#003366');
    gradient.addColorStop(1, '#004d99');
    this.textureCtx.fillStyle = gradient;
    this.textureCtx.fillRect(0, 0, 512, 256);
    
    // Ajouter un texte "Chargement..." temporaire
    this.textureCtx.fillStyle = '#ffffff';
    this.textureCtx.font = '20px Arial';
    this.textureCtx.textAlign = 'center';
    this.textureCtx.fillText('Chargement...', 256, 128);

    this.globeTexture = new THREE.CanvasTexture(this.textureCanvas);
    this.globeTexture.needsUpdate = true;

    const material = new THREE.MeshPhongMaterial({
      map: this.globeTexture,
      transparent: false,
      shininess: 30
    });

    this.globe = new THREE.Mesh(geometry, material);
    this.scene.add(this.globe);

    console.log('[Globe3D] Globe créé');
  }

  /**
   * Crée le système de particules 3D
   */
  createParticles() {
    // Détecter mobile pour réduire nombre de particules
    const isMobile = window.innerWidth < 768;
    this.particleCount = isMobile ? 1500 : 3000;

    // BufferGeometry pour performance
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(this.particleCount * 3);
    const colors = new Float32Array(this.particleCount * 3);

    // Initialiser positions aléatoires sur la sphère
    // Convention : theta = longitude [0, 2π], phi = latitude [-π/2, π/2]
    for (let i = 0; i < this.particleCount; i++) {
      const theta = Math.random() * Math.PI * 2; // Longitude [0, 2π]
      const phi = Math.asin(2 * Math.random() - 1); // Latitude [-π/2, π/2] (distribution uniforme)
      const radius = 1.01; // Juste au-dessus de la surface

      const idx = i * 3;
      // Conversion sphérique → cartésien
      positions[idx] = radius * Math.cos(phi) * Math.cos(theta);
      positions[idx + 1] = radius * Math.sin(phi);
      positions[idx + 2] = radius * Math.cos(phi) * Math.sin(theta);

      // Couleur blanche
      colors[idx] = 1;
      colors[idx + 1] = 1;
      colors[idx + 2] = 1;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Matériau pour particules
    const material = new THREE.PointsMaterial({
      size: 0.02,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending
    });

    this.particleSystem = new THREE.Points(geometry, material);
    this.scene.add(this.particleSystem);

    // Stocker données particules (theta, phi, age, etc.)
    this.particleData = [];
    for (let i = 0; i < this.particleCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.asin(2 * Math.random() - 1); // Distribution uniforme sur la sphère
      this.particleData.push({
        theta: theta,
        phi: phi,
        age: Math.random() * 200,
        maxAge: 200 + Math.random() * 100
      });
    }

    console.log('[Globe3D] Système de particules créé:', this.particleCount);
  }

  /**
   * Génère la texture du globe depuis les données météo
   * @param {Object} weatherData - Données météo avec temp/rh et grid
   * @param {string} layer - 'temp' ou 'rh'
   */
  updateTexture(weatherData, layer = 'temp') {
    if (!weatherData || !this.textureCtx) return;

    this.weatherData = weatherData;
    this.currentLayer = layer;
    this.gridConfig = weatherData.grid;

    const field = layer === 'temp' ? weatherData.temp : weatherData.rh;
    const grid = weatherData.grid;

    // Calculer min/max pour normalisation
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < field.length; i++) {
      for (let j = 0; j < field[i].length; j++) {
        const val = field[i][j];
        if (isFinite(val)) {
          min = Math.min(min, val);
          max = Math.max(max, val);
        }
      }
    }

    const range = max - min || 1;

    // Générer texture pixel par pixel
    const width = this.textureCanvas.width;
    const height = this.textureCanvas.height;
    const imgData = this.textureCtx.createImageData(width, height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        // Convertir (x, y) → (lat, lon)
        const lon = (x / width - 0.5) * 360;
        const lat = (0.5 - y / height) * 180;

        // Interpoler valeur à (lat, lon)
        const val = this.bilinearInterp(lat, lon, field, grid);

        // Mapper valeur → couleur
        let r, g, b;
        if (val === null || !isFinite(val)) {
          r = g = b = 0; // Noir si hors zone
        } else {
          const tNorm = (val - min) / range;
          if (layer === 'temp') {
            // Palette température : bleu → cyan → vert → jaune → rouge
            if (tNorm < 0.25) {
              const t = tNorm / 0.25;
              r = 0;
              g = Math.floor(255 * t);
              b = 255;
            } else if (tNorm < 0.5) {
              const t = (tNorm - 0.25) / 0.25;
              r = 0;
              g = 255;
              b = Math.floor(255 * (1 - t));
            } else if (tNorm < 0.75) {
              const t = (tNorm - 0.5) / 0.25;
              r = Math.floor(255 * t);
              g = 255;
              b = 0;
            } else {
              const t = (tNorm - 0.75) / 0.25;
              r = 255;
              g = Math.floor(255 * (1 - t));
              b = 0;
            }
          } else {
            // Palette humidité : bleu foncé → bleu clair → violet → blanc
            if (tNorm < 0.5) {
              const t = tNorm / 0.5;
              r = Math.floor(27 * t);
              g = Math.floor(27 + (107 - 27) * t);
              b = 255;
            } else {
              const t = (tNorm - 0.5) / 0.5;
              r = Math.floor(27 + (255 - 27) * t);
              g = Math.floor(107 + (255 - 107) * t);
              b = 255;
            }
          }
        }

        const idx = (y * width + x) * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }

    this.textureCtx.putImageData(imgData, 0, 0);
    this.globeTexture.needsUpdate = true;

    // Mettre à jour légende
    this.updateLegend(min, max, layer);

    console.log('[Globe3D] Texture mise à jour:', layer, `[${min.toFixed(1)}, ${max.toFixed(1)}]`);
  }

  /**
   * Interpolation bilinéaire (lat, lon) → valeur dans grille 2D
   */
  bilinearInterp(lat, lon, field, grid) {
    // Gérer wrap longitude ±180°
    while (lon < -180) lon += 360;
    while (lon > 180) lon -= 360;

    // Clamp latitude ±90°
    lat = Math.max(-90, Math.min(90, lat));

    const ny = grid.ny;
    const nx = grid.nx;

    // Convertir (lat, lon) → coordonnées normalisées [0, 1]
    const ty = (lat - grid.lat_min) / (grid.lat_max - grid.lat_min);
    const tx = (lon - grid.lon_min) / (grid.lon_max - grid.lon_min);

    // Vérifier si dans la grille
    if (ty < 0 || ty > 1 || tx < 0 || tx > 1) {
      return null;
    }

    // Convertir en indices
    const y = ty * (ny - 1);
    const x = tx * (nx - 1);

    const i0 = Math.floor(y);
    const j0 = Math.floor(x);
    const i1 = Math.min(i0 + 1, ny - 1);
    const j1 = Math.min(j0 + 1, nx - 1);

    const sy = y - i0;
    const sx = x - j0;

    // Interpolation bilinéaire
    const f00 = field[i0][j0];
    const f10 = field[i0][j1];
    const f01 = field[i1][j0];
    const f11 = field[i1][j1];

    const f0 = f00 * (1 - sx) + f10 * sx;
    const f1 = f01 * (1 - sx) + f11 * sx;
    return f0 * (1 - sy) + f1 * sy;
  }

  /**
   * Met à jour les positions des particules selon champs de vent (u, v)
   */
  animateParticles() {
    if (!this.weatherData || !this.particleSystem || !this.gridConfig) return;

    const uField = this.weatherData.u;
    const vField = this.weatherData.v;
    const grid = this.gridConfig;

    const positions = this.particleSystem.geometry.attributes.position.array;
    const dt = 0.016; // ~60 FPS
    const speedScale = 0.0001; // Facteur d'échelle pour vitesse
    const radius = 1.01;

    for (let i = 0; i < this.particleCount; i++) {
      const p = this.particleData[i];

      // Convertir (theta, phi) → (lat, lon)
      const lat = (p.phi * 180) / Math.PI;
      const lon = (p.theta * 180) / Math.PI;

      // Échantillonner u, v à cette position
      const u = this.bilinearInterp(lat, lon, uField, grid);
      const v = this.bilinearInterp(lat, lon, vField, grid);

      // Si hors zone ou données invalides, respawn
      if (u === null || v === null || !isFinite(u) || !isFinite(v)) {
        this.respawnParticle(i);
        continue;
      }

      // Convertir (u, v) en déplacement sphérique
      // u = vitesse vers l'Est (m/s), v = vitesse vers le Nord (m/s)
      // Conversion en radians par seconde
      // Éviter division par zéro près des pôles
      const cosPhi = Math.max(0.01, Math.abs(Math.cos(p.phi)));
      const dTheta = (u * speedScale * dt) / (radius * cosPhi);
      const dPhi = (v * speedScale * dt) / radius;

      // Mettre à jour angles
      p.theta += dTheta;
      p.phi = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, p.phi + dPhi));

      // Recalculer position cartésienne
      const idx = i * 3;
      positions[idx] = radius * Math.cos(p.phi) * Math.cos(p.theta);
      positions[idx + 1] = radius * Math.sin(p.phi);
      positions[idx + 2] = radius * Math.cos(p.phi) * Math.sin(p.theta);

      // Vieillissement
      p.age += 1;
      if (p.age > p.maxAge) {
        this.respawnParticle(i);
      }
    }

    this.particleSystem.geometry.attributes.position.needsUpdate = true;
  }

  /**
   * Réinitialise une particule à une position aléatoire
   */
  respawnParticle(index) {
    const p = this.particleData[index];
    p.theta = Math.random() * Math.PI * 2;
    p.phi = Math.asin(2 * Math.random() - 1); // Distribution uniforme sur la sphère
    p.age = 0;
    p.maxAge = 200 + Math.random() * 100;

    const radius = 1.01;
    const idx = index * 3;
    const positions = this.particleSystem.geometry.attributes.position.array;
    positions[idx] = radius * Math.cos(p.phi) * Math.cos(p.theta);
    positions[idx + 1] = radius * Math.sin(p.phi);
    positions[idx + 2] = radius * Math.cos(p.phi) * Math.sin(p.theta);
  }

  /**
   * Configure les contrôles manuels (rotation, zoom) sans OrbitControls
   */
  setupControls() {
    this.controls = new ManualControls(this.camera, this.renderer.domElement, this.globe);
    console.log('[Globe3D] Contrôles configurés');
  }

  /**
   * Configure le raycasting pour hover info
   */
  setupRaycaster() {
    this.raycaster = new THREE.Raycaster();

    // Utiliser le même élément hover-info que le mode 2D
    this.hoverInfo = document.getElementById('hover-info');
    if (!this.hoverInfo) {
      // Créer l'élément si absent
      this.hoverInfo = document.createElement('div');
      this.hoverInfo.id = 'hover-info';
      this.hoverInfo.style.cssText = 'position:absolute; z-index:700; background:linear-gradient(180deg, rgba(6,8,15,0.86), rgba(8,10,20,0.86)); color:#fff; padding:10px; border-radius:8px; font-size:12px; pointer-events:none; display:none; max-width:240px; box-shadow:0 10px 24px rgba(2,6,20,0.6);';
      document.body.appendChild(this.hoverInfo);
    }

    // Événements souris
    this.renderer.domElement.addEventListener('mousemove', (e) => {
      this.onMouseMove(e);
    });

    this.renderer.domElement.addEventListener('mouseout', () => {
      if (this.hoverInfo) {
        this.hoverInfo.style.display = 'none';
      }
      if (this.hoverTimeout) {
        clearTimeout(this.hoverTimeout);
        this.hoverTimeout = null;
      }
    });
  }

  /**
   * Gestionnaire mousemove pour raycasting
   */
  onMouseMove(event) {
    if (!this.globe || !this.weatherData) return;

    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Raycast
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObject(this.globe);

    if (intersects.length > 0) {
      const point = intersects[0].point;
      
      // Normaliser le point (il est déjà sur une sphère de rayon 1)
      const radius = Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
      const normalizedPoint = new THREE.Vector3(
        point.x / radius,
        point.y / radius,
        point.z / radius
      );
      
      // Convertir point 3D normalisé → (lat, lon)
      // Latitude : angle depuis l'équateur (y = sin(lat))
      const lat = Math.asin(Math.max(-1, Math.min(1, normalizedPoint.y))) * (180 / Math.PI);
      // Longitude : angle dans le plan XY (x = cos(lat)*cos(lon), z = cos(lat)*sin(lon))
      const lon = Math.atan2(normalizedPoint.z, normalizedPoint.x) * (180 / Math.PI);

      // Afficher info après délai
      if (this.hoverTimeout) clearTimeout(this.hoverTimeout);
      this.hoverTimeout = setTimeout(() => {
        this.showHoverInfo(lat, lon, event.clientX, event.clientY);
      }, 300);
    } else {
      if (this.hoverInfo) {
        this.hoverInfo.style.display = 'none';
      }
    }
  }

  /**
   * Affiche les infos au survol
   */
  showHoverInfo(lat, lon, screenX, screenY) {
    if (!this.weatherData || !this.hoverInfo) return;

    const grid = this.gridConfig;
    const temp = this.bilinearInterp(lat, lon, this.weatherData.temp, grid);
    const rh = this.bilinearInterp(lat, lon, this.weatherData.rh, grid);
    const u = this.bilinearInterp(lat, lon, this.weatherData.u, grid);
    const v = this.bilinearInterp(lat, lon, this.weatherData.v, grid);

    if (temp === null || u === null || v === null) return;

    const speed = Math.sqrt(u * u + v * v);
    const dirRad = Math.atan2(-u, -v);
    const dirDeg = (dirRad * 180 / Math.PI + 360) % 360;

    this.hoverInfo.innerHTML = `
      <div style="font-weight: 700; margin-bottom: 4px;">${lat.toFixed(4)}°, ${lon.toFixed(4)}°</div>
      <div>Temp : ${temp.toFixed(1)} °C</div>
      <div>Humidité : ${rh !== null ? rh.toFixed(1) + ' %' : '--'}</div>
      <div>Vent : ${speed.toFixed(2)} m/s</div>
      <div>Direction : ${dirDeg.toFixed(0)}°</div>
    `;

    this.hoverInfo.style.left = (screenX + 15) + 'px';
    this.hoverInfo.style.top = (screenY + 15) + 'px';
    this.hoverInfo.style.display = 'block';
  }

  /**
   * Met à jour la légende
   */
  updateLegend(min, max, layer) {
    const legendTitle = document.getElementById('legend-title');
    const legendMin = document.getElementById('tmin');
    const legendMax = document.getElementById('tmax');

    if (legendTitle) {
      legendTitle.textContent = layer === 'temp' ? 'Température (°C)' : 'Humidité relative (%)';
    }
    if (legendMin) {
      legendMin.textContent = layer === 'temp' ? `${min.toFixed(1)} °C` : `${min.toFixed(1)} %`;
    }
    if (legendMax) {
      legendMax.textContent = layer === 'temp' ? `${max.toFixed(1)} °C` : `${max.toFixed(1)} %`;
    }
  }

  /**
   * Boucle d'animation principale
   */
  animate() {
    if (this.isAnimating) return;
    this.isAnimating = true;

    const loop = () => {
      if (!this.isAnimating) return;

      // Mettre à jour particules si données disponibles
      if (this.weatherData) {
        this.animateParticles();
      }

      // Rotation automatique légère (optionnel)
      // this.globe.rotation.y += 0.001;

      // Rendu
      if (this.renderer && this.scene && this.camera) {
        this.renderer.render(this.scene, this.camera);
      }

      this.animationId = requestAnimationFrame(loop);
    };

    loop();
    console.log('[Globe3D] Animation démarrée');
  }

  /**
   * Arrête l'animation
   */
  stopAnimation() {
    this.isAnimating = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
    console.log('[Globe3D] Animation arrêtée');
  }

  /**
   * Récupère les données depuis l'API et met à jour le globe
   */
  async fetchAndUpdate(endpoint, layer = 'temp') {
    try {
      console.log('[Globe3D] Fetch:', endpoint);
      const resp = await fetch(endpoint);
      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({ error: 'Erreur serveur', message: `HTTP ${resp.status}` }));
        console.error('[Globe3D] Erreur HTTP:', errorData);
        alert(`Erreur lors du chargement des données: ${errorData.error || 'Erreur inconnue'}\n${errorData.message || ''}`);
        return null;
      }
      
      const data = await resp.json();
      
      // Vérifier si c'est une réponse d'erreur
      if (data.status === 'error') {
        console.error('[Globe3D] Erreur dans les données:', data);
        alert(`Erreur: ${data.error || 'Erreur inconnue'}\n${data.message || ''}`);
        return null;
      }
      
      this.updateTexture(data, layer);
      
      // Réinitialiser particules
      if (this.particleData) {
        for (let i = 0; i < this.particleCount; i++) {
          this.respawnParticle(i);
        }
      }

      return data;
    } catch (err) {
      console.error('[Globe3D] Erreur fetch:', err);
      alert(`Erreur lors du chargement des données: ${err.message || 'Erreur inconnue'}`);
      return null;
    }
  }

  /**
   * Change le layer affiché (temp/rh)
   */
  setLayer(layer) {
    if (this.weatherData) {
      this.updateTexture(this.weatherData, layer);
    }
  }

  /**
   * Gestion redimensionnement fenêtre
   */
  resize() {
    const container = document.getElementById(this.containerId);
    if (!container) return;

    this.width = container.clientWidth;
    this.height = container.clientHeight;

    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height);
  }

  /**
   * Nettoie les ressources (pour retour mode 2D)
   */
  destroy() {
    this.stopAnimation();

    if (this.controls) {
      this.controls.dispose();
    }

    // Nettoyer Three.js
    if (this.particleSystem) {
      this.scene.remove(this.particleSystem);
      this.particleSystem.geometry.dispose();
      this.particleSystem.material.dispose();
    }

    if (this.globe) {
      this.scene.remove(this.globe);
      this.globe.geometry.dispose();
      this.globe.material.dispose();
    }

    if (this.globeTexture) {
      this.globeTexture.dispose();
    }

    if (this.renderer) {
      this.renderer.dispose();
      const container = document.getElementById(this.containerId);
      if (container && this.renderer.domElement.parentNode) {
        container.removeChild(this.renderer.domElement);
      }
    }

    window.removeEventListener('resize', () => this.resize());

    console.log('[Globe3D] Ressources nettoyées');
  }
}

// ========================================================
// CLASSE : ManualControls (sans OrbitControls)
// ========================================================

class ManualControls {
  constructor(camera, domElement, target = null) {
    this.camera = camera;
    this.domElement = domElement;
    this.target = target || new THREE.Vector3(0, 0, 0);

    // État rotation
    this.isDragging = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;
    this.rotationX = 0; // Rotation verticale (inclinaison)
    this.rotationY = 0; // Rotation horizontale

    // État zoom
    this.zoom = 3.5;
    this.minZoom = 1.5;
    this.maxZoom = 8;

    // Touch
    this.touches = [];
    this.lastTouchDistance = 0;

    this.setupEvents();
  }

  setupEvents() {
    // Mouse
    this.domElement.addEventListener('mousedown', (e) => this.onMouseDown(e));
    this.domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.domElement.addEventListener('mouseup', () => this.onMouseUp());
    this.domElement.addEventListener('wheel', (e) => this.onWheel(e));

    // Touch
    this.domElement.addEventListener('touchstart', (e) => this.onTouchStart(e));
    this.domElement.addEventListener('touchmove', (e) => this.onTouchMove(e));
    this.domElement.addEventListener('touchend', (e) => this.onTouchEnd(e));
  }

  onMouseDown(e) {
    this.isDragging = true;
    this.lastMouseX = e.clientX;
    this.lastMouseY = e.clientY;
    this.domElement.style.cursor = 'grabbing';
  }

  onMouseMove(e) {
    if (!this.isDragging) return;

    const dx = e.clientX - this.lastMouseX;
    const dy = e.clientY - this.lastMouseY;

    // Rotation horizontale
    this.rotationY += dx * 0.01;

    // Rotation verticale (limiter entre -80° et +80°)
    this.rotationX += dy * 0.01;
    this.rotationX = Math.max(-Math.PI / 2.25, Math.min(Math.PI / 2.25, this.rotationX));

    this.updateCamera();
    this.lastMouseX = e.clientX;
    this.lastMouseY = e.clientY;
  }

  onMouseUp() {
    this.isDragging = false;
    this.domElement.style.cursor = 'grab';
  }

  onWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 1.1 : 0.9;
    this.zoom *= delta;
    this.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.zoom));
    this.updateCamera();
  }

  onTouchStart(e) {
    e.preventDefault();
    this.touches = Array.from(e.touches);
    if (this.touches.length === 2) {
      this.lastTouchDistance = this.getTouchDistance(this.touches[0], this.touches[1]);
    }
  }

  onTouchMove(e) {
    e.preventDefault();
    this.touches = Array.from(e.touches);

    if (this.touches.length === 1) {
      // Rotation avec un doigt
      const touch = this.touches[0];
      const rect = this.domElement.getBoundingClientRect();
      const x = touch.clientX - rect.left;
      const y = touch.clientY - rect.top;

      if (this.lastMouseX !== 0 || this.lastMouseY !== 0) {
        const dx = x - this.lastMouseX;
        const dy = y - this.lastMouseY;
        this.rotationY += dx * 0.01;
        this.rotationX += dy * 0.01;
        this.rotationX = Math.max(-Math.PI / 2.25, Math.min(Math.PI / 2.25, this.rotationX));
      }

      this.lastMouseX = x;
      this.lastMouseY = y;
    } else if (this.touches.length === 2) {
      // Zoom avec pincement
      const distance = this.getTouchDistance(this.touches[0], this.touches[1]);
      if (this.lastTouchDistance > 0) {
        const delta = distance / this.lastTouchDistance;
        this.zoom *= delta;
        this.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.zoom));
      }
      this.lastTouchDistance = distance;
    }

    this.updateCamera();
  }

  onTouchEnd(e) {
    e.preventDefault();
    this.touches = Array.from(e.touches);
    if (this.touches.length === 0) {
      this.lastMouseX = 0;
      this.lastMouseY = 0;
      this.lastTouchDistance = 0;
    }
  }

  getTouchDistance(touch1, touch2) {
    const dx = touch1.clientX - touch2.clientX;
    const dy = touch1.clientY - touch2.clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  updateCamera() {
    // Position caméra en coordonnées sphériques
    const x = this.zoom * Math.cos(this.rotationX) * Math.sin(this.rotationY);
    const y = this.zoom * Math.sin(this.rotationX);
    const z = this.zoom * Math.cos(this.rotationX) * Math.cos(this.rotationY);

    this.camera.position.set(x, y, z);
    this.camera.lookAt(this.target);
  }

  dispose() {
    // Nettoyage des événements si nécessaire
    this.domElement.style.cursor = 'default';
  }
}

