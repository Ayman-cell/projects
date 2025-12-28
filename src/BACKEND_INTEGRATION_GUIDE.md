# 🚀 Guide d'Intégration Backend - Airboard Dashboard

## 📁 Architecture Actuelle

```
/
├── App.tsx                      # Routeur principal (navigation par state)
├── components/
│   ├── dashboard/              # ✅ NOUVEAU - Tous les composants Dashboard
│   │   ├── TimeFilterBar.tsx   # Barre de filtres (station, date, période)
│   │   ├── LeftPanel.tsx       # Panneau gauche (rose des vents, métriques)
│   │   ├── RightPanel.tsx      # Panneau droit (tableau + graphiques)
│   │   ├── HourlyTable.tsx     # Tableau des données horaires
│   │   ├── TimeSeriesCharts.tsx # Graphiques temporels
│   │   ├── WindRoseChart.tsx   # Rose des vents
│   │   ├── MetricCard.tsx      # Cartes de métriques
│   │   └── MapSection.tsx      # Carte des stations (modal)
│   ├── pages/
│   │   └── DashboardPage.tsx   # Page principale Dashboard
│   └── ...
└── styles/globals.css          # Styles adaptés au thème Airboard
```

---

## 🎯 Étapes pour l'Intégration Backend

### **Phase 1 : Configuration API (Recommandé : Supabase ou FastAPI)**

#### Option A : Supabase (Recommandé pour démarrage rapide)
```bash
npm install @supabase/supabase-js
```

**Créer `/lib/supabase.ts` :**
```typescript
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY

export const supabase = createClient(supabaseUrl, supabaseKey)

// Types pour vos données
export interface StationData {
  id: string
  name: string
  location: { lat: number; lng: number }
  status: 'active' | 'warning' | 'error'
}

export interface HourlyData {
  station_id: string
  timestamp: Date
  direction: number
  vitesse: number
  temperature: number
  humidite: number
  power: number
  scenario?: string
}
```

#### Option B : API REST (FastAPI/Express)
**Créer `/lib/api.ts` :**
```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = {
  // Récupérer les stations
  getStations: async () => {
    const response = await fetch(`${API_BASE_URL}/stations`)
    return response.json()
  },

  // Récupérer les données horaires
  getHourlyData: async (stationId: string, date: Date, period: 'day' | 'month' | 'year') => {
    const params = new URLSearchParams({
      station_id: stationId,
      date: date.toISOString(),
      period
    })
    const response = await fetch(`${API_BASE_URL}/hourly-data?${params}`)
    return response.json()
  },

  // Récupérer les données en temps réel
  getCurrentData: async (stationId: string) => {
    const response = await fetch(`${API_BASE_URL}/current/${stationId}`)
    return response.json()
  }
}
```

---

### **Phase 2 : Remplacer les Données Mock**

#### **1. TimeFilterBar.tsx** - Charger les stations depuis l'API
```typescript
// AVANT (mock data)
const stations = [
  { id: 'GP1', name: 'GP1 - Site Principal' },
  // ...
]

// APRÈS (API)
const [stations, setStations] = useState([])

useEffect(() => {
  api.getStations().then(setStations)
}, [])
```

#### **2. LeftPanel.tsx** - Données en temps réel
```typescript
// AVANT (mock data)
const currentData = {
  direction: 272,
  vitesse: 2.1,
  // ...
}

// APRÈS (API avec polling)
const [currentData, setCurrentData] = useState(null)

useEffect(() => {
  const fetchCurrent = () => {
    api.getCurrentData(selectedStation).then(setCurrentData)
  }
  
  fetchCurrent()
  const interval = setInterval(fetchCurrent, 30000) // Refresh toutes les 30s
  
  return () => clearInterval(interval)
}, [selectedStation])
```

#### **3. RightPanel.tsx** - Données historiques
```typescript
// AVANT (mock data)
const generateHourlyData = () => { /* ... */ }

// APRÈS (API)
const [hourlyData, setHourlyData] = useState([])
const [loading, setLoading] = useState(true)

useEffect(() => {
  setLoading(true)
  api.getHourlyData(selectedStation, selectedDate, selectedPeriod)
    .then(data => {
      setHourlyData(data)
      setLoading(false)
    })
}, [selectedStation, selectedDate, selectedPeriod])
```

---

### **Phase 3 : WebSockets pour Données en Temps Réel (Optionnel)**

**Créer `/lib/websocket.ts` :**
```typescript
export class StationWebSocket {
  private ws: WebSocket | null = null
  
  connect(stationId: string, onData: (data: any) => void) {
    this.ws = new WebSocket(`ws://localhost:8000/ws/${stationId}`)
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      onData(data)
    }
  }
  
  disconnect() {
    this.ws?.close()
  }
}
```

**Utilisation dans LeftPanel.tsx :**
```typescript
useEffect(() => {
  const wsClient = new StationWebSocket()
  wsClient.connect(selectedStation, (data) => {
    setCurrentData(data)
  })
  
  return () => wsClient.disconnect()
}, [selectedStation])
```

---

### **Phase 4 : State Management (Si nécessaire)**

Pour une gestion d'état complexe, utiliser **Zustand** (léger et simple) :

```bash
npm install zustand
```

**Créer `/stores/dashboardStore.ts` :**
```typescript
import { create } from 'zustand'

interface DashboardState {
  selectedStation: string
  selectedDate: Date
  selectedPeriod: 'day' | 'month' | 'year'
  currentData: any
  hourlyData: any[]
  
  setStation: (station: string) => void
  setDate: (date: Date) => void
  setPeriod: (period: 'day' | 'month' | 'year') => void
  updateCurrentData: (data: any) => void
  updateHourlyData: (data: any[]) => void
}

export const useDashboardStore = create<DashboardState>((set) => ({
  selectedStation: 'GP1',
  selectedDate: new Date(),
  selectedPeriod: 'day',
  currentData: null,
  hourlyData: [],
  
  setStation: (station) => set({ selectedStation: station }),
  setDate: (date) => set({ selectedDate: date }),
  setPeriod: (period) => set({ selectedPeriod: period }),
  updateCurrentData: (data) => set({ currentData: data }),
  updateHourlyData: (data) => set({ hourlyData: data }),
}))
```

**Utilisation dans DashboardPage.tsx :**
```typescript
import { useDashboardStore } from '../../stores/dashboardStore'

export default function DashboardPage({ onBack }: DashboardPageProps) {
  const { selectedStation, setStation, selectedDate, setDate } = useDashboardStore()
  
  // Plus besoin de useState ici !
}
```

---

### **Phase 5 : Variables d'Environnement**

**Créer `.env` :**
```env
# API Configuration
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000

# Supabase (si utilisé)
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key

# Autres
VITE_REFRESH_INTERVAL=30000
```

---

## 📊 Structure Backend Recommandée (FastAPI)

```python
# main.py
from fastapi import FastAPI, WebSocket
from datetime import datetime, timedelta
import pandas as pd

app = FastAPI()

@app.get("/stations")
async def get_stations():
    return [
        {"id": "GP1", "name": "GP1 - Site Principal", "lat": 32.29, "lng": -9.23},
        {"id": "GP2", "name": "GP2 - Zone Nord", "lat": 32.30, "lng": -9.22},
        # ...
    ]

@app.get("/current/{station_id}")
async def get_current_data(station_id: str):
    # Récupérer les dernières données depuis votre base de données
    return {
        "station_id": station_id,
        "timestamp": datetime.now(),
        "direction": 272,
        "vitesse": 2.1,
        "temperature": 21.8,
        "humidite": 69,
        "power": 225
    }

@app.get("/hourly-data")
async def get_hourly_data(
    station_id: str, 
    date: datetime, 
    period: str = "day"
):
    # Récupérer les données horaires depuis votre base
    # Exemple avec pandas
    df = pd.read_sql(
        f"SELECT * FROM hourly_data WHERE station_id = '{station_id}' AND date = '{date}'",
        con=db_connection
    )
    return df.to_dict(orient='records')

@app.websocket("/ws/{station_id}")
async def websocket_endpoint(websocket: WebSocket, station_id: str):
    await websocket.accept()
    while True:
        # Envoyer des données en temps réel toutes les 5 secondes
        data = get_realtime_data(station_id)
        await websocket.send_json(data)
        await asyncio.sleep(5)
```

---

## 🔐 Sécurité et Authentification (Future)

Quand tu auras besoin d'authentification :

1. **Ajouter un Context d'authentification** :
```typescript
// contexts/AuthContext.tsx
export const useAuth = () => {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(null)
  
  const login = async (email, password) => { /* ... */ }
  const logout = () => { /* ... */ }
  
  return { user, token, login, logout }
}
```

2. **Protéger les routes** :
```typescript
// App.tsx
{currentPage === 'dashboard' && (
  user ? <DashboardPage onBack={handleBack} /> : <LoginPage />
)}
```

---

## 📝 Checklist d'Intégration

- [ ] Installer les dépendances (`@supabase/supabase-js` ou axios)
- [ ] Créer `/lib/api.ts` ou `/lib/supabase.ts`
- [ ] Configurer les variables d'environnement (`.env`)
- [ ] Remplacer les données mock dans `LeftPanel.tsx`
- [ ] Remplacer les données mock dans `RightPanel.tsx`
- [ ] Ajouter le polling/WebSocket pour les données live
- [ ] Tester avec votre backend
- [ ] (Optionnel) Ajouter Zustand pour le state management
- [ ] (Optionnel) Ajouter l'authentification

---

## 🎨 Points d'Attention

1. **Calcul des scénarios** : Le calcul est actuellement fait côté frontend dans `HourlyTable.tsx`. Tu peux le garder ou le déplacer vers le backend.

2. **Format des dates** : Utilise `date-fns` pour la cohérence (déjà installé).

3. **Loading states** : Ajoute des skeletons ou spinners pendant le chargement :
```typescript
{loading ? <Skeleton className="h-40" /> : <TimeSeriesCharts data={data} />}
```

4. **Error handling** : Toujours gérer les erreurs API :
```typescript
try {
  const data = await api.getHourlyData(...)
  setHourlyData(data)
} catch (error) {
  toast.error("Erreur lors du chargement des données")
}
```

---

## 🚀 Pour Démarrer

1. Teste d'abord que le Dashboard fonctionne avec les données mock
2. Développe ton backend en parallèle
3. Intègre progressivement, composant par composant
4. Commence par `LeftPanel.tsx` (données en temps réel simples)
5. Puis `RightPanel.tsx` (données historiques)

**Bonne chance ! 🎉**
