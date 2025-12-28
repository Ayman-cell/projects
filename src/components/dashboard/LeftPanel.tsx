import React, { useEffect, useState } from 'react';
import { WindRoseChart } from './WindRoseChart';
import { MetricCard } from './MetricCard';
import { MapPin } from 'lucide-react';
import { format } from 'date-fns';
import { fr } from 'date-fns/locale';
import { ScrollArea } from '../ui/scroll-area';
import { useDataDir } from '../../contexts/DataDirContext';

interface LeftPanelProps {
  selectedStation: string;
  selectedDate: Date;
  selectedHour?: string;
  isLive?: boolean;
  onNavigateToMap?: () => void;
}

// Function to calculate scenario based on wind speed and direction
// Selon le tableau officiel des scénarios
const calculateScenario = (vitesse: number, direction: number): string | null => {
  // S1: V ≥ 5 m/s (pas de condition de direction)
  if (vitesse >= 5) return 'S1';
  
  // S3b: V ≤ 2 m/s et (WD > 156 and WD <= 203) - priorité sur les autres
  if (vitesse <= 2 && direction > 156 && direction <= 203) return 'S3b';
  
  // S4: V < 0,5 m/s et (WD > 90 and WD <= 293)
  if (vitesse < 0.5 && direction > 90 && direction <= 293) return 'S4';
  
  // S3: V < 0,5 m/s et [(WD > 293 and WD <= 360) or (WD >= 0 and WD <= 90)]
  if (vitesse < 0.5 && ((direction > 293 && direction <= 360) || (direction >= 0 && direction <= 90))) return 'S3';
  
  // S2b: V ≤ 1 m/s et (WD > 90 and WD <= 293)
  if (vitesse <= 1 && direction > 90 && direction <= 293) return 'S2b';
  
  // S2: 1 < V < 4 m/s et [(WD > 293 and WD <= 360) or (WD >= 0 and WD <= 90)]
  if (vitesse > 1 && vitesse < 4 && ((direction > 293 && direction <= 360) || (direction >= 0 && direction <= 90))) return 'S2';
  
  return null;
};

export function LeftPanel({ selectedStation, selectedDate, selectedHour, isLive = true, onNavigateToMap }: LeftPanelProps) {
  const { dataDir } = useDataDir();
  const currentTime = format(new Date(), 'HH:mm', { locale: fr });
  const formattedDate = format(selectedDate, "EEEE dd/MM/yyyy", { locale: fr });
  
  // Charger les données réelles depuis l'API
  const [currentData, setCurrentData] = useState<{
    direction: number;
    vitesse: number;
    temperature: number;
    humidite: number;
    power?: number;
    scenario: string | null;
  } | null>(null);

  useEffect(() => {
    // Réinitialiser les données quand le dossier change
    setCurrentData(null);
    
    const loadData = async () => {
      try {
        // Construire l'URL avec le paramètre data_dir si fourni
        // Ne pas envoyer "data" car c'est le dossier par défaut
        // Ajouter un timestamp pour éviter le cache du navigateur
        let url = `http://127.0.0.1:5000/api/dashboard/data`;
        const params: string[] = [];
        
        if (dataDir && dataDir.trim() !== '' && dataDir.trim().toLowerCase() !== 'data') {
          params.push(`data_dir=${encodeURIComponent(dataDir)}`);
        }
        
        // Si en mode historique, ajouter les paramètres de date
        if (!isLive) {
          const dateStr = selectedDate.toISOString().split('T')[0]; // YYYY-MM-DD
          params.push(`target_date=${dateStr}`);
          if (selectedHour) {
            params.push(`target_hour=${selectedHour}`);
          }
        } else {
          // Mode temps réel : ajouter un timestamp pour éviter le cache
          params.push(`_t=${Date.now()}`);
        }
        
        if (params.length > 0) {
          url += `?${params.join('&')}`;
        } else {
          url += `?_t=${Date.now()}`;
        }
        
        const response = await fetch(url, {
          cache: 'no-cache',
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.status === 'ok' && result.data && result.data.length > 0) {
          // Prendre la dernière mesure (la plus récente)
          const latest = result.data[result.data.length - 1];
          const scenario = calculateScenario(latest.vitesse, latest.direction);
          
          setCurrentData({
            direction: latest.direction,
            vitesse: latest.vitesse,
            temperature: latest.temperature,
            humidite: latest.humidite,
            power: latest.power,
            scenario: scenario,
          });
        } else {
          console.warn('Aucune donnée disponible pour le dossier:', dataDir || 'défaut');
        }
      } catch (error) {
        console.error('Erreur lors du chargement des données pour la rose des vents:', error);
        setCurrentData(null);
      }
    };

    loadData();
    
    // Actualiser toutes les 30 secondes
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [dataDir]);

  // Données par défaut si pas encore chargées
  const displayData = currentData || {
    direction: 0,
    vitesse: 0,
    temperature: 0,
    humidite: 0,
    scenario: null,
  };

  return (
    <div className="w-full lg:w-[30%] flex flex-col gap-2 overflow-hidden">
      {/* Wind Rose Chart - Full height */}
      <div className="bg-white dark:bg-[rgba(0,0,0,0.6)] rounded-lg p-4 shadow-sm border border-[rgba(47,163,111,0.2)] dark:border-[rgba(14,107,87,0.15)] flex-1 flex flex-col">
        <h3 className="text-sm font-semibold dark:text-white text-[#1A2A23] mb-2 text-center" style={{ fontFamily: 'var(--font-heading)' }}>
          Rose des Vents
          {displayData.scenario && (
            <span className="ml-2 text-xs font-normal opacity-75">
              (Scénario: {displayData.scenario})
            </span>
          )}
        </h3>
        <div className="flex-1">
          <WindRoseChart direction={displayData.direction} />
        </div>
      </div>
    </div>
  );
}