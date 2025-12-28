import React, { useMemo, useState, useEffect } from "react";
import { HourlyTableTransposed } from "./HourlyTableTransposed";
import { TimeSeriesCharts } from "./TimeSeriesCharts";
import { MetricCard } from "./MetricCard";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "../ui/tabs";
import { useDataDir } from "../../contexts/DataDirContext";

interface RightPanelProps {
  selectedStation: string;
  selectedDate: Date;
  selectedPeriod: "day" | "month" | "year";
  selectedHour?: string;
  isLive?: boolean;
}

export function RightPanel({
  selectedStation,
  selectedDate,
  selectedPeriod,
  selectedHour,
  isLive,
}: RightPanelProps) {
  const { dataDir } = useDataDir();
  const [activeTab, setActiveTab] = useState("tableau");

  // Charger les données réelles depuis l'API (10 dernières mesures)
  const [hourlyData, setHourlyData] = useState<any[]>([]);
  const [isLoadingData, setIsLoadingData] = useState(true);

  useEffect(() => {
    // Réinitialiser les données quand le dossier change
      setHourlyData([]);
    setIsLoadingData(true);

    const loadData = async () => {
      try {
        // Construire l'URL avec le paramètre data_dir si fourni
        // Ne pas envoyer "data" car c'est le dossier par défaut
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
        
        if (result.status === 'error') {
          console.error('Erreur du serveur:', result.error);
          console.error('Détails:', result);
          setIsLoadingData(false);
          setHourlyData([]);
          return;
        }
        
        if (result.status === 'ok' && result.data) {
          // Fonction pour calculer le scénario selon le tableau officiel
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
          
          // Transformer les données pour le format attendu
          let transformedData = result.data.map((item: any) => ({
            time: item.time,
            date: item.date,
            timestamp: item.timestamp,
            direction: item.direction,
            vitesse: item.vitesse,
            temperature: item.temperature,
            humidite: item.humidite,
            power: item.power,
            scenario: calculateScenario(item.vitesse, item.direction) || 'S1a', // Calculer le scénario réel
          }));
          
          // En mode historique, s'assurer que les données sont triées en ordre croissant
          if (!isLive) {
            transformedData.sort((a: any, b: any) => {
              const dateA = new Date(a.timestamp).getTime();
              const dateB = new Date(b.timestamp).getTime();
              return dateA - dateB; // Ordre croissant
            });
          }
          
          // En mode historique, remplacer complètement les données
          // En mode temps réel, faire la mise à jour incrémentale
          if (!isLive) {
            // Mode historique : remplacer complètement les données
            setHourlyData(transformedData);
          } else {
            // Mode temps réel : mise à jour incrémentale
          setHourlyData((prevData) => {
              // Si c'est le premier chargement ou si les données sont vides, utiliser les nouvelles données
            if (prevData.length === 0) {
              return transformedData;
            }
              
              // Vérifier si les données proviennent du même dossier en comparant les timestamps
              // Si les nouvelles données sont complètement différentes, remplacer tout
              const firstNewTimestamp = transformedData[0]?.timestamp;
              const lastPrevTimestamp = prevData[prevData.length - 1]?.timestamp;
              
              // Si les timestamps ne se chevauchent pas du tout, remplacer complètement
              if (!lastPrevTimestamp || !firstNewTimestamp || 
                  (firstNewTimestamp > lastPrevTimestamp && 
                   transformedData.length === prevData.length)) {
                // Probablement un changement de dossier, remplacer tout
                return transformedData;
              }
            
            // Sinon, comparer avec les données existantes et ajouter seulement les nouvelles
            const newData = transformedData.filter((item: any) => {
              // Garder seulement les données plus récentes que la dernière connue
                return !lastPrevTimestamp || item.timestamp > lastPrevTimestamp;
            });
            
            // Combiner les anciennes et nouvelles données, garder seulement les 10 dernières
            const combined = [...prevData, ...newData];
            return combined.slice(-10); // Garder seulement les 10 dernières
          });
          }
        } else {
          console.error('Erreur dans la réponse API:', result);
          if (result.error) {
            console.error('Erreur du serveur:', result.error);
            if (result.files_in_directory) {
              console.warn('Fichiers présents dans le dossier:', result.files_in_directory);
            }
            if (result.data_dir) {
              console.warn('Dossier utilisé:', result.data_dir);
            }
          }
        }
      } catch (error) {
        console.error('Erreur lors du chargement des données:', error);
        setIsLoadingData(false);
        setHourlyData([]);
      } finally {
        setIsLoadingData(false);
      }
    };

    // Charger immédiatement
    loadData();
    
    // En mode temps réel, actualiser toutes les 30 secondes
    // En mode historique, ne pas actualiser automatiquement
    let interval: NodeJS.Timeout | null = null;
    if (isLive) {
      interval = setInterval(() => {
      loadData();
    }, 30000); // 30 secondes
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isLive, dataDir, selectedDate, selectedHour]);

  // Use all entries for charts
  const chartData = useMemo(() => hourlyData, [hourlyData]);

  // Get current metrics from the latest data point
  const currentMetrics = useMemo(() => {
    if (hourlyData.length > 0) {
      const latest = hourlyData[hourlyData.length - 1];
      return {
        direction: latest.direction || 0,
        vitesse: latest.vitesse || 0,
        temperature: latest.temperature || 0,
        humidite: latest.humidite || 0,
      };
    }
    
    // Fallback si pas de données
    return {
      direction: 0,
      vitesse: 0,
      temperature: 0,
      humidite: 0,
    };
  }, [hourlyData]);

  return (
    <div className="w-full lg:w-[70%] flex flex-col gap-3 overflow-hidden">
      <Tabs
        value={activeTab}
        onValueChange={setActiveTab}
        className="flex flex-col flex-1 overflow-hidden"
      >
        <TabsList className="w-full grid grid-cols-2 h-12 bg-white/80 dark:bg-[rgba(0,0,0,0.6)] border border-[rgba(47,163,111,0.2)] dark:border-[rgba(14,107,87,0.15)]">
          <TabsTrigger
            value="tableau"
            className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-[#0E6B57] data-[state=active]:via-[#2FA36F] data-[state=active]:to-[#0E6B57] data-[state=active]:text-white font-semibold"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            Tableau
          </TabsTrigger>
          <TabsTrigger
            value="courbes"
            className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-[#0E6B57] data-[state=active]:via-[#2FA36F] data-[state=active]:to-[#0E6B57] data-[state=active]:text-white font-semibold"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            Courbes
          </TabsTrigger>
        </TabsList>

        <TabsContent
          value="tableau"
          className="flex-1 overflow-hidden mt-3 flex flex-col gap-3"
        >
          <div className="bg-white dark:bg-[rgba(0,0,0,0.6)] rounded-lg shadow-sm overflow-auto flex-1 border border-[rgba(47,163,111,0.2)] dark:border-[rgba(14,107,87,0.15)]">
            {isLoadingData ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="text-muted-foreground">Chargement des données...</div>
                </div>
              </div>
            ) : hourlyData.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="text-muted-foreground">Aucune donnée disponible</div>
                </div>
              </div>
            ) : (
              <HourlyTableTransposed data={hourlyData} />
            )}
          </div>
        </TabsContent>

        <TabsContent
          value="courbes"
          className="flex-1 overflow-hidden mt-3"
        >
          <div className="bg-white dark:bg-[rgba(0,0,0,0.6)] rounded-lg p-4 shadow-sm h-full overflow-auto border border-[rgba(47,163,111,0.2)] dark:border-[rgba(14,107,87,0.15)]">
            <TimeSeriesCharts data={chartData} />
          </div>
        </TabsContent>
      </Tabs>

      {/* Metrics Cards - At the bottom */}
      <div className="grid grid-cols-4 gap-2 flex-shrink-0">
        <MetricCard
          label="Direction"
          value={`${Math.round(currentMetrics.direction)}°`}
          icon="compass"
        />
        <MetricCard
          label="Vitesse"
          value={`${currentMetrics.vitesse.toFixed(1)} m/s`}
          icon="wind"
        />
        <MetricCard
          label="Température"
          value={`${currentMetrics.temperature.toFixed(1)} °C`}
          icon="thermometer"
        />
        <MetricCard
          label="Humidité"
          value={`${Math.round(currentMetrics.humidite)} %`}
          icon="droplet"
        />
      </div>
    </div>
  );
}