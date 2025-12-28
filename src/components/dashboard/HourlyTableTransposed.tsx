import { ArrowUp, Info } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';
import { useTheme } from '../ThemeContext';

interface HourlyDataPoint {
  time: string;
  direction: number;
  vitesse: number;
  temperature: number;
  scenario: string;
  humidite: number;
  power: number;
}

interface HourlyTableTransposedProps {
  data: HourlyDataPoint[];
}

// Scenario definitions with colors and descriptions
const scenarioInfo: Record<string, { decision: string; color: string; vitesseRange: string; directionRange: string }> = {
  'S1': {
    decision: 'Marche normale',
    color: '#B7EFB9',
    vitesseRange: '≥ 5 m/s',
    directionRange: '—'
  },
  'S2': {
    decision: 'Marche normale + Interdiction de démarrage',
    color: '#79D6A3',
    vitesseRange: '1 < V < 4 m/s',
    directionRange: '(WD > 293 and WD ≤ 360) or (WD ≥ 0 and WD ≤ 90)'
  },
  'S2b': {
    decision: 'Interdiction de démarrage + Réduction des cadences MP2 80% et MC et MP1 à 60%',
    color: '#85D5FF',
    vitesseRange: '≤ 1 m/s',
    directionRange: '(WD > 90 and WD ≤ 293)'
  },
  'S3': {
    decision: 'Interdiction de démarrage + Réduction des cadences au minimum technique (Réduction des cadences MP2 80% et MC et MP1 à 60%) + Démarrage injection de produit anti-odeur',
    color: '#FFB74D',
    vitesseRange: '< 0,5 m/s',
    directionRange: '(WD > 293 and WD ≤ 360) or (WD ≥ 0 and WD ≤ 90)'
  },
  'S3b': {
    decision: 'Interdiction de démarrage + Réduction des cadences au minimum technique (Réduction des cadences MP2 80% et MC et MP1 à 60%) + Démarrage injection de produit anti-odeur',
    color: '#FFA726',
    vitesseRange: '≤ 2 m/s',
    directionRange: '(WD > 156 and WD ≤ 203)'
  },
  'S4': {
    decision: 'Interdiction de démarrage + Réduction des cadences au minimum technique (Réduction des cadences MP2 80% et MC et MP1 à 60%)',
    color: '#FF8A65',
    vitesseRange: '< 0,5 m/s',
    directionRange: '(WD > 90 and WD ≤ 293)'
  }
};

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

export function HourlyTableTransposed({ data }: HourlyTableTransposedProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const getVitesseColor = (vitesse: number) => {
    if (isDark) {
      if (vitesse < 1.5) return { bg: 'rgba(20, 83, 45, 0.6)', text: '#B7EFB9' };
      if (vitesse < 2.0) return { bg: 'rgba(14, 107, 87, 0.7)', text: '#79D6A3' };
      if (vitesse < 2.5) return { bg: 'rgba(47, 163, 111, 0.5)', text: '#FFFFFF' };
      return { bg: 'rgba(14, 107, 87, 0.9)', text: '#FFFFFF' };
    }
    if (vitesse < 1.5) return { bg: '#D4EDDA', text: '#155724' };
    if (vitesse < 2.0) return { bg: '#A8D5BA', text: '#0A3D1A' };
    if (vitesse < 2.5) return { bg: '#79D6A3', text: '#FFFFFF' };
    return { bg: '#2FA36F', text: '#FFFFFF' };
  };

  const getScenarioColor = (scenario: string | null) => {
    if (!scenario) return { bg: 'transparent', text: 'inherit' };
    const color = scenarioInfo[scenario]?.color || 'transparent';
    
    if (isDark) {
      const darkColors: Record<string, string> = {
        '#B7EFB9': 'rgba(183, 239, 185, 0.3)',
        '#79D6A3': 'rgba(121, 214, 163, 0.3)',
        '#85D5FF': 'rgba(133, 213, 255, 0.3)',
        '#FFB74D': 'rgba(255, 183, 77, 0.3)',
        '#FFA726': 'rgba(255, 167, 38, 0.3)',
        '#FF8A65': 'rgba(255, 138, 101, 0.3)',
      };
      return { 
        bg: darkColors[color] || 'rgba(47, 163, 111, 0.2)', 
        text: '#FFFFFF' 
      };
    }
    
    const isDarkBg = ['#79D6A3', '#2FA36F', '#0E6B57'].some(c => color.includes(c));
    return { bg: color, text: isDarkBg ? '#FFFFFF' : '#000000' };
  };

  const getHumiditeColor = (humidite: number) => {
    if (isDark) {
      if (humidite < 50) return { bg: 'rgba(133, 213, 255, 0.3)', text: '#85D5FF' };
      if (humidite < 70) return { bg: 'rgba(14, 107, 87, 0.5)', text: '#79D6A3' };
      return { bg: 'rgba(47, 163, 111, 0.5)', text: '#FFFFFF' };
    }
    if (humidite < 50) return { bg: '#D1ECFD', text: '#004085' };
    if (humidite < 70) return { bg: '#A8D5BA', text: '#0A3D1A' };
    return { bg: '#79D6A3', text: '#FFFFFF' };
  };

  const getPowerColor = (power: number) => {
    if (isDark) {
      if (power < 215) return { bg: 'rgba(255, 183, 77, 0.3)', text: '#FFB74D' };
      if (power < 225) return { bg: 'rgba(14, 107, 87, 0.5)', text: '#79D6A3' };
      return { bg: 'rgba(47, 163, 111, 0.5)', text: '#FFFFFF' };
    }
    if (power < 215) return { bg: '#FFE4CC', text: '#7A4100' };
    if (power < 225) return { bg: '#A8D5BA', text: '#0A3D1A' };
    return { bg: '#79D6A3', text: '#FFFFFF' };
  };

  // Define the rows (data types)
  const rows = [
    { label: 'Direction', key: 'direction' as const },
    { label: 'Vit (m/s)', key: 'vitesse' as const },
    { label: 'Temp (°C)', key: 'temperature' as const },
    { label: 'Scénario', key: 'scenario' as const },
    { label: 'Hum. (%)', key: 'humidite' as const },
    { label: 'Power (V)', key: 'power' as const },
  ];

  return (
    <TooltipProvider>
      <div className="overflow-hidden">
        {/* Legend */}
        <div className="px-3 py-1.5 text-xs flex items-center gap-2" style={{ backgroundColor: isDark ? 'rgba(47, 163, 111, 0.2)' : 'rgba(47, 163, 111, 0.1)', color: isDark ? '#79D6A3' : '#2FA36F' }}>
          <Info className="h-3.5 w-3.5" />
          <span>Les flèches indiquent la direction vers laquelle le vent se déplace</span>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead style={{ backgroundColor: isDark ? 'rgba(14, 107, 87, 0.8)' : '#2FA36F', color: 'white' }}>
              <tr>
                <th className="px-3 py-2 text-left sticky left-0 z-10" style={{ backgroundColor: isDark ? 'rgba(14, 107, 87, 0.8)' : '#2FA36F' }}>
                  Paramètre
                </th>
                {data.map((point, index) => (
                  <th key={index} className="px-3 py-2 text-center whitespace-nowrap">
                    {point.time}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={row.key} className={`border-b ${isDark ? 'border-gray-800' : 'border-gray-100'} hover:bg-gray-50 dark:hover:bg-gray-800/50`}>
                  <td 
                    className="px-3 py-2 font-semibold sticky left-0 z-10" 
                    style={{ 
                      backgroundColor: isDark ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.95)',
                      color: isDark ? '#79D6A3' : '#2FA36F'
                    }}
                  >
                    {row.label}
                  </td>
                  {data.map((point, colIndex) => {
                    let cellContent: React.ReactNode;
                    let cellStyle: React.CSSProperties = { textAlign: 'center' };
                    
                    if (row.key === 'direction') {
                      cellContent = (
                        <div className="flex items-center justify-center gap-1">
                          <ArrowUp
                            className="h-3.5 w-3.5"
                            style={{
                              color: isDark ? '#79D6A3' : '#0E6B57',
                              transform: `rotate(${(point.direction + 180) % 360}deg)`
                            }}
                          />
                          <span>{Math.round(point.direction)}°</span>
                        </div>
                      );
                    } else if (row.key === 'vitesse') {
                      const colors = getVitesseColor(point.vitesse);
                      cellStyle = { ...cellStyle, backgroundColor: colors.bg, color: colors.text, fontWeight: 600 };
                      cellContent = point.vitesse.toFixed(1);
                    } else if (row.key === 'temperature') {
                      cellContent = point.temperature.toFixed(1);
                    } else if (row.key === 'scenario') {
                      const calculatedScenario = calculateScenario(point.vitesse, point.direction);
                      if (calculatedScenario) {
                        const colors = getScenarioColor(calculatedScenario);
                        const scenarioData = scenarioInfo[calculatedScenario];
                        cellStyle = { ...cellStyle, backgroundColor: colors.bg, color: colors.text, fontWeight: 600 };
                        cellContent = (
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="cursor-help">{calculatedScenario}</span>
                            </TooltipTrigger>
                            <TooltipContent 
                              className="!bg-white dark:!bg-gray-800 !text-black dark:!text-white shadow-lg rounded-md border-2"
                              style={{ 
                                borderColor: '#2FA36F',
                                padding: '8px 12px',
                                backgroundColor: '#ffffff',
                                color: '#000000'
                              }}
                            >
                              <div className="text-xs" style={{ color: '#000000' }}>
                                <p className="mb-1" style={{ color: '#000000' }}>
                                  <span className="font-semibold" style={{ color: '#000000' }}>Scénario :</span> <span style={{ color: '#000000' }}>{calculatedScenario}</span>
                                </p>
                                <p className="mb-1" style={{ color: '#000000' }}>
                                  <span className="font-semibold" style={{ color: '#000000' }}>Décision :</span> <span style={{ color: '#000000' }}>{scenarioData?.decision}</span>
                                </p>
                                <p className="mb-1" style={{ color: '#000000' }}>
                                  <span className="font-semibold" style={{ color: '#000000' }}>Vitesse :</span> <span style={{ color: '#000000' }}>{point.vitesse.toFixed(1)} m/s ({scenarioData?.vitesseRange})</span>
                                </p>
                                <p style={{ color: '#000000' }}>
                                  <span className="font-semibold" style={{ color: '#000000' }}>Direction :</span> <span style={{ color: '#000000' }}>{Math.round(point.direction)}° ({scenarioData?.directionRange})</span>
                                </p>
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        );
                      } else {
                        cellContent = '—';
                      }
                    } else if (row.key === 'humidite') {
                      const colors = getHumiditeColor(point.humidite);
                      cellStyle = { ...cellStyle, backgroundColor: colors.bg, color: colors.text, fontWeight: 600 };
                      cellContent = Math.round(point.humidite);
                    } else if (row.key === 'power') {
                      const colors = getPowerColor(point.power);
                      cellStyle = { ...cellStyle, backgroundColor: colors.bg, color: colors.text, fontWeight: 600 };
                      cellContent = point.power.toFixed(1);
                    }
                    
                    return (
                      <td key={colIndex} className="px-3 py-2 whitespace-nowrap" style={cellStyle}>
                        {cellContent}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </TooltipProvider>
  );
}
