import { ArrowUp, Info } from 'lucide-react';
import { ScrollArea } from '../ui/scroll-area';
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

interface HourlyTableProps {
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

export function HourlyTable({ data }: HourlyTableProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const getVitesseColor = (vitesse: number) => {
    if (isDark) {
      // Dark mode: couleurs plus sombres
      if (vitesse < 1.5) return { bg: 'rgba(20, 83, 45, 0.6)', text: '#B7EFB9' };
      if (vitesse < 2.0) return { bg: 'rgba(14, 107, 87, 0.7)', text: '#79D6A3' };
      if (vitesse < 2.5) return { bg: 'rgba(47, 163, 111, 0.5)', text: '#FFFFFF' };
      return { bg: 'rgba(14, 107, 87, 0.9)', text: '#FFFFFF' };
    }
    // Light mode
    if (vitesse < 1.5) return { bg: '#D4EDDA', text: '#155724' };
    if (vitesse < 2.0) return { bg: '#A8D5BA', text: '#0A3D1A' };
    if (vitesse < 2.5) return { bg: '#79D6A3', text: '#FFFFFF' };
    return { bg: '#2FA36F', text: '#FFFFFF' };
  };

  const getScenarioColor = (scenario: string | null) => {
    if (!scenario) return { bg: 'transparent', text: 'inherit' };
    const color = scenarioInfo[scenario]?.color || 'transparent';
    
    if (isDark) {
      // Couleurs adaptées au dark mode
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

  return (
    <TooltipProvider>
      <div className="overflow-hidden">
        {/* Legend */}
        <div className="px-2 py-0.5 text-xs flex items-center gap-1" style={{ backgroundColor: isDark ? 'rgba(47, 163, 111, 0.2)' : 'rgba(47, 163, 111, 0.1)', color: isDark ? '#79D6A3' : '#2FA36F' }}>
          <Info className="h-3 w-3" />
          <span>Les flèches indiquent la direction vers laquelle le vent se déplace</span>
        </div>
        
        <table className="w-full text-xs">
          <thead style={{ backgroundColor: isDark ? 'rgba(14, 107, 87, 0.8)' : '#2FA36F', color: 'white' }}>
            <tr>
              <th className="px-2 py-1 text-left">Heure</th>
              <th className="px-2 py-1 text-center">Direction</th>
              <th className="px-2 py-1 text-center">Vit (m/s)</th>
              <th className="px-2 py-1 text-center">Temp (°C)</th>
              <th className="px-2 py-1 text-center">Scénario</th>
              <th className="px-2 py-1 text-center">Hum. (%)</th>
              <th className="px-2 py-1 text-center">Power (V)</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, index) => {
              const calculatedScenario = calculateScenario(row.vitesse, row.direction);
              const scenarioData = calculatedScenario ? scenarioInfo[calculatedScenario] : null;
              const vitesseColors = getVitesseColor(row.vitesse);
              const scenarioColors = getScenarioColor(calculatedScenario);
              const humiditeColors = getHumiditeColor(row.humidite);
              const powerColors = getPowerColor(row.power);
              
              return (
                <tr key={index} className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                  <td className="px-2 py-1">{row.time}</td>
                  <td className="px-2 py-1 text-center">
                    <div className="flex items-center justify-center">
                      <ArrowUp
                        className="h-3 w-3"
                        style={{
                          color: isDark ? '#79D6A3' : '#0E6B57',
                          transform: `rotate(${(row.direction + 180) % 360}deg)`
                        }}
                      />
                      <span className="ml-0.5 text-xs">{Math.round(row.direction)}°</span>
                    </div>
                  </td>
                  <td
                    className="px-2 py-1 text-center font-medium"
                    style={{ 
                      backgroundColor: vitesseColors.bg,
                      color: vitesseColors.text
                    }}
                  >
                    {row.vitesse.toFixed(1)}
                  </td>
                  <td className="px-2 py-1 text-center">
                    {row.temperature.toFixed(1)}
                  </td>
                  {calculatedScenario ? (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <td
                          className="px-2 py-1 text-center cursor-help transition-opacity hover:opacity-90 font-medium"
                          style={{ 
                            backgroundColor: scenarioColors.bg,
                            color: scenarioColors.text
                          }}
                        >
                          {calculatedScenario}
                        </td>
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
                            <span className="font-semibold" style={{ color: '#000000' }}>Vitesse :</span> <span style={{ color: '#000000' }}>{row.vitesse.toFixed(1)} m/s ({scenarioData?.vitesseRange})</span>
                          </p>
                          <p style={{ color: '#000000' }}>
                            <span className="font-semibold" style={{ color: '#000000' }}>Direction :</span> <span style={{ color: '#000000' }}>{Math.round(row.direction)}° ({scenarioData?.directionRange})</span>
                          </p>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  ) : (
                    <td className="px-2 py-1 text-center">—</td>
                  )}
                  <td
                    className="px-2 py-1 text-center font-medium"
                    style={{ 
                      backgroundColor: humiditeColors.bg,
                      color: humiditeColors.text
                    }}
                  >
                    {Math.round(row.humidite)}
                  </td>
                  <td
                    className="px-2 py-1 text-center font-medium"
                    style={{ 
                      backgroundColor: powerColors.bg,
                      color: powerColors.text
                    }}
                  >
                    {row.power.toFixed(1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </TooltipProvider>
  );
}