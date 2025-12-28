import { Clock, Play, Loader2 } from 'lucide-react';
import { Slider } from '../ui/slider';
import { Button } from '../ui/button';
import { useState, useEffect } from 'react';

interface ForecastSliderProps {
  forecastHours: number;
  onForecastHoursChange: (hours: number) => void;
  isLive: boolean;
}

export function ForecastSlider({ forecastHours, onForecastHoursChange, isLive }: ForecastSliderProps) {
  const [tempValue, setTempValue] = useState(forecastHours);
  const [isCalculating, setIsCalculating] = useState(false);

  // Sync tempValue with forecastHours when it changes externally
  useEffect(() => {
    setTempValue(forecastHours);
  }, [forecastHours]);

  const getCurrentPlusHours = (hours: number) => {
    const now = new Date();
    const future = new Date(now.getTime() + hours * 60 * 60 * 1000);
    return future.toLocaleTimeString('fr-FR', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleCalculate = async () => {
    setIsCalculating(true);
    
    // Simulate API call/calculation (1.5 seconds)
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    onForecastHoursChange(tempValue);
    setIsCalculating(false);
  };

  const hasChanged = tempValue !== forecastHours;

  if (!isLive) return null;

  return (
    <div className="w-full flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-400/30 rounded-lg backdrop-blur-sm">
      <Clock className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
      
      <div className="flex-1 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold dark:text-white/80 text-[#1A2A23]/80" style={{ fontFamily: 'var(--font-heading)' }}>
            Prévision Météo
          </span>
          <span className="text-sm font-bold text-blue-600 dark:text-blue-400" style={{ fontFamily: 'var(--font-heading)' }}>
            +{tempValue}h {tempValue > 0 && `(${getCurrentPlusHours(tempValue)})`}
          </span>
        </div>
        
        <div className="flex items-center gap-3">
          <Slider
            value={[tempValue]}
            onValueChange={(value) => setTempValue(value[0])}
            max={3}
            step={0.5}
            className="flex-1"
            disabled={isCalculating}
          />
          
          <Button
            onClick={handleCalculate}
            disabled={!hasChanged || isCalculating}
            size="sm"
            className="h-8 px-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold transition-all duration-300 disabled:opacity-50"
            style={{ fontFamily: 'var(--font-heading)' }}
          >
            {isCalculating ? (
              <>
                <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                Calcul...
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 mr-1.5" />
                Calculer
              </>
            )}
          </Button>
        </div>
        
        <div className="flex justify-between text-xs dark:text-white/40 text-[#1A2A23]/40">
          <span>Maintenant</span>
          <span>+3h</span>
        </div>
      </div>
    </div>
  );
}