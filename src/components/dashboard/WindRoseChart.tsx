import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer } from 'recharts';
import { useTheme } from '../ThemeContext';

interface WindRoseChartProps {
  direction: number;
}

export function WindRoseChart({ direction }: WindRoseChartProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  
  // Convert measured direction (where wind comes from) to display direction (where wind goes)
  // Using 360 - direction transformation
  const displayDirection = (360 - direction) % 360;
  
  // Create wind rose data with emphasis on the display direction
  const createWindRoseData = () => {
    const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
    const angleStep = 360 / directions.length;
    
    return directions.map((dir, index) => {
      const angle = index * angleStep;
      const diff = Math.abs(angle - displayDirection);
      const normalizedDiff = Math.min(diff, 360 - diff);
      const value = normalizedDiff < 45 ? 100 - normalizedDiff : 20;
      
      return {
        direction: dir,
        value: value
      };
    });
  };

  const data = createWindRoseData();

  return (
    <div className="w-full h-full flex flex-col justify-center">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data}>
          <PolarGrid 
            stroke={isDark ? '#6B7280' : '#9CA3AF'} 
            strokeWidth={2}
          />
          <PolarAngleAxis
            dataKey="direction"
            tick={{ fill: isDark ? '#E5E7EB' : '#4B5563', fontSize: 11, fontWeight: 600 }}
          />
          <Radar
            name="Direction"
            dataKey="value"
            stroke={isDark ? '#79D6A3' : '#2FA36F'}
            strokeWidth={3}
            fill={isDark ? '#79D6A3' : '#2FA36F'}
            fillOpacity={0.5}
          />
        </RadarChart>
      </ResponsiveContainer>
      <div className="text-center mt-2 text-sm font-semibold" style={{ color: isDark ? '#79D6A3' : '#2FA36F' }}>
        Direction vers: {Math.round(displayDirection)}°
      </div>
    </div>
  );
}