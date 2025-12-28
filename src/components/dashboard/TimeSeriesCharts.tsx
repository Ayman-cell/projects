import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface HourlyDataPoint {
  time: string;
  direction: number;
  vitesse: number;
  temperature: number;
  scenario: string;
  humidite: number;
  power: number;
}

interface TimeSeriesChartsProps {
  data: HourlyDataPoint[];
}

export function TimeSeriesCharts({ data }: TimeSeriesChartsProps) {
  const chartConfig = {
    backgroundColor: 'rgba(47, 163, 111, 0.05)',
    lineColor: '#2FA36F',
    gridColor: '#CCCCCC'
  };

  // Calculer les domaines Y dynamiquement basés sur les valeurs mesurées
  const calculateYDomain = (key: keyof HourlyDataPoint, paddingPercent: number = 10) => {
    if (data.length === 0) {
      // Valeurs par défaut si pas de données
      const defaults: Record<string, [number, number]> = {
        direction: [0, 360],
        vitesse: [0, 5],
        temperature: [0, 30],
        humidite: [0, 100],
        power: [0, 250]
      };
      return defaults[key] || [0, 100];
    }

    const values = data.map(d => d[key] as number).filter(v => !isNaN(v) && v !== null && v !== undefined);
    if (values.length === 0) {
      const defaults: Record<string, [number, number]> = {
        direction: [0, 360],
        vitesse: [0, 5],
        temperature: [0, 30],
        humidite: [0, 100],
        power: [0, 250]
      };
      return defaults[key] || [0, 100];
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    const padding = range * (paddingPercent / 100);

    // Pour direction, garder 0-360
    if (key === 'direction') {
      return [0, 360];
    }

    // Pour les autres, ajouter un padding mais avec des limites raisonnables
    const minDomain = Math.max(0, min - padding);
    const maxDomain = max + padding;

    // Arrondir pour des valeurs propres
    const roundTo = (value: number, decimals: number) => {
      const factor = Math.pow(10, decimals);
      return Math.round(value * factor) / factor;
    };

    return [roundTo(minDomain, 1), roundTo(maxDomain, 1)];
  };

  const directionDomain = calculateYDomain('direction', 0);
  const vitesseDomain = calculateYDomain('vitesse', 15);
  const temperatureDomain = calculateYDomain('temperature', 15);
  const humiditeDomain = calculateYDomain('humidite', 10);
  const powerDomain = calculateYDomain('power', 10);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-gray-800 p-3 shadow-lg rounded-lg border-2" style={{ borderColor: '#2FA36F' }}>
          <p className="text-sm dark:text-white">{label}</p>
          <p className="text-sm" style={{ color: '#2FA36F' }}>
            {payload[0].name}: {payload[0].value.toFixed(1)} {payload[0].unit}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-full flex flex-col gap-2">
      <h3 className="text-center text-sm font-semibold mb-1" style={{ color: '#2FA36F', fontFamily: 'var(--font-heading)' }}>Évolution temporelle des paramètres</h3>

      {/* Direction Chart */}
      <div className="flex-1 min-h-0">
        <h4 className="text-xs font-medium mb-1 text-gray-600 dark:text-gray-400">Direction (°)</h4>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} style={{ backgroundColor: chartConfig.backgroundColor }} margin={{ left: 5, right: 10, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartConfig.gridColor} className="dark:stroke-gray-600" strokeOpacity={0.5} />
            <XAxis
              dataKey="time"
              tick={{ fill: '#808080', fontSize: 9 }}
              className="dark:fill-gray-400"
              interval={0}
              height={20}
            />
            <YAxis
              tick={{ fill: '#808080', fontSize: 9 }}
              className="dark:fill-gray-400"
              domain={directionDomain}
              width={35}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="direction"
              stroke={chartConfig.lineColor}
              strokeWidth={3}
              dot={{ r: 3, fill: chartConfig.lineColor, strokeWidth: 2, stroke: '#fff' }}
              activeDot={{ r: 5 }}
              name="Direction"
              unit="°"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Vitesse Chart */}
      <div className="flex-1 min-h-0">
        <h4 className="text-xs font-medium mb-1 text-gray-600 dark:text-gray-400">Vitesse (m/s)</h4>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} style={{ backgroundColor: chartConfig.backgroundColor }} margin={{ left: 5, right: 10, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartConfig.gridColor} className="dark:stroke-gray-600" strokeOpacity={0.5} />
            <XAxis
              dataKey="time"
              tick={{ fill: '#808080', fontSize: 9 }}
              className="dark:fill-gray-400"
              interval={0}
              height={20}
            />
            <YAxis
              tick={{ fill: '#808080', fontSize: 9 }}
              className="dark:fill-gray-400"
              domain={vitesseDomain}
              width={35}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="vitesse"
              stroke={chartConfig.lineColor}
              strokeWidth={3}
              dot={{ r: 3, fill: chartConfig.lineColor, strokeWidth: 2, stroke: '#fff' }}
              activeDot={{ r: 5 }}
              name="Vitesse"
              unit="m/s"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Temperature Chart */}
      <div className="flex-1 min-h-0">
        <h4 className="text-xs font-medium mb-1 text-gray-600 dark:text-gray-400">Température (°C)</h4>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} style={{ backgroundColor: chartConfig.backgroundColor }} margin={{ left: 5, right: 10, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartConfig.gridColor} className="dark:stroke-gray-600" strokeOpacity={0.5} />
            <XAxis
              dataKey="time"
              tick={{ fill: '#808080', fontSize: 9 }}
              className="dark:fill-gray-400"
              interval={0}
              height={20}
            />
            <YAxis
              tick={{ fill: '#808080', fontSize: 9 }}
              className="dark:fill-gray-400"
              domain={temperatureDomain}
              width={35}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="temperature"
              stroke={chartConfig.lineColor}
              strokeWidth={3}
              dot={{ r: 3, fill: chartConfig.lineColor, strokeWidth: 2, stroke: '#fff' }}
              activeDot={{ r: 5 }}
              name="Température"
              unit="°C"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Humidite Chart */}
      <div className="flex-1 min-h-0">
        <h4 className="text-xs font-medium mb-1 text-gray-600 dark:text-gray-400">Humidité (%)</h4>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} style={{ backgroundColor: chartConfig.backgroundColor }} margin={{ left: 5, right: 10, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartConfig.gridColor} className="dark:stroke-gray-600" strokeOpacity={0.5} />
            <XAxis
              dataKey="time"
              tick={{ fill: '#808080', fontSize: 9 }}
              className="dark:fill-gray-400"
              interval={0}
              height={20}
            />
            <YAxis
              tick={{ fill: '#808080', fontSize: 9 }}
              className="dark:fill-gray-400"
              domain={humiditeDomain}
              width={35}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="humidite"
              stroke={chartConfig.lineColor}
              strokeWidth={3}
              dot={{ r: 3, fill: chartConfig.lineColor, strokeWidth: 2, stroke: '#fff' }}
              activeDot={{ r: 5 }}
              name="Humidité"
              unit="%"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Power Chart */}
      <div className="flex-1 min-h-0">
        <h4 className="text-xs font-medium mb-1 text-gray-600 dark:text-gray-400">Power (V)</h4>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} style={{ backgroundColor: chartConfig.backgroundColor }} margin={{ left: 5, right: 10, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartConfig.gridColor} className="dark:stroke-gray-600" strokeOpacity={0.5} />
            <XAxis
              dataKey="time"
              tick={{ fill: '#808080', fontSize: 9 }}
              className="dark:fill-gray-400"
              interval={0}
              height={20}
            />
            <YAxis
              tick={{ fill: '#808080', fontSize: 9 }}
              className="dark:fill-gray-400"
              domain={powerDomain}
              width={35}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="power"
              stroke={chartConfig.lineColor}
              strokeWidth={3}
              dot={{ r: 3, fill: chartConfig.lineColor, strokeWidth: 2, stroke: '#fff' }}
              activeDot={{ r: 5 }}
              name="Power"
              unit="V"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}