import { Compass, Wind, Thermometer, Droplet, Zap } from 'lucide-react';

interface MetricCardProps {
  label: string;
  value: string;
  icon: 'compass' | 'wind' | 'thermometer' | 'droplet' | 'zap';
}

export function MetricCard({ label, value, icon }: MetricCardProps) {
  const iconMap = {
    compass: Compass,
    wind: Wind,
    thermometer: Thermometer,
    droplet: Droplet,
    zap: Zap
  };

  const Icon = iconMap[icon];

  return (
    <div className="bg-white dark:bg-[rgba(0,0,0,0.7)] rounded-lg p-1.5 shadow-sm flex items-center gap-1.5 border border-[rgba(47,163,111,0.15)] dark:border-[rgba(14,107,87,0.2)]">
      <div className="p-1 rounded-full" style={{ backgroundColor: 'rgba(121, 214, 163, 0.15)' }}>
        <Icon className="h-3 w-3" style={{ color: '#3DAA7C' }} />
      </div>
      <div className="flex-1">
        <p className="text-xs leading-tight text-gray-600 dark:text-gray-400">{label}</p>
        <p className="text-xs leading-tight font-semibold" style={{ color: '#3DAA7C' }}>{value}</p>
      </div>
    </div>
  );
}
