import { Clock } from 'lucide-react';
import { useEffect, useState } from 'react';

export function LiveClock() {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      setTime(new Date());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-1.5 px-2 py-1 bg-emerald-500/10 border border-emerald-400/30 rounded-md backdrop-blur-sm">
      <Clock className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
      <span className="text-xs font-mono font-semibold text-emerald-700 dark:text-emerald-300">
        {time.toLocaleTimeString('fr-FR')}
      </span>
    </div>
  );
}