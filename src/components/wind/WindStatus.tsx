interface WindStatusProps {
  temp: number;
  windSpeed: number;
  windDirection: number;
  time: string;
  visible: boolean;
}

export function WindStatus({
  temp,
  windSpeed,
  windDirection,
  time,
  visible,
}: WindStatusProps) {
  if (!visible) return null;

  return (
    <div className="absolute bottom-6 left-6 bg-card/95 backdrop-blur-sm rounded-xl shadow-lg px-4 py-3 z-[1000]">
      <div className="flex items-center gap-4 font-mono text-foreground">
        <span>Temp: {temp.toFixed(1)}°C</span>
        <span className="text-muted-foreground">·</span>
        <span>Vent: {windSpeed.toFixed(1)} m/s</span>
        <span className="text-muted-foreground">·</span>
        <span>Dir: {windDirection.toFixed(0)}°</span>
        <span className="text-muted-foreground">·</span>
        <span className="text-muted-foreground">{time}</span>
      </div>
    </div>
  );
}
