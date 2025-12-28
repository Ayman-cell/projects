interface WindLegendProps {
  tempMin: number;
  tempMax: number;
}

export function WindLegend({ tempMin, tempMax }: WindLegendProps) {
  const tempMid = (tempMin + tempMax) / 2;

  return (
    <div className="fixed bottom-32 left-6 bg-card/95 backdrop-blur-sm rounded-xl shadow-lg p-4 z-[1000] min-w-[280px] border border-border">
      <div className="space-y-3">
        <p className="text-sm font-semibold text-foreground">Température Air (°C)</p>
        <div className="relative h-8 rounded-lg overflow-hidden">
          <div
            className="absolute inset-0"
            style={{
              background:
                'linear-gradient(to right, #2c7bb6 0%, #abd9e9 25%, #ffffbf 50%, #fdae61 75%, #d7191c 100%)',
            }}
          />
        </div>
        <div className="flex justify-between items-center">
          <span className="font-mono text-sm">{tempMin.toFixed(1)}°</span>
          <span className="font-mono text-sm text-muted-foreground">
            {tempMid.toFixed(1)}°
          </span>
          <span className="font-mono text-sm">{tempMax.toFixed(1)}°</span>
        </div>
      </div>
    </div>
  );
}