import { X, MapPin } from 'lucide-react';
import { Button } from '../ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { useState } from 'react';

interface MapSectionProps {
  onClose: () => void;
  onStationSelect: (station: string) => void;
  selectedStation: string;
}

export function MapSection({ onClose, onStationSelect, selectedStation }: MapSectionProps) {
  const [selectedPollutant, setSelectedPollutant] = useState<string>('all');

  const stations = [
    { id: 'GP1', name: 'Station GP1', lat: 48.856, lng: 2.352, status: 'active' },
    { id: 'GP2', name: 'Station GP2', lat: 48.860, lng: 2.340, status: 'active' },
    { id: 'GP3', name: 'Station GP3', lat: 48.850, lng: 2.360, status: 'active' },
    { id: 'GP4', name: 'Station GP4', lat: 48.845, lng: 2.350, status: 'warning' }
  ];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-6">
      <div className="bg-white dark:bg-[rgba(0,0,0,0.9)] rounded-lg w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col border border-[rgba(47,163,111,0.2)] dark:border-[rgba(14,107,87,0.3)]">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[rgba(47,163,111,0.2)] dark:border-[rgba(14,107,87,0.3)]" style={{ backgroundColor: '#2FA36F' }}>
          <h2 className="text-white font-semibold">Carte des stations en temps réel</h2>
          <div className="flex items-center gap-3">
            <Select value={selectedPollutant} onValueChange={setSelectedPollutant}>
              <SelectTrigger className="w-40 bg-white">
                <SelectValue placeholder="Filtrer par polluant" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Tous</SelectItem>
                <SelectItem value="pm25">PM2.5</SelectItem>
                <SelectItem value="no2">NO₂</SelectItem>
                <SelectItem value="o3">O₃</SelectItem>
                <SelectItem value="co">CO</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="ghost" onClick={onClose} className="text-white hover:bg-white hover:bg-opacity-20">
              <X className="h-5 w-5" />
            </Button>
          </div>
        </div>

        {/* Map Content */}
        <div className="flex-1 relative" style={{ backgroundColor: 'rgba(121, 214, 163, 0.1)' }}>
          {/* Simulated Map Background */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="relative w-full h-full">
              {/* Grid lines to simulate map */}
              <svg className="absolute inset-0 w-full h-full opacity-20">
                <defs>
                  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#2FA36F" strokeWidth="0.5"/>
                  </pattern>
                </defs>
                <rect width="100%" height="100%" fill="url(#grid)" />
              </svg>

              {/* Station Markers */}
              {stations.map((station, index) => (
                <button
                  key={station.id}
                  onClick={() => {
                    onStationSelect(station.id);
                    onClose();
                  }}
                  className="absolute transform -translate-x-1/2 -translate-y-1/2 transition-transform hover:scale-110"
                  style={{
                    left: `${30 + index * 15}%`,
                    top: `${35 + (index % 2) * 20}%`
                  }}
                >
                  <div className="flex flex-col items-center">
                    <div
                      className="p-2 rounded-full shadow-lg"
                      style={{
                        backgroundColor: station.status === 'active' ? '#2FA36F' : '#FFA726',
                        border: selectedStation === station.id ? '3px solid #0E6B57' : 'none'
                      }}
                    >
                      <MapPin className="h-6 w-6 text-white" />
                    </div>
                    <div className="mt-1 bg-white dark:bg-gray-800 px-2 py-1 rounded shadow-sm text-xs whitespace-nowrap">
                      {station.name}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="absolute bottom-4 left-4 bg-white dark:bg-gray-800 p-4 rounded-lg shadow-lg border border-[rgba(47,163,111,0.2)] dark:border-[rgba(14,107,87,0.3)]">
            <h4 className="text-sm mb-2" style={{ color: '#2FA36F' }}>Légende</h4>
            <div className="space-y-2 text-xs dark:text-gray-300">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#2FA36F' }}></div>
                <span>Station active</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#FFA726' }}></div>
                <span>Alerte</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full border-2" style={{ borderColor: '#0E6B57' }}></div>
                <span>Sélectionnée</span>
              </div>
            </div>
          </div>

          {/* Info Box */}
          <div className="absolute top-4 right-4 bg-white dark:bg-gray-800 p-4 rounded-lg shadow-lg max-w-xs border border-[rgba(47,163,111,0.2)] dark:border-[rgba(14,107,87,0.3)]">
            <h4 className="text-sm mb-2" style={{ color: '#2FA36F' }}>Instructions</h4>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              Cliquez sur une station pour afficher ses données en temps réel dans le panneau principal.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
