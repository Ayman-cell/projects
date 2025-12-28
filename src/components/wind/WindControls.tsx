import { Play, Target } from 'lucide-react';
import { Switch } from '../ui/switch';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Slider } from '../ui/slider';
import { Label } from '../ui/label';
import { Progress } from '../ui/progress';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '../ui/accordion';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select';

interface WindControlsProps {
  isLive: boolean;
  autoUpdate: boolean;
  onAutoUpdateChange: (enabled: boolean) => void;
  onResumeLive: () => void;
  timeOffset: number;
  onTimeOffsetChange: (offset: number) => void;
  showParticles: boolean;
  onShowParticlesChange: (show: boolean) => void;
  particleOpacity: number;
  onParticleOpacityChange: (opacity: number) => void;
  showTempBackground: boolean;
  onShowTempBackgroundChange: (show: boolean) => void;
  tempBackgroundOpacity: number;
  onTempBackgroundOpacityChange: (opacity: number) => void;
  particleDensity: string;
  onParticleDensityChange: (density: string) => void;
  showLabels: boolean;
  onShowLabelsChange: (show: boolean) => void;
  showCursorInfo: boolean;
  onShowCursorInfoChange: (show: boolean) => void;
  onCenterMap: () => void;
  isLoading?: boolean;
  loadingProgress?: number;
}

export function WindControls({
  isLive,
  autoUpdate,
  onAutoUpdateChange,
  onResumeLive,
  timeOffset,
  onTimeOffsetChange,
  showParticles,
  onShowParticlesChange,
  particleOpacity,
  onParticleOpacityChange,
  showTempBackground,
  onShowTempBackgroundChange,
  tempBackgroundOpacity,
  onTempBackgroundOpacityChange,
  particleDensity,
  onParticleDensityChange,
  showLabels,
  onShowLabelsChange,
  showCursorInfo,
  onShowCursorInfoChange,
  onCenterMap,
  isLoading = false,
  loadingProgress = 0,
}: WindControlsProps) {
  const formatTimeOffset = (minutes: number) => {
    if (minutes === 0) return 'Maintenant';
    if (minutes > 0) {
      const hours = Math.floor(minutes / 60);
      const mins = minutes % 60;
      return `D'ici ${hours}h${mins > 0 ? mins.toString().padStart(2, '0') : ''}`;
    } else {
      const hours = Math.floor(Math.abs(minutes) / 60);
      const mins = Math.abs(minutes) % 60;
      return `Il y a ${hours}h${mins > 0 ? mins.toString().padStart(2, '0') : ''}`;
    }
  };

  return (
    <aside className="w-80 border-l bg-card overflow-y-auto flex flex-col">
      <div className="p-4 space-y-6">
        {/* Source & Live */}
        <div className="space-y-4">
          <div>
            <Badge variant="outline" className="bg-muted">
              Open-Meteo (Live)
            </Badge>
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="auto-update">Mise à jour auto (5 s)</Label>
            <Switch
              id="auto-update"
              checked={autoUpdate}
              onCheckedChange={onAutoUpdateChange}
            />
          </div>

          {isLoading && (
            <div className="space-y-2">
              <p className="text-muted-foreground">
                Premier chargement 21×21
              </p>
              <Progress value={loadingProgress} className="h-1" />
            </div>
          )}

          {!isLive && (
            <Button
              onClick={onResumeLive}
              variant="outline"
              className="w-full rounded-xl"
            >
              <Play className="h-4 w-4 mr-2" />
              Reprendre Live
            </Button>
          )}
        </div>

        {/* Temps */}
        <div className="space-y-4 pt-4 border-t">
          <Label>Temps</Label>
          <div className="space-y-2">
            <Slider
              value={[timeOffset]}
              onValueChange={(value) => onTimeOffsetChange(value[0])}
              min={-180}
              max={180}
              step={15}
              className="w-full"
            />
            <div className="flex justify-between text-muted-foreground text-xs">
              <span>-3h</span>
              <span>-1h</span>
              <span className="font-semibold">0</span>
              <span>+1h</span>
              <span>+3h</span>
            </div>
          </div>
          <div className="text-center">
            <Badge variant="secondary" className="px-4 py-1.5">
              {formatTimeOffset(timeOffset)}
            </Badge>
          </div>
        </div>

        {/* Couches */}
        <Accordion type="single" collapsible className="border-t pt-4">
          <AccordionItem value="layers" className="border-none">
            <AccordionTrigger>Couches</AccordionTrigger>
            <AccordionContent className="space-y-4">
              {/* Temperature Background */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label htmlFor="temp-bg">Température Air (°C)</Label>
                  <Switch
                    id="temp-bg"
                    checked={showTempBackground}
                    onCheckedChange={onShowTempBackgroundChange}
                  />
                </div>
                {showTempBackground && (
                  <div className="space-y-2 pl-4">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Opacité</span>
                      <span className="text-muted-foreground">
                        {tempBackgroundOpacity}%
                      </span>
                    </div>
                    <Slider
                      value={[tempBackgroundOpacity]}
                      onValueChange={(value) =>
                        onTempBackgroundOpacityChange(value[0])
                      }
                      max={100}
                      step={5}
                    />
                  </div>
                )}
              </div>

              {/* Particles */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label htmlFor="particles">Vent (particules)</Label>
                  <Switch
                    id="particles"
                    checked={showParticles}
                    onCheckedChange={onShowParticlesChange}
                  />
                </div>
                {showParticles && (
                  <div className="space-y-3 pl-4">
                    <div className="space-y-2">
                      <Label htmlFor="density">Densité</Label>
                      <Select value={particleDensity} onValueChange={onParticleDensityChange}>
                        <SelectTrigger id="density">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="low">Faible</SelectItem>
                          <SelectItem value="medium">Moyenne</SelectItem>
                          <SelectItem value="high">Forte</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Opacité</span>
                        <span className="text-muted-foreground">
                          {particleOpacity}%
                        </span>
                      </div>
                      <Slider
                        value={[particleOpacity]}
                        onValueChange={(value) =>
                          onParticleOpacityChange(value[0])
                        }
                        max={100}
                        step={5}
                      />
                    </div>
                  </div>
                )}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {/* Affichage */}
        <Accordion type="single" collapsible>
          <AccordionItem value="display" className="border-none">
            <AccordionTrigger>Affichage</AccordionTrigger>
            <AccordionContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="labels">Étiquettes numériques (°C)</Label>
                <Switch
                  id="labels"
                  checked={showLabels}
                  onCheckedChange={onShowLabelsChange}
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="cursor-info">Info sous curseur</Label>
                <Switch
                  id="cursor-info"
                  checked={showCursorInfo}
                  onCheckedChange={onShowCursorInfoChange}
                />
              </div>

              <Button
                onClick={onCenterMap}
                variant="outline"
                className="w-full rounded-xl"
              >
                <Target className="h-4 w-4 mr-2" />
                Centrer sur Safi
              </Button>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {/* Aide */}
        <Accordion type="single" collapsible>
          <AccordionItem value="help" className="border-none">
            <AccordionTrigger>Aide</AccordionTrigger>
            <AccordionContent>
              <p className="text-muted-foreground leading-relaxed">
                Le fond coloré représente la température de l'air. Les
                particules montrent la direction et la vitesse du vent (traits
                plus longs = vent plus fort).
              </p>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </aside>
  );
}