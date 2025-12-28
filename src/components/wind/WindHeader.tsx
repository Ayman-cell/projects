import { RefreshCw, HelpCircle, LayoutDashboard } from 'lucide-react';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';

interface WindHeaderProps {
  isLive: boolean;
  lastUpdate: Date;
  onRefresh: () => void;
  onHelpClick: () => void;
  onBackToDashboard?: () => void;
  onLogoClick?: () => void;
}

export function WindHeader({ isLive, lastUpdate, onRefresh, onHelpClick, onBackToDashboard, onLogoClick }: WindHeaderProps) {
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('fr-FR', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <header className="h-14 border-b bg-card px-6 flex items-center justify-between z-40 shadow-sm">
      <div className="flex items-center gap-4">
        <h1 className="text-foreground" style={{ fontSize: '1.1rem', fontFamily: 'var(--font-heading)', fontWeight: 600 }}>Carte de Visualisation</h1>
      </div>

      <div className="flex items-center gap-3">
        {onBackToDashboard && (
          <Button
            variant="outline"
            size="sm"
            onClick={onBackToDashboard}
            className="gap-2 h-9"
          >
            <LayoutDashboard className="h-4 w-4" />
            Tableau de bord
          </Button>
        )}
        
        <Badge
          variant={isLive ? 'default' : 'secondary'}
          className={`${
            isLive
              ? 'bg-primary hover:bg-primary/90 animate-pulse'
              : 'bg-muted-foreground/20'
          }`}
        >
          {isLive ? 'Live' : 'Gelé'}
        </Badge>

        <span className="text-sm text-muted-foreground">
          Dernière mise à jour: {formatTime(lastUpdate)}
        </span>

        <Button
          variant="outline"
          size="icon"
          onClick={onHelpClick}
          className="rounded-xl h-9 w-9"
          title="Aide"
        >
          <HelpCircle className="h-4 w-4" />
        </Button>

        <Button
          variant="outline"
          size="icon"
          onClick={onRefresh}
          className="rounded-xl h-9 w-9"
          title="Actualiser"
        >
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}