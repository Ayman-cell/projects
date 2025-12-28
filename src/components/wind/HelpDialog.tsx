import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { Button } from '../ui/button';

interface HelpDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function HelpDialog({ open, onOpenChange }: HelpDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[640px] max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Aide — Carte des vents</DialogTitle>
          <DialogDescription>
            Guide complet pour comprendre et utiliser la carte
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Lire la carte */}
          <div className="space-y-2">
            <h3 className="text-primary font-semibold">Lire la carte</h3>
            <p className="text-foreground dark:text-muted-foreground leading-relaxed">
              Le sol est coloré selon la température de l'air. Les particules
              indiquent la direction d'où vient le vent et sa vitesse par la
              longueur.
            </p>
          </div>

          {/* Source et Interpolation */}
          <div className="space-y-2">
            <h3 className="text-primary font-semibold">Source des données</h3>
            <p className="text-foreground dark:text-muted-foreground leading-relaxed">
              Les données sont <strong>interpolées</strong> entre les prévisions Open-Meteo 
              et les mesures en temps réel de la station GP1 (capteurs sur site).
              <br />
              Open-Meteo · Heure locale Africa/Casablanca
              <br />
              Grille 21×21 points · Mise à jour toutes les 5 secondes en mode Live
            </p>
          </div>

          {/* Temps */}
          <div className="space-y-2">
            <h3 className="text-primary font-semibold">Temps</h3>
            <p className="text-foreground dark:text-muted-foreground leading-relaxed">
              Le mode Live met à jour toutes les 5 secondes. Le curseur permet
              de revoir jusqu'à 3 heures dans le passé (-3h) ou de prévoir jusqu'à 3 heures dans le futur (+3h). 
              Bouger le curseur met en pause le Live.
            </p>
          </div>

          {/* Contrôles rapides */}
          <div className="space-y-2">
            <h3 className="text-primary font-semibold">Contrôles rapides</h3>
            <ul className="space-y-2 text-foreground dark:text-muted-foreground leading-relaxed">
              <li>
                • <strong>Température Air</strong> règle l'opacité du fond
                thermique
              </li>
              <li>
                • <strong>Vent (particules)</strong> active les traits de vent
                et leur densité
              </li>
              <li>
                • <strong>Reprendre Live</strong> revient au mode temps réel
              </li>
            </ul>
          </div>

          {/* Interprétation */}
          <div className="space-y-2">
            <h3 className="text-primary font-semibold">Interprétation</h3>
            <p className="text-foreground dark:text-muted-foreground leading-relaxed">
              <strong>Couleurs:</strong> bleu = frais, jaune = modéré, rouge =
              chaud.
              <br />
              <strong>Particules:</strong> plus longues = vent plus fort.
              <br />
              La direction montre d'où vient le vent (convention
              météorologique).
            </p>
          </div>

          {/* Station GP1 */}
          <div className="space-y-2">
            <h3 className="text-primary font-semibold">Station GP1</h3>
            <p className="text-foreground dark:text-muted-foreground leading-relaxed">
              La station GP1 est marquée sur la carte par une flèche rouge avec l'indication "GP1". 
              Elle correspond aux capteurs installés sur le site principal.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 pt-4 border-t">
          <Button onClick={() => onOpenChange(false)} className="flex-1">
            Compris
          </Button>
          <Button
            variant="outline"
            onClick={() =>
              window.open('https://open-meteo.com/', '_blank')
            }
          >
            Plus d'infos
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}