import React, { useState, useEffect } from 'react';
import { FolderOpen, HelpCircle, Radio } from 'lucide-react';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '../ui/dialog';
import { LiveClock } from './LiveClock';
import { useDataDir } from '../../contexts/DataDirContext';
import { toast } from 'sonner';

interface TimeFilterBarProps {
  selectedPeriod: 'day' | 'month' | 'year';
  onPeriodChange: (period: 'day' | 'month' | 'year') => void;
  selectedDate: Date;
  onDateChange: (date: Date) => void;
  selectedHour?: string;
  onHourChange?: (hour: string) => void;
  isLive?: boolean;
  onLiveChange?: (isLive: boolean) => void;
}

export function TimeFilterBar({ selectedPeriod, onPeriodChange, selectedDate, onDateChange, selectedHour, onHourChange, isLive, onLiveChange }: TimeFilterBarProps) {
  const { dataDir, setDataDir } = useDataDir();
  const [dataFolderPath, setDataFolderPath] = useState(dataDir || '');
  const [helpOpen, setHelpOpen] = useState(false);
  const fileInputRef = useState<HTMLInputElement | null>(null)[0];
  const [fileInputRefState, setFileInputRefState] = useState<HTMLInputElement | null>(null);

  // Synchroniser avec le contexte dataDir
  useEffect(() => {
    // Ne pas afficher "data" dans le champ si c'est le dossier par défaut
    // (pour éviter la confusion)
    if (dataDir === null || dataDir === '') {
      setDataFolderPath('');
    } else {
      setDataFolderPath(dataDir);
    }
  }, [dataDir]);

  const handleLoadData = async () => {
    const trimmedPath = dataFolderPath.trim();
    if (trimmedPath) {
      // Mettre à jour le contexte global
      setDataDir(trimmedPath);
      // Synchroniser avec le backend pour l'interface Windy
      try {
        await fetch('http://127.0.0.1:5000/api/data-dir', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data_dir: trimmedPath })
        });
      } catch (error) {
        console.error('Erreur lors de la synchronisation du data_dir:', error);
      }
    } else {
      // Revenir au dossier par défaut
      setDataDir(null);
      // Synchroniser avec le backend
      try {
        await fetch('http://127.0.0.1:5000/api/data-dir', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data_dir: null })
        });
      } catch (error) {
        console.error('Erreur lors de la synchronisation du data_dir:', error);
      }
    }
  };

  const handleResetToDefault = async () => {
    setDataFolderPath('');
    setDataDir(null);
    // Synchroniser avec le backend
    try {
      await fetch('http://127.0.0.1:5000/api/data-dir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data_dir: null })
      });
    } catch (error) {
      console.error('Erreur lors de la synchronisation du data_dir:', error);
    }
  };

  const handleFolderSelect = async () => {
    // Essayer d'utiliser l'API File System Access (Chrome/Edge) si disponible
    if ('showDirectoryPicker' in window) {
      try {
        const directoryHandle = await (window as any).showDirectoryPicker();
        // Obtenir le nom du dossier
        const folderName = directoryHandle.name;
        // Note: On ne peut pas obtenir le chemin complet pour des raisons de sécurité
        // On utilise le nom du dossier comme identifiant
        // L'utilisateur devra peut-être entrer le chemin complet manuellement si nécessaire
        setDataFolderPath(folderName);
        setDataDir(folderName);
        // Synchroniser avec le backend
        try {
          await fetch('http://127.0.0.1:5000/api/data-dir', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data_dir: folderName })
          });
        } catch (error) {
          console.error('Erreur lors de la synchronisation du data_dir:', error);
        }
        toast.success(`Dossier sélectionné: ${folderName}`);
      } catch (error: any) {
        // L'utilisateur a annulé la sélection
        if (error.name !== 'AbortError') {
          console.error('Erreur lors de la sélection du dossier:', error);
          toast.error('Erreur lors de la sélection du dossier');
        }
      }
    } else {
      // Fallback: utiliser l'input file avec webkitdirectory
      if (fileInputRefState) {
        fileInputRefState.click();
      }
    }
  };

  const handleFolderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      // Extraire le chemin du dossier depuis le premier fichier
      const firstFile = files[0];
      
      // Essayer d'obtenir le chemin complet (non disponible dans tous les navigateurs)
      const filePath = (firstFile as any).path;
      
      if (filePath) {
        // Extraire le chemin du dossier (sans le nom du fichier)
        const pathParts = filePath.split(/[\\/]/);
        pathParts.pop(); // Enlever le nom du fichier
        const folderFullPath = pathParts.join('/');
        setDataFolderPath(folderFullPath);
        setDataDir(folderFullPath);
        toast.success(`Dossier sélectionné: ${folderFullPath}`);
      } else if (firstFile.webkitRelativePath) {
        // Utiliser webkitRelativePath pour obtenir le nom du dossier
        const relativePath = firstFile.webkitRelativePath;
        const folderName = relativePath.split('/')[0];
        setDataFolderPath(folderName);
        setDataDir(folderName);
        toast.success(`Dossier sélectionné: ${folderName}`);
        toast.info('Pour un chemin complet, entrez-le manuellement dans le champ');
      } else {
        // Fallback: utiliser le nom du premier fichier
        const fileName = firstFile.name;
        setDataFolderPath(fileName);
        setDataDir(fileName);
        toast.info('Veuillez entrer le chemin complet du dossier manuellement');
      }
    }
    // Réinitialiser l'input pour permettre de sélectionner le même dossier à nouveau
    event.target.value = '';
  };


  return (
    <>
      <div className="bg-white/80 dark:bg-[rgba(0,0,0,0.6)] backdrop-blur-md shadow-sm border-b border-[rgba(47,163,111,0.2)] dark:border-[rgba(14,107,87,0.15)] px-6 py-3 mt-0">
        {/* Single Row with all controls - FIXED HEIGHT */}
        <div className="flex items-center justify-between gap-4 h-10">
          {/* Left: Title */}
          <h1 
            className="gradient-text flex-shrink-0"
            style={{ 
              fontFamily: "'Playfair Display', serif", 
              fontSize: '1.75rem', 
              fontWeight: 800,
              letterSpacing: '-0.02em'
            }}
          >
            Tableau de bord
          </h1>
          
          {/* Center: Controls - Compact horizontal layout */}
          <div className="flex items-center gap-2 flex-1 justify-center">
            {/* Data Folder Input - More compact */}
            <div className="flex items-center gap-1">
              <Input
                type="text"
                placeholder="Chemin des données (défaut: data)..."
                value={dataFolderPath}
                onChange={(e) => setDataFolderPath(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleLoadData();
                  }
                }}
                className="w-48 h-9 text-xs"
                title={dataFolderPath || "Cliquez sur le bouton dossier pour sélectionner un dossier, ou entrez le chemin manuellement"}
              />
              {/* Input file caché pour la sélection de dossier */}
              <input
                type="file"
                ref={(el) => setFileInputRefState(el)}
                style={{ display: 'none' }}
                {...({ webkitdirectory: '', directory: '' } as any)}
                multiple
                onChange={handleFolderChange}
              />
              <Button
                variant="outline"
                size="icon"
                onClick={handleFolderSelect}
                className="h-9 w-9 flex-shrink-0"
                title="Sélectionner un dossier depuis l'explorateur de fichiers"
              >
                <FolderOpen className="h-4 w-4" />
              </Button>
              {dataDir && (
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleResetToDefault}
                  className="h-9 w-9 flex-shrink-0"
                  title="Revenir au dossier par défaut (data)"
                >
                  <span className="text-xs">↩</span>
                </Button>
              )}
            </div>

            {/* Live Button - Fixed indicator (non-clickable) */}
            <Button
              disabled
              className="h-9 px-3 font-semibold text-xs bg-gradient-to-r from-[#0E6B57] via-[#2FA36F] to-[#0E6B57] text-white shadow-lg shadow-emerald-500/30 cursor-default"
              style={{ fontFamily: 'var(--font-heading)' }}
            >
              <Radio className="mr-1.5 h-3.5 w-3.5" />
              LIVE
            </Button>

            {/* Live Clock - Inline compact */}
            <LiveClock />
          </div>

          {/* Right: Help Button */}
          <Button
            variant="outline"
            size="icon"
            onClick={() => setHelpOpen(true)}
            className="h-9 w-9 flex-shrink-0"
            title="Aide"
          >
            <HelpCircle className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Help Dialog - Improved Content */}
      <Dialog open={helpOpen} onOpenChange={setHelpOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle style={{ fontFamily: "'Playfair Display', serif", fontSize: '1.5rem', fontWeight: 700 }}>
              Guide du Tableau de Bord AirBoard
            </DialogTitle>
            <DialogDescription>
              Manuel d'utilisation complet pour exploiter les fonctionnalités du tableau de bord
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-6 text-sm">
            <div className="border-l-4 border-emerald-500 pl-4">
              <h4 className="font-semibold mb-2 text-base">📊 Métriques Principales</h4>
              <p className="text-muted-foreground leading-relaxed">
                Les cartes métriques affichent en temps réel les dernières valeurs mesurées pour la température, 
                la vitesse du vent, la direction et l'humidité relative. Ces données sont mises à jour automatiquement 
                toutes les 5 minutes pour garantir la précision des informations affichées.
              </p>
            </div>
            
            <div className="border-l-4 border-blue-500 pl-4">
              <h4 className="font-semibold mb-2 text-base">📈 Graphiques Temporels</h4>
              <p className="text-muted-foreground leading-relaxed">
                Visualisez l'évolution des paramètres environnementaux sur la période sélectionnée. 
                Vous pouvez choisir d'afficher les données par <strong>jour</strong> (détail heure par heure), 
                <strong>mois</strong> (agrégation quotidienne) ou <strong>année</strong> (tendances mensuelles). 
                Les courbes sont interactives : survolez-les pour obtenir des valeurs précises à un instant donné.
              </p>
            </div>
            
            <div className="border-l-4 border-cyan-500 pl-4">
              <h4 className="font-semibold mb-2 text-base">🧭 Rose des Vents</h4>
              <p className="text-muted-foreground leading-relaxed">
                La rose des vents montre la distribution des directions du vent sur la période analysée. 
                Plus le segment est long, plus le vent vient fréquemment de cette direction. Les couleurs 
                représentent l'intensité : du bleu (vents faibles) au rouge (vents forts). Utilisez cet outil 
                pour identifier les vents dominants et planifier les opérations sensibles aux conditions météorologiques.
              </p>
            </div>
            
            <div className="border-l-4 border-purple-500 pl-4">
              <h4 className="font-semibold mb-2 text-base">📁 Charger des Données Personnalisées</h4>
              <p className="text-muted-foreground leading-relaxed">
                Vous pouvez charger des données depuis un dossier personnalisé en entrant le chemin complet 
                dans le champ prévu (ex: <code className="bg-muted px-1 py-0.5 rounded">/data/mesures/2025/janvier</code>) 
                et en cliquant sur l'icône dossier. Le système accepte les fichiers aux formats CSV, TXT et Excel. 
                Assurez-vous que les données respectent le format attendu (timestamp, température, vent, etc.).
              </p>
            </div>
            
            <div className="border-l-4 border-teal-500 pl-4">
              <h4 className="font-semibold mb-2 text-base">🔄 Navigation et Export</h4>
              <p className="text-muted-foreground leading-relaxed">
                Utilisez la <strong>Carte de visualisation</strong> (accessible via le panneau latéral gauche) 
                pour une vue cartographique interactive des données météorologiques. Les données peuvent être 
                exportées via l'API REST du backend pour des analyses approfondies ou des rapports personnalisés.
              </p>
            </div>
            
            <div className="bg-emerald-50 dark:bg-emerald-950/20 p-4 rounded-lg">
              <h4 className="font-semibold mb-2 text-base flex items-center gap-2">
                <span>💡</span>Astuce Pro
              </h4>
              <p className="text-muted-foreground text-xs leading-relaxed">
                Pour une analyse comparative, ouvrez plusieurs onglets avec des stations différentes 
                ou des périodes distinctes. Les données sont mises en cache pour un chargement rapide.
              </p>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}