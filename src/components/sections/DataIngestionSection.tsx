import { motion } from 'motion/react'
import { Database } from 'lucide-react'
import { Card } from '../ui/card'
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from '../ui/accordion'

export default function DataIngestionSection() {
  return (
    <section className="py-20 px-4">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
        >
          <Card className="p-8 md:p-12 rounded-3xl glass-card border border-cyan-500/20 shadow-lg">
            {/* Header */}
            <div className="flex flex-col md:flex-row items-start gap-6 mb-8">
              <motion.div 
                animate={{ 
                  y: [0, -10, 0],
                  rotate: [0, 10, -10, 0]
                }}
                transition={{ 
                  duration: 5, 
                  repeat: Infinity,
                  ease: 'easeInOut'
                }}
                whileHover={{ scale: 1.2, rotate: 360 }}
                className="w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0 dark:bg-black/40 dark:border-cyan-500/30 bg-cyan-500/20 border border-cyan-500/40"
              >
                <Database size={32} className="text-cyan-600 dark:text-cyan-400" />
              </motion.div>
              <div>
                <h2 className="text-3xl font-bold dark:text-white text-[#1A2A23] mb-3" style={{ fontFamily: 'var(--font-heading)' }}>
                  Données & Ingestion
                </h2>
                <p className="dark:text-[#EAF7F0]/80 text-[#1A2A23]/80 leading-relaxed" style={{ fontFamily: 'var(--font-body)' }}>
                  Collecte et traitement de données provenant d'environ 50 capteurs répartis sur le site, garantissant une traçabilité complète et une qualité optimale
                </p>
              </div>
            </div>

            {/* Accordion */}
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="data-details" className="border-cyan-500/20">
                <AccordionTrigger className="text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300">
                  <span className="font-semibold">Détails Techniques</span>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="p-6 rounded-xl glass-card space-y-6">
                    {/* Sources */}
                    <div>
                      <h4 className="font-semibold text-cyan-600 dark:text-cyan-400 mb-2">
                        Sources de données
                      </h4>
                      <ul className="list-disc list-inside space-y-1 dark:text-[#EAF7F0]/80 text-[#1A2A23]/80">
                        <li>Fichiers GP2_*.txt générés par les stations de mesure météorologiques</li>
                        <li>Données météorologiques : vitesse du vent, direction, température, humidité</li>
                        <li>Données d'alimentation (tension)</li>
                        <li>API Open-Meteo pour données météorologiques complémentaires</li>
                      </ul>
                    </div>

                    {/* Pipeline */}
                    <div>
                      <h4 className="font-semibold text-cyan-600 dark:text-cyan-400 mb-2">
                        Pipeline d'ingestion
                      </h4>
                      <ul className="list-disc list-inside space-y-1 dark:text-[#EAF7F0]/80 text-[#1A2A23]/80">
                        <li>Lecture automatique du fichier GP2 le plus récent dans le dossier data</li>
                        <li>Parsing des fichiers GP2 avec Pandas (format séparé par espaces)</li>
                        <li>Validation et nettoyage automatique des données (NaN, valeurs aberrantes)</li>
                        <li>Filtrage par plage de dates pour analyse et rapports</li>
                      </ul>
                    </div>

                    {/* Qualité */}
                    <div>
                      <h4 className="font-semibold text-cyan-600 dark:text-cyan-400 mb-2">
                        Politique qualité des données
                      </h4>
                      <ul className="list-disc list-inside space-y-1 dark:text-[#EAF7F0]/80 text-[#1A2A23]/80">
                        <li>Seuils d'alerte si {'>'} 10% de données manquantes sur 1h</li>
                        <li>Détection automatique des valeurs aberrantes</li>
                        <li>Traçabilité complète de chaque point de donnée</li>
                        <li>Rapports quotidiens de qualité des données</li>
                      </ul>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </Card>
        </motion.div>
      </div>
    </section>
  )
}