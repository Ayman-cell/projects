import { motion } from 'motion/react'
import { CloudRain } from 'lucide-react'
import { Card } from '../ui/card'
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from '../ui/accordion'

export default function ForecastingSection() {
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
                  rotate: [0, -10, 10, 0]
                }}
                transition={{ 
                  duration: 5, 
                  repeat: Infinity,
                  ease: 'easeInOut',
                  delay: 1
                }}
                whileHover={{ scale: 1.2, rotate: 360 }}
                className="w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0 dark:bg-black/40 dark:border-cyan-500/30 bg-cyan-500/20 border border-cyan-500/40"
              >
                <CloudRain size={32} className="text-cyan-600 dark:text-cyan-400" />
              </motion.div>
              <div>
                <h2 className="text-3xl font-bold dark:text-white text-[#1A2A23] mb-3" style={{ fontFamily: 'var(--font-heading)' }}>
                  Prévisions Météorologiques
                </h2>
                <p className="dark:text-[#EAF7F0]/80 text-[#1A2A23]/80 leading-relaxed" style={{ fontFamily: 'var(--font-body)' }}>
                  Modèles de Machine Learning générant des prévisions toutes les 3 heures pour optimiser les opérations et anticiper les conditions critiques
                </p>
              </div>
            </div>

            {/* Accordion */}
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="technical-details" className="border-cyan-500/20">
                <AccordionTrigger className="text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300">
                  <span className="font-semibold">Détails Techniques</span>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="p-6 rounded-xl glass-card space-y-6">
                    {/* Horizon et fréquence */}
                    <div>
                      <h4 className="font-semibold text-cyan-600 dark:text-cyan-400 mb-2">
                        Horizon et fréquence
                      </h4>
                      <ul className="list-disc list-inside space-y-1 dark:text-[#EAF7F0]/80 text-[#1A2A23]/80">
                        <li>Prévisions générées toutes les 3 heures</li>
                        <li>Horizon de prévision : 12-24 heures</li>
                        <li>Variables : Vitesse du vent, direction, température, humidité, pression</li>
                      </ul>
                    </div>

                    {/* Modèles ML */}
                    <div>
                      <h4 className="font-semibold text-cyan-600 dark:text-cyan-400 mb-2">
                        Modèles ML utilisés
                      </h4>
                      <ul className="list-disc list-inside space-y-1 dark:text-[#EAF7F0]/80 text-[#1A2A23]/80">
                        <li>LightGBM pour prévisions météorologiques optimisées</li>
                        <li>XGBoost pour les prévisions à court terme (24h, 48h, 72h)</li>
                        <li>Fusion avec données Open-Meteo pour précision accrue</li>
                        <li>TensorFlow disponible pour modèles deep learning avancés</li>
                      </ul>
                    </div>

                    {/* Évaluation */}
                    <div>
                      <h4 className="font-semibold text-cyan-600 dark:text-cyan-400 mb-2">
                        Évaluation des performances
                      </h4>
                      <ul className="list-disc list-inside space-y-1 dark:text-[#EAF7F0]/80 text-[#1A2A23]/80">
                        <li>RMSE (Root Mean Square Error) et MAE (Mean Absolute Error)</li>
                        <li>Précision cible : {'>'} 85% pour les prévisions à 6h</li>
                        <li>Comparaison avec les données réelles post-opération</li>
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