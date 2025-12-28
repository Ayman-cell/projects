import { motion } from 'motion/react'
import { Server } from 'lucide-react'
import { Card } from '../ui/card'
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from '../ui/accordion'

export default function ArchitectureSection() {
  return (
    <section className="py-20 px-4">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
        >
          <Card className="p-8 md:p-12 rounded-3xl bg-[#0E1411]/50 dark:bg-[#0E1411]/50 border border-[var(--accent-primary)]/20 shadow-lg">
            {/* Header */}
            <div className="flex flex-col md:flex-row items-start gap-6 mb-8">
              <div className="w-16 h-16 rounded-2xl bg-[var(--accent-primary)]/15 flex items-center justify-center flex-shrink-0">
                <Server size={32} className="text-[var(--accent-primary)]" />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-[#E9F3EE] mb-3">
                  Architecture Back-end & Base de Données
                </h2>
                <p className="text-[#C7DED3] leading-relaxed">
                  Backend sécurisé et scalable avec API REST/GraphQL pour servir toutes les données en temps réel
                </p>
              </div>
            </div>

            {/* Accordion */}
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="architecture-details" className="border-[var(--accent-primary)]/20">
                <AccordionTrigger className="text-[var(--accent-primary)] hover:text-[var(--accent-hover)]">
                  <span className="font-semibold">Détails de l'Architecture</span>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="p-6 rounded-xl bg-[var(--accent-primary)]/8 space-y-6">
                    {/* Tech Stack */}
                    <div>
                      <h4 className="font-semibold text-[var(--accent-primary)] mb-2">
                        Stack technologique
                      </h4>
                      <ul className="list-disc list-inside space-y-1 text-[#C7DED3]">
                        <li>Backend : Python (Flask) avec API REST</li>
                        <li>Stockage : Fichiers GP2_*.txt dans le dossier data</li>
                        <li>ML : LightGBM, XGBoost, TensorFlow pour prévisions météorologiques</li>
                        <li>LLMs : Cerebras et Google Gemini pour analyse et génération de rapports</li>
                        <li>Visualisation : Plotly, Pandas pour traitement et analyse de données</li>
                      </ul>
                    </div>

                    {/* Services */}
                    <div>
                      <h4 className="font-semibold text-[var(--accent-primary)] mb-2">
                        Services back-end
                      </h4>
                      <ul className="list-disc list-inside space-y-1 text-[#C7DED3]">
                        <li>Service de lecture automatique des fichiers GP2_*.txt</li>
                        <li>Pipeline de nettoyage et transformation avec Pandas</li>
                        <li>Prédictions ML avec LightGBM et XGBoost</li>
                        <li>Moteur de calcul de scénarios basé sur vitesse et direction du vent</li>
                        <li>API REST Flask pour frontend React</li>
                        <li>Génération de rapports PDF avec FPDF2 et analyse LLM</li>
                      </ul>
                    </div>

                    {/* Déploiement */}
                    <div>
                      <h4 className="font-semibold text-[var(--accent-primary)] mb-2">
                        Déploiement et infrastructure
                      </h4>
                      <ul className="list-disc list-inside space-y-1 text-[#C7DED3]">
                        <li>Serveur Flask local pour développement et déploiement</li>
                        <li>Frontend React avec Vite pour build et développement</li>
                        <li>Intégration Open-Meteo API pour données météorologiques</li>
                        <li>Fusion Helmholtz pour optimisation des champs de vent</li>
                      </ul>
                    </div>

                    {/* Monitoring */}
                    <div>
                      <h4 className="font-semibold text-[var(--accent-primary)] mb-2">
                        Monitoring système
                      </h4>
                      <ul className="list-disc list-inside space-y-1 text-[#C7DED3]">
                        <li>Health checks sur le serveur Flask (/api/health)</li>
                        <li>Mise à jour automatique des données toutes les 30 secondes</li>
                        <li>Cache serveur pour optimiser les performances</li>
                        <li>Logs détaillés pour debugging et suivi</li>
                      </ul>
                    </div>

                    {/* Versioning */}
                    <div>
                      <h4 className="font-semibold text-[var(--accent-primary)] mb-2">
                        Versioning et traçabilité
                      </h4>
                      <ul className="list-disc list-inside space-y-1 text-[#C7DED3]">
                        <li>Fichiers GP2 horodatés pour historique des données</li>
                        <li>Git pour versioning du code source</li>
                        <li>Stockage des modèles ML avec joblib</li>
                        <li>Génération de rapports PDF avec traçabilité complète</li>
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
