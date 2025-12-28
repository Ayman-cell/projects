import { motion } from 'motion/react'
import { GitBranch, CheckCircle, AlertTriangle, XCircle } from 'lucide-react'
import { Card } from '../ui/card'
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from '../ui/accordion'

export default function ScenariosSection() {
  const statuses = [
    {
      icon: CheckCircle,
      color: '#51C57B',
      label: 'Vert',
      status: 'Normal',
      description: 'Conditions optimales',
    },
    {
      icon: AlertTriangle,
      color: '#F9C74F',
      label: 'Jaune',
      status: 'Attention',
      description: 'Surveillance accrue',
    },
    {
      icon: XCircle,
      color: '#FF6B6B',
      label: 'Rouge',
      status: 'Critique',
      description: 'Action immédiate',
    },
  ]

  return (
    <section className="py-20 px-4">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
        >
          <Card className="p-8 md:p-12 rounded-3xl bg-[var(--bg-card)] border border-[var(--card-border)] shadow-lg">
            {/* Header */}
            <div className="flex flex-col md:flex-row items-start gap-6 mb-8">
              <div className="w-16 h-16 rounded-2xl bg-[#85D5FF]/15 flex items-center justify-center flex-shrink-0">
                <GitBranch size={32} className="text-[#85D5FF]" />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-[var(--text-primary)] mb-3">
                  Scénarios & Actions Automatiques
                </h2>
                <p className="text-[var(--text-secondary)] leading-relaxed">
                  Génération automatique de scénarios (vert/jaune/rouge) avec recommandations d'actions adaptées à chaque situation
                </p>
              </div>
            </div>

            {/* Status Grid */}
            <div className="grid md:grid-cols-3 gap-6 mb-8">
              {statuses.map((status, index) => (
                <motion.div
                  key={status.label}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ y: -5, scale: 1.03 }}
                >
                  <Card 
                    className="p-6 rounded-2xl border-2 hover:shadow-lg transition-all"
                    style={{ borderColor: status.color }}
                  >
                    <status.icon size={32} style={{ color: status.color }} className="mb-3" />
                    <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-1">
                      {status.label} — {status.status}
                    </h3>
                    <p className="text-sm text-[var(--text-secondary)]">
                      {status.description}
                    </p>
                  </Card>
                </motion.div>
              ))}
            </div>

            {/* Accordion */}
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="scenario-details">
                <AccordionTrigger className="text-[var(--accent-primary)] hover:text-[var(--accent-hover)]">
                  <span className="font-semibold">Détails des Scénarios</span>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="p-6 rounded-xl bg-[var(--accent-primary)]/8 space-y-6">
                    {/* Actions */}
                    <div>
                      <h4 className="font-semibold text-[var(--accent-primary)] mb-2">
                        Actions recommandées
                      </h4>
                      <ul className="list-disc list-inside space-y-1 text-[var(--text-secondary)]">
                        <li>Réduire ou augmenter la cadence de production</li>
                        <li>Ajuster les systèmes de ventilation</li>
                        <li>Surveiller les émissions en temps réel</li>
                        <li>Déclencher les systèmes anti-odeurs</li>
                        <li>Notifier les équipes HSE</li>
                      </ul>
                    </div>

                    {/* Logique */}
                    <div>
                      <h4 className="font-semibold text-[var(--accent-primary)] mb-2">
                        Logique des règles
                      </h4>
                      <ul className="list-disc list-inside space-y-1 text-[var(--text-secondary)]">
                        <li>Combinaison de seuils d'émissions et prévisions météo</li>
                        <li>Règles métier définies avec les équipes HSE</li>
                        <li>Prise en compte de l'historique récent (tendances)</li>
                        <li>Adaptation selon le contexte opérationnel</li>
                      </ul>
                    </div>

                    {/* Priorisation */}
                    <div>
                      <h4 className="font-semibold text-[var(--accent-primary)] mb-2">
                        Priorisation
                      </h4>
                      <ul className="list-disc list-inside space-y-1 text-[var(--text-secondary)]">
                        <li>Score de risque de 0 à 100 pour chaque scénario</li>
                        <li>Priorisation automatique selon l'urgence</li>
                        <li>Escalade vers les responsables si nécessaire</li>
                      </ul>
                    </div>

                    {/* Automatisation */}
                    <div>
                      <h4 className="font-semibold text-[var(--accent-primary)] mb-2">
                        Automatisation
                      </h4>
                      <ul className="list-disc list-inside space-y-1 text-[var(--text-secondary)]">
                        <li>Phase 1 : Recommandations semi-automatiques</li>
                        <li>Objectif : &gt;70% d'automatisation des actions</li>
                        <li>Validation humaine pour les actions critiques</li>
                      </ul>
                    </div>

                    {/* Traçabilité */}
                    <div>
                      <h4 className="font-semibold text-[var(--accent-primary)] mb-2">
                        Traçabilité
                      </h4>
                      <ul className="list-disc list-inside space-y-1 text-[var(--text-secondary)]">
                        <li>Tous les scénarios stockés en base de données</li>
                        <li>Journaux d'audit pour chaque décision</li>
                        <li>Rapports mensuels de performance</li>
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
