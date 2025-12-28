import { motion } from 'motion/react'
import { ArrowLeft, Cpu, CheckCircle, AlertTriangle, AlertCircle } from 'lucide-react'
import { Button } from '../ui/button'
import { Card } from '../ui/card'
import { PageHeader } from '../PageHeader'
import { PageFooter } from '../PageFooter'

interface DecisionsPageProps {
  onBack: () => void
}

export default function DecisionsPage({ onBack }: DecisionsPageProps) {
  const scenarios = [
    {
      icon: CheckCircle,
      title: 'Scénario Normal',
      gradient: 'from-green-500 to-emerald-500',
      description: 'Conditions optimales détectées',
    },
    {
      icon: AlertTriangle,
      title: 'Scénario Attention',
      gradient: 'from-yellow-500 to-orange-500',
      description: 'Surveillance accrue requise',
    },
    {
      icon: AlertCircle,
      title: 'Scénario Critique',
      gradient: 'from-red-500 to-pink-500',
      description: 'Action immédiate nécessaire',
    },
  ]

  const recommendations = [
    'Réduire la cadence de production de 20%',
    'Activer le système anti-odeurs zone Nord',
    'Surveiller les émissions SO₂ pendant 2h',
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50 dark:from-[#0B0F0C] dark:to-[#0E1411] relative">
      {/* Page Header with Logo and Theme Toggle */}
      <PageHeader onLogoClick={onBack} />

      <div className="p-4 md:p-8 pt-24">
        {/* Back Button */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
          className="mb-6"
        >
          <Button
            onClick={onBack}
            variant="outline"
            size="icon"
            className="rounded-full w-12 h-12 bg-white dark:bg-[rgba(0,0,0,0.6)] hover:scale-110 transition-transform border-2 border-purple-500"
            aria-label="Retour à l'accueil"
          >
            <ArrowLeft size={20} style={{ color: '#a855f7' }} />
          </Button>
        </motion.div>

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="mb-8"
        >
          <div className="flex items-center gap-4 mb-4">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
              <Cpu size={24} className="text-white" />
            </div>
            <div className="flex-1">
              <h1 
                className="text-3xl md:text-4xl dark:text-white text-[#1A2A23]"
                style={{ fontFamily: "'Playfair Display', serif" }}
              >
                Décisions IoT
              </h1>
              <p className="dark:text-[#EAF7F0]/70 text-[#1A2A23]/70">
                Scénarios automatiques et recommandations IA
              </p>
            </div>
            <div className="hidden md:flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/10 border border-green-500/30">
              <span className="w-2 h-2 bg-green-400 rounded-full" />
              <span className="text-sm font-semibold text-green-400">Système Actif</span>
            </div>
          </div>
        </motion.div>

        {/* Scenarios Grid */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
        {scenarios.map((scenario, index) => (
          <motion.div
            key={scenario.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1, duration: 0.4 }}
            whileHover={{ scale: 1.05, y: -8 }}
          >
            <Card className="p-6 rounded-2xl glass-card border-2 hover:shadow-xl transition-all">
              <div className={`w-12 h-12 rounded-2xl bg-gradient-to-r ${scenario.gradient} flex items-center justify-center mb-4`}>
                <scenario.icon size={24} className="text-white" />
              </div>
              <h3 
                className="text-lg font-semibold dark:text-white text-[#1A2A23] mb-2"
                style={{ fontFamily: 'var(--font-heading)' }}
              >
                {scenario.title}
              </h3>
              <p className="text-sm dark:text-[#EAF7F0]/70 text-[#1A2A23]/70">
                {scenario.description}
              </p>
            </Card>
          </motion.div>
        ))}
        </div>

        {/* Recommendations Panel */}
        <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.4 }}
        >
          <Card className="p-8 rounded-3xl glass-card shadow-lg">
            <h2 
              className="text-2xl dark:text-white text-[#1A2A23] mb-6"
              style={{ fontFamily: 'var(--font-heading)' }}
            >
              Recommandations en temps réel
            </h2>
            <div className="space-y-4">
              {recommendations.map((rec, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 + index * 0.1 }}
                  className="flex items-start gap-4 p-4 rounded-xl bg-purple-50 dark:bg-purple-900/10 border border-purple-100 dark:border-purple-800/20"
                >
                  <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center flex-shrink-0 text-white font-semibold">
                    {index + 1}
                  </div>
                  <div>
                    <h4 className="font-semibold dark:text-white text-[#1A2A23] mb-1">
                      Recommandation {index + 1}
                    </h4>
                    <p className="text-sm dark:text-[#EAF7F0]/80 text-[#1A2A23]/80">
                      {rec}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </Card>
        </motion.div>

        {/* Footer */}
        <PageFooter className="mt-12" />
      </div>
    </div>
  )
}
