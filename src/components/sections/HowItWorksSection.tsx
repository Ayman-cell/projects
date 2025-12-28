import { motion } from 'motion/react'
import { Radio, Brain, Bell } from 'lucide-react'

export default function HowItWorksSection() {
  const steps = [
    {
      icon: Radio,
      badge: 'Étape 1',
      title: 'Collecte Continue',
      description: 'Des capteurs répartis sur le site collectent en temps réel les données d\'émissions, météo et qualité de l\'air',
      color: '#2FA36F',
    },
    {
      icon: Brain,
      badge: 'Étape 2',
      title: 'Prévisions ML',
      description: 'Modèles de Machine Learning génèrent des prévisions météorologiques toutes les 3 heures avec une précision optimale',
      color: '#79D6A3',
    },
    {
      icon: Bell,
      badge: 'Étape 3',
      title: 'Scénarios & Alertes',
      description: 'Le système génère automatiquement des scénarios (vert/jaune/rouge) avec des actions recommandées pour chaque situation',
      color: '#0E6B57',
    },
  ]

  return (
    <section className="py-28 md:py-36 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-24"
        >
          <h2 
            className="gradient-text mb-6"
            style={{ 
              fontFamily: 'var(--font-heading)',
              fontSize: 'clamp(2rem, 4vw, 3.5rem)',
              fontWeight: 700,
              lineHeight: 1.25
            }}
          >
            Comment Ça Marche
          </h2>
          <p className="text-xl dark:text-[#EAF7F0] text-[#1A2A23] opacity-85 max-w-3xl mx-auto" style={{ fontFamily: 'var(--font-body)' }}>
            Un système intelligent en trois étapes pour une gestion environnementale proactive
          </p>
        </motion.div>

        {/* Steps Grid */}
        <div className="relative grid md:grid-cols-3 gap-10">
          {steps.map((step, index) => (
            <motion.div
              key={step.title}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.15, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              whileHover={{ y: -12, scale: 1.03 }}
              className="relative group"
            >
              <div className="glass-card p-8 h-full relative overflow-hidden transition-all duration-300 hover:shadow-xl">
                {/* Badge */}
                <div 
                  className="inline-block px-4 py-1.5 rounded-full text-xs font-bold mb-6 border"
                  style={{ 
                    backgroundColor: `${step.color}20`,
                    color: step.color,
                    borderColor: `${step.color}40`,
                    fontFamily: 'var(--font-heading)'
                  }}
                >
                  {step.badge}
                </div>

                {/* Icon - avec animation continue */}
                <motion.div 
                  animate={{ 
                    y: [0, -10, 0],
                    rotate: [0, 10, -10, 0]
                  }}
                  transition={{ 
                    duration: 5, 
                    repeat: Infinity,
                    ease: 'easeInOut',
                    delay: index * 0.7
                  }}
                  whileHover={{ scale: 1.2, rotate: 360 }}
                  className="w-20 h-20 rounded-2xl flex items-center justify-center mb-6 border dark:bg-black/40 dark:border-emerald-500/30 bg-emerald-500/20 border-emerald-500/40"
                >
                  <step.icon size={40} className="text-emerald-600 dark:text-emerald-400" />
                </motion.div>

                {/* Content */}
                <h3 
                  className="text-2xl font-bold dark:text-white text-[#1A2A23] mb-4 mt-2" 
                  style={{ fontFamily: 'var(--font-heading)' }}
                >
                  {step.title}
                </h3>
                <p 
                  className="dark:text-[#EAF7F0]/80 text-[#1A2A23]/80 leading-relaxed" 
                  style={{ fontFamily: 'var(--font-body)' }}
                >
                  {step.description}
                </p>

                {/* Decorative gradient */}
                <div 
                  className="absolute bottom-0 left-0 right-0 h-1 opacity-50"
                  style={{ 
                    background: `linear-gradient(90deg, transparent, ${step.color}, transparent)`
                  }}
                />
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}