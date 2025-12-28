import { motion } from 'motion/react'
import { Target, Lightbulb, Shield } from 'lucide-react'

export default function MissionSection() {
  const features = [
    {
      icon: Lightbulb,
      title: 'Innovation Continue',
      description: 'Utilisation de l\'IA et du Machine Learning pour anticiper et optimiser les émissions industrielles',
    },
    {
      icon: Target,
      title: 'Objectifs Clairs',
      description: 'Réduction de 30% de l\'impact environnemental tout en maintenant l\'efficacité opérationnelle',
    },
    {
      icon: Shield,
      title: 'Conformité Garantie',
      description: 'Respect des normes ISO 14001 et des réglementations environnementales locales et internationales',
    },
  ]

  return (
    <section id="mission" className="py-24 md:py-32 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-5 py-2 rounded-full glass-card mb-6">
            <Target size={18} className="text-emerald-500 dark:text-emerald-400" />
            <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400" style={{ fontFamily: 'var(--font-heading)' }}>
              Notre Mission
            </span>
          </div>
          <h2 
            className="gradient-text mb-6"
            style={{ 
              fontFamily: 'var(--font-heading)',
              fontSize: 'clamp(2rem, 4vw, 3.5rem)',
              fontWeight: 700,
              lineHeight: 1.25
            }}
          >
            Réduire l'Impact Environnemental
          </h2>
        </motion.div>

        {/* Content Card */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          className="glass-card p-8 md:p-12"
        >
          {/* Intro */}
          <p className="text-center dark:text-[#EAF7F0] text-[#1A2A23] opacity-90 mb-14 max-w-4xl mx-auto leading-relaxed text-lg" style={{ fontFamily: 'var(--font-body)' }}>
            Le projet Airboard a été conçu pour <strong className="text-emerald-600 dark:text-emerald-400 font-semibold">réduire l'impact environnemental du site Safi</strong> en combinant surveillance en temps réel, prévisions météorologiques avancées et recommandations automatiques basées sur l'intelligence artificielle.
            <br /><br />
            <strong className="text-emerald-600 dark:text-emerald-400 font-semibold">Avec Monsieur Hicham Smaiti</strong>, Responsable Projet HSE — OCP Safi.
          </p>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                whileHover={{ y: -8, scale: 1.03 }}
                className="group relative"
              >
                <div className="glass-card p-8 h-full flex flex-col items-center text-center transition-all duration-300 hover:shadow-xl">
                  {/* Icon - avec animation continue */}
                  <motion.div
                    animate={{ 
                      y: [0, -8, 0],
                      rotate: [0, 5, -5, 0]
                    }}
                    transition={{ 
                      duration: 4, 
                      repeat: Infinity,
                      ease: 'easeInOut',
                      delay: index * 0.5
                    }}
                    whileHover={{ scale: 1.2, rotate: 360 }}
                    className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6 border dark:bg-black/40 dark:border-emerald-500/30 bg-emerald-500/20 border-emerald-500/40"
                  >
                    <feature.icon size={32} className="text-emerald-600 dark:text-emerald-400" />
                  </motion.div>

                  {/* Content */}
                  <h3 className="text-xl font-bold dark:text-white text-[#1A2A23] mb-3" style={{ fontFamily: 'var(--font-heading)' }}>
                    {feature.title}
                  </h3>
                  <p className="dark:text-[#EAF7F0]/80 text-[#1A2A23]/80 leading-relaxed" style={{ fontFamily: 'var(--font-body)' }}>
                    {feature.description}
                  </p>

                  {/* Hover glow */}
                  <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-emerald-500/0 to-cyan-500/0 group-hover:from-emerald-500/5 group-hover:to-cyan-500/5 transition-all duration-300 pointer-events-none" />
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}