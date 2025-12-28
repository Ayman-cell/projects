import { motion } from 'motion/react'
import { TrendingUp, Clock, Target, Shield } from 'lucide-react'

export default function KPISection() {
  const kpis = [
    {
      icon: Clock,
      title: 'Fréquence',
      value: 'Temps réel',
      description: 'Génération de scénarios',
    },
    {
      icon: Target,
      title: 'Automatisation',
      value: '>70%',
      description: 'Objectif Phase 1',
    },
    {
      icon: Shield,
      title: 'Précision',
      value: '95%+',
      description: 'Détection proactive',
    },
    {
      icon: TrendingUp,
      title: 'Réduction',
      value: '-30%',
      description: 'Impact environnemental',
    },
  ]

  const technicalMetrics = [
    { label: 'RMSE & MAE', value: 'Optimisés', description: 'Prévisions météo' },
    { label: 'API Latency', value: '<200ms', description: 'P95 response time' },
    { label: 'Uptime', value: '99.5%+', description: 'SLA contractuel' },
  ]

  return (
    <section className="py-24 md:py-32 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-20"
        >
          <div className="inline-flex items-center gap-2 px-5 py-2 rounded-full glass-card mb-6">
            <TrendingUp size={18} className="text-emerald-500 dark:text-emerald-400" />
            <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400" style={{ fontFamily: 'var(--font-heading)' }}>
              Indicateurs de Performance
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
            KPI & Bénéfices
          </h2>
          <p className="text-xl dark:text-[#EAF7F0] text-[#1A2A23] opacity-85 max-w-3xl mx-auto" style={{ fontFamily: 'var(--font-body)' }}>
            Mesure de l'impact opérationnel et environnemental du système
          </p>
        </motion.div>

        {/* KPI Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          {kpis.map((kpi, index) => (
            <motion.div
              key={kpi.title}
              initial={{ opacity: 0, y: 30, scale: 0.9 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              whileHover={{ y: -8, scale: 1.03 }}
              className="group relative"
            >
              <div className="glass-card p-8 text-center h-full relative overflow-hidden transition-all duration-300 hover:shadow-xl">
                {/* Icon - avec animation */}
                <motion.div 
                  animate={{ 
                    y: [0, -10, 0],
                    rotate: [0, 10, -10, 0]
                  }}
                  transition={{ 
                    duration: 5, 
                    repeat: Infinity,
                    ease: 'easeInOut',
                    delay: index * 0.5
                  }}
                  whileHover={{ scale: 1.2, rotate: 360 }}
                  className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-6 dark:bg-black/40 dark:border-emerald-500/30 bg-emerald-500/20 border border-emerald-500/40"
                >
                  <kpi.icon size={32} className="text-emerald-600 dark:text-emerald-400" />
                </motion.div>

                {/* Value */}
                <div 
                  className="text-4xl md:text-5xl font-black mb-3 text-emerald-600 dark:text-emerald-400"
                  style={{ fontFamily: 'var(--font-heading)' }}
                >
                  {kpi.value}
                </div>

                {/* Title */}
                <div className="font-bold dark:text-white text-[#1A2A23] mb-2 text-lg" style={{ fontFamily: 'var(--font-heading)' }}>
                  {kpi.title}
                </div>

                {/* Description */}
                <div className="text-sm dark:text-[#EAF7F0]/70 text-[#1A2A23]/70" style={{ fontFamily: 'var(--font-body)' }}>
                  {kpi.description}
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Technical Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="glass-card p-10 md:p-12"
        >
          <h3 
            className="text-2xl md:text-3xl font-bold text-center dark:text-white text-[#1A2A23] mb-10"
            style={{ fontFamily: 'var(--font-heading)' }}
          >
            Métriques Techniques
          </h3>
          <div className="grid md:grid-cols-3 gap-10">
            {technicalMetrics.map((metric, index) => (
              <motion.div 
                key={metric.label} 
                className="text-center group"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.4 }}
                whileHover={{ y: -4 }}
              >
                <div className="text-3xl md:text-4xl font-black text-emerald-600 dark:text-emerald-400 mb-2" style={{ fontFamily: 'var(--font-heading)' }}>
                  {metric.value}
                </div>
                <div className="font-bold dark:text-white text-[#1A2A23] mb-1 text-lg" style={{ fontFamily: 'var(--font-heading)' }}>
                  {metric.label}
                </div>
                <div className="text-sm dark:text-[#EAF7F0]/70 text-[#1A2A23]/70" style={{ fontFamily: 'var(--font-body)' }}>
                  {metric.description}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}
