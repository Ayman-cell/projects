import { motion } from 'motion/react'
import { Monitor, Smartphone } from 'lucide-react'

export default function InterfaceSection() {
  const interfaces = [
    {
      icon: Monitor,
      title: 'Dashboard Web',
      description: 'Interface de visualisation complète style Windy avec carte interactive',
      color: '#51C57B',
      features: [
        'Carte interactive avec couches vent/émissions',
        'Graphiques temps réel (recharts)',
        'Tableaux de bord personnalisables',
        'Export PDF/CSV des rapports',
      ],
      slideFrom: 'left',
    },
    {
      icon: Smartphone,
      title: 'Application Mobile',
      description: 'App iOS/Android pour suivi en mobilité et notifications push',
      color: '#85D5FF',
      features: [
        'Notifications push pour scénarios critiques',
        'Vue synthétique des KPI',
        'Actions rapides (valider/rejeter)',
        'Mode hors ligne',
      ],
      slideFrom: 'right',
    },
  ]

  return (
    <section className="py-24 md:py-32 px-4 relative">
      <div className="max-w-6xl mx-auto relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
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
            Interfaces Utilisateur
          </h2>
          <p className="text-xl dark:text-[#EAF7F0] text-[#1A2A23] opacity-85 max-w-3xl mx-auto" style={{ fontFamily: 'var(--font-body)' }}>
            Dashboard web interactif et application mobile pour un accès complet aux données en temps réel
          </p>
        </motion.div>

        {/* Interfaces Grid */}
        <div className="grid md:grid-cols-2 gap-8">
          {interfaces.map((item, index) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, x: item.slideFrom === 'left' ? -50 : 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.2, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              whileHover={{ y: -8, scale: 1.03 }}
            >
              <div 
                className="glass-card p-8 h-full relative overflow-hidden transition-all duration-300 hover:shadow-xl"
                style={{
                  backgroundColor: 'var(--card)',
                  border: `2px solid ${item.color}40`
                }}
              >
                {/* Icon */}
                <div 
                  className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6 border-2"
                  style={{ 
                    backgroundColor: `${item.color}20`,
                    borderColor: `${item.color}60`
                  }}
                >
                  <item.icon size={32} style={{ color: item.color }} />
                </div>

                {/* Content */}
                <h3 className="text-2xl font-bold dark:text-white text-[#1A2A23] mb-3" style={{ fontFamily: 'var(--font-heading)' }}>
                  {item.title}
                </h3>
                <p className="dark:text-[#EAF7F0]/80 text-[#1A2A23]/80 mb-6 leading-relaxed" style={{ fontFamily: 'var(--font-body)' }}>
                  {item.description}
                </p>

                {/* Features */}
                <ul className="space-y-3">
                  {item.features.map((feature, i) => (
                    <li key={i} className="flex items-start gap-3 dark:text-[#EAF7F0] text-[#1A2A23]">
                      <span 
                        className="w-1.5 h-1.5 rounded-full mt-2 flex-shrink-0"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-sm leading-relaxed" style={{ fontFamily: 'var(--font-body)' }}>{feature}</span>
                    </li>
                  ))}
                </ul>

                {/* Bottom gradient accent */}
                <div 
                  className="absolute bottom-0 left-0 right-0 h-1 opacity-60"
                  style={{ background: `linear-gradient(90deg, transparent, ${item.color}, transparent)` }}
                />
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
