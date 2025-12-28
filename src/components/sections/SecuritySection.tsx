import { motion } from 'motion/react'
import { Shield, Lock, Eye, Database } from 'lucide-react'
import { Card } from '../ui/card'

export default function SecuritySection() {
  const securityFeatures = [
    {
      icon: Shield,
      title: 'IT/OT Sécurité',
      color: '#51C57B',
      items: [
        'Séparation réseau IT/OT',
        'VPN sécurisé pour accès distant',
        'DMZ pour isolation des services',
        'Firewall et IDS/IPS',
      ],
    },
    {
      icon: Lock,
      title: 'Authentification',
      color: '#79D6A3',
      items: [
        'SSO avec Active Directory',
        'RBAC (Role-Based Access Control)',
        'Audit trail de toutes les actions',
        '2FA pour comptes critiques',
      ],
    },
    {
      icon: Eye,
      title: 'Conformité',
      color: '#85D5FF',
      items: [
        'ISO 14001 (Environnement)',
        'RGPD pour données personnelles',
        'Normes OCP internes',
        'Audits réguliers',
      ],
    },
    {
      icon: Database,
      title: 'Données',
      color: '#0E6B57',
      items: [
        'Chiffrement at-rest et in-transit',
        'Backups quotidiens automatiques',
        'Redondance des serveurs critiques',
        'Rétention selon politique OCP',
      ],
    },
  ]

  return (
    <section className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[var(--accent-primary)]/10 border border-[var(--accent-primary)]/30 mb-4">
            <Shield size={16} className="text-[var(--accent-primary)]" />
            <span className="text-sm font-medium text-[var(--accent-primary)]">
              Sécurité & Conformité
            </span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-[var(--text-primary)] to-[var(--accent-primary)] bg-clip-text text-transparent mb-4">
            Protection & Conformité
          </h2>
          <p className="text-lg text-[var(--text-secondary)] max-w-3xl mx-auto">
            Sécurité IT/OT renforcée et conformité aux normes environnementales internationales
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {securityFeatures.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1, duration: 0.35 }}
              whileHover={{ y: -5, scale: 1.03 }}
            >
              <Card className="h-full p-6 rounded-3xl bg-[#0E1411]/50 dark:bg-[#0E1411]/50 border border-[var(--accent-primary)]/20 shadow-md hover:shadow-lg transition-all">
                {/* Icon */}
                <div 
                  className="w-14 h-14 rounded-2xl flex items-center justify-center mb-4 border-2"
                  style={{ 
                    backgroundColor: `${feature.color}15`,
                    borderColor: `${feature.color}40`
                  }}
                >
                  <feature.icon size={28} style={{ color: feature.color }} />
                </div>

                {/* Title */}
                <h3 className="text-lg font-semibold text-[#E9F3EE] mb-4">
                  {feature.title}
                </h3>

                {/* Items */}
                <ul className="space-y-2">
                  {feature.items.map((item, i) => (
                    <li key={i} className="flex items-start gap-2 text-[#C7DED3]">
                      <span 
                        className="w-1.5 h-1.5 rounded-full mt-2 flex-shrink-0"
                        style={{ backgroundColor: feature.color }}
                      />
                      <span className="text-sm">{item}</span>
                    </li>
                  ))}
                </ul>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
