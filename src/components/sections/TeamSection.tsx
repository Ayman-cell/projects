import { motion } from 'motion/react'
import { Users, Mail, FileText } from 'lucide-react'
import { Button } from '../ui/button'

export default function TeamSection() {
  const contactMethods = [
    {
      icon: Mail,
      title: 'Contact Email',
      value: 'hse@ocp-safi.ma',
      link: 'mailto:hse@ocp-safi.ma',
      color: '#2FA36F',
    },
    {
      icon: FileText,
      title: 'Documentation',
      value: 'Télécharger PDF',
      link: '#',
      color: '#85D5FF',
    },
    {
      icon: FileText,
      title: 'Rapports RSE',
      value: 'Consulter',
      link: '#',
      color: '#79D6A3',
    },
  ]

  return (
    <section id="team" className="py-24 md:py-32 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-5 py-2 rounded-full bg-emerald-500/10 border border-emerald-400/30 backdrop-blur-sm mb-6">
            <Users size={18} className="text-emerald-400" />
            <span className="text-sm font-bold text-emerald-400" style={{ fontFamily: 'var(--font-heading)' }}>Contact</span>
          </div>
          <h2 
            className="gradient-text"
            style={{ 
              fontFamily: 'var(--font-heading)',
              fontSize: 'clamp(2rem, 4vw, 3.5rem)',
              fontWeight: 700,
              lineHeight: 1.2
            }}
          >
            Équipe & Contact
          </h2>
        </motion.div>

        {/* Main Card */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          className="glass-card rounded-3xl p-8 md:p-12"
        >
          {/* Contact Info */}
          <div className="text-center mb-12">
            <h3 
              className="text-3xl md:text-4xl font-bold text-white mb-3"
              style={{ fontFamily: 'var(--font-heading)' }}
            >
              M. Hicham Smaiti
            </h3>
            <p className="text-xl text-emerald-400 font-semibold" style={{ fontFamily: 'var(--font-body)' }}>
              Responsable Projet HSE — OCP Safi
            </p>
          </div>

          {/* Contact Methods */}
          <div className="grid md:grid-cols-3 gap-6 mb-10">
            {contactMethods.map((method, index) => (
              <motion.a
                key={method.title}
                href={method.link}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.4 }}
                whileHover={{ y: -6, scale: 1.03 }}
                className="group"
              >
                <div className="glass-card rounded-2xl p-6 text-center h-full transition-all duration-300 hover:border-white/30 hover:shadow-xl relative overflow-hidden">
                  {/* Background gradient on hover */}
                  <div 
                    className="absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity duration-300"
                    style={{ background: `linear-gradient(135deg, ${method.color}, transparent)` }}
                  />
                  
                  <method.icon 
                    size={36} 
                    className="mx-auto mb-4 group-hover:scale-110 transition-transform duration-300" 
                    style={{ color: method.color }}
                  />
                  <h4 className="font-bold text-white mb-2 text-lg" style={{ fontFamily: 'var(--font-heading)' }}>
                    {method.title}
                  </h4>
                  <p 
                    className="text-sm font-semibold group-hover:underline" 
                    style={{ 
                      color: method.color,
                      fontFamily: 'var(--font-body)'
                    }}
                  >
                    {method.value}
                  </p>
                </div>
              </motion.a>
            ))}
          </div>

          {/* CTA Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.3, duration: 0.4 }}
            className="text-center"
          >
            <motion.div 
              whileHover={{ scale: 1.05, y: -4 }} 
              whileTap={{ scale: 0.98 }}
            >
              <Button
                size="lg"
                onClick={() => window.location.href = 'mailto:hse@ocp-safi.ma'}
                className="h-14 px-10 bg-gradient-to-r from-[#0E6B57] via-[#2FA36F] to-[#0E6B57] text-white font-bold rounded-xl shadow-2xl shadow-emerald-500/30 hover:shadow-emerald-500/50 transition-all duration-300 border border-white/20"
                style={{ 
                  fontFamily: 'var(--font-heading)',
                  backgroundSize: '200% 100%',
                }}
              >
                <Mail size={22} className="mr-3" />
                Contacter l'équipe HSE
              </Button>
            </motion.div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}
