import { motion, useScroll, useTransform } from 'motion/react'
import { ArrowRight, Mail } from 'lucide-react'
import { Button } from '../ui/button'
import { useRef } from 'react'

export default function HeroSection() {
  const containerRef = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end start'],
  })

  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0.2])
  const y = useTransform(scrollYProgress, [0, 0.5], [0, -50])

  const handleScrollTo = (id: string) => {
    const element = document.getElementById(id)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <motion.section
      ref={containerRef}
      style={{ opacity }}
      className="min-h-screen flex flex-col items-center justify-center px-4 md:px-8 pt-32 pb-16"
    >
      <div className="max-w-6xl mx-auto text-center">
        {/* Logo Badge */}
        <motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          whileHover={{ scale: 1.05 }}
          className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full glass-card mb-10 shadow-lg"
        >
          <motion.span
            animate={{
              scale: [1, 1.4, 1],
              opacity: [1, 0.6, 1],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
            className="w-2 h-2 bg-emerald-500 dark:bg-emerald-400 rounded-full shadow-lg shadow-emerald-400/50"
          />
          <span 
            className="text-sm font-bold dark:text-white text-[#1A2A23]" 
            style={{ fontFamily: 'var(--font-heading)' }}
          >
            Airboard — OCP SAFI
          </span>
          <div className="flex items-center gap-2 pl-3 ml-3 border-l border-emerald-500/30 dark:border-white/30">
            <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400">LIVE</span>
          </div>
        </motion.div>

        {/* Main Title - Police élégante Playfair Display */}
        <motion.div
          style={{ y }}
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          className="mb-12"
        >
          <h1 
            style={{ 
              fontFamily: "'Playfair Display', serif",
              fontSize: 'clamp(2.5rem, 5vw, 5.5rem)',
              lineHeight: 1.1,
              letterSpacing: '-0.02em',
              fontWeight: 800
            }}
          >
            <motion.span 
              className="block dark:text-white text-[#1A2A23]"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.25 }}
            >
              Pilotage Intelligent
            </motion.span>
            <motion.span 
              className="relative inline-block mt-3"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.35 }}
            >
              <span 
                style={{ 
                  fontFamily: "'Playfair Display', serif",
                  fontWeight: 800,
                  background: 'linear-gradient(135deg, #0E6B57 0%, #2FA36F 50%, #51C57B 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text'
                }}
                className="dark:text-white text-[#0E6B57]"
              >
                des Émissions
              </span>
              {/* Animated Underline - plus fin et mieux placé */}
              <motion.span
                className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 h-0.5 rounded-full"
                style={{
                  background: 'linear-gradient(90deg, #2FA36F, #85D5FF, #2FA36F)',
                  backgroundSize: '200% 100%',
                  width: '90%'
                }}
                animate={{
                  backgroundPosition: ['0% 0%', '100% 0%', '0% 0%'],
                }}
                transition={{
                  duration: 4,
                  repeat: Infinity,
                  ease: 'linear',
                }}
              />
            </motion.span>
          </h1>
        </motion.div>

        {/* Subtitle - plus belle */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="text-lg md:text-xl dark:text-[#EAF7F0] text-[#1A2A23] opacity-85 text-center max-w-4xl mx-auto mb-14 leading-relaxed"
          style={{ fontFamily: 'var(--font-body)', fontWeight: 400 }}
        >
          Système intelligent de surveillance et d'optimisation des émissions industrielles 
          <span className="block mt-1">en temps réel pour le site OCP Safi</span>
        </motion.p>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="flex flex-col items-center gap-3 dark:text-white/60 text-[#1A2A23]/60"
        >
          <span className="text-sm font-medium" style={{ fontFamily: 'var(--font-body)' }}>
            Découvrir le projet
          </span>
          <div className="w-6 h-11 border-2 dark:border-white/40 border-[#1A2A23]/40 rounded-full p-1.5 backdrop-blur-sm">
            <motion.div
              animate={{ y: [0, 12, 0] }}
              transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
              className="w-1.5 h-1.5 dark:bg-white bg-[#1A2A23] rounded-full mx-auto shadow-lg"
            />
          </div>
        </motion.div>
      </div>
    </motion.section>
  )
}