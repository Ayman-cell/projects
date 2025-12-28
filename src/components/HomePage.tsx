import { motion } from 'motion/react'
import TopBar from './TopBar'
import HeroSection from './sections/HeroSection'
import MissionSection from './sections/MissionSection'
import HowItWorksSection from './sections/HowItWorksSection'
import ForecastingSection from './sections/ForecastingSection'
import DataIngestionSection from './sections/DataIngestionSection'
import ScenariosSection from './sections/ScenariosSection'
import InterfaceSection from './sections/InterfaceSection'
import ArchitectureSection from './sections/ArchitectureSection'
import SecuritySection from './sections/SecuritySection'
import KPISection from './sections/KPISection'
import TeamSection from './sections/TeamSection'

type Page = 'home' | 'dashboard' | 'map' | 'rapports' | 'about-us' | 'how-it-works'

interface HomePageProps {
  onNavigate: (page: Page) => void
}

export default function HomePage({ onNavigate }: HomePageProps) {
  return (
    <>
      <TopBar currentPage="home" onNavigate={onNavigate} />
      <div className="relative pt-16" style={{ background: 'transparent' }}>
        <HeroSection />
        <MissionSection />
        <HowItWorksSection />
        <ForecastingSection />
        <DataIngestionSection />
        <ScenariosSection />
        <InterfaceSection />
        <ArchitectureSection />
        <SecuritySection />
        <KPISection />
        <TeamSection />
        
        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="py-12 text-center"
        >
          <p className="text-white/70 font-semibold mb-2" style={{ fontFamily: 'var(--font-heading)' }}>
            AirBoard — Copyright © 2025 — OCP Safi
          </p>
          <p className="text-sm text-white/50 mb-2" style={{ fontFamily: 'var(--font-body)' }}>
            Projet info de 1ère année Cycle Ingénieur — EMINES, École de Management Industriel
          </p>
          <p className="text-sm text-white/50 mb-2" style={{ fontFamily: 'var(--font-body)' }}>
            Université Mohammed VI Polytechnique de Benguerir
          </p>
          <p className="text-sm text-white/50" style={{ fontFamily: 'var(--font-body)' }}>
            Powered by Flask, React, Plotly & Machine Learning
          </p>
        </motion.footer>
      </div>
    </>
  )
}