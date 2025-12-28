import { useState } from 'react'
import { ThemeProvider } from './components/ThemeContext'
import { DataDirProvider } from './contexts/DataDirContext'
import ShaderBackground from './components/ShaderBackground'
import HomePage from './components/HomePage'
import DashboardPage from './components/pages/DashboardPage'
import MapPage from './components/pages/MapPage'
import RapportsPage from './components/pages/RapportsPage'
import AboutUsPage from './components/pages/AboutUsPage'
import HowItWorksPage from './components/pages/HowItWorksPage'
import { Toaster } from './components/ui/sonner'

type Page = 'home' | 'dashboard' | 'map' | 'rapports' | 'about-us' | 'how-it-works'

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('home')

  const handleNavigate = (page: Page) => {
    setCurrentPage(page)
  }

  return (
    <ThemeProvider>
      <DataDirProvider>
      <ShaderBackground>
        {currentPage === 'home' && <HomePage onNavigate={handleNavigate} />}
        {currentPage === 'dashboard' && <DashboardPage onNavigate={handleNavigate} />}
        {currentPage === 'map' && <MapPage onNavigate={handleNavigate} />}
        {currentPage === 'rapports' && <RapportsPage onNavigate={handleNavigate} />}
        {currentPage === 'about-us' && <AboutUsPage onNavigate={handleNavigate} />}
        {currentPage === 'how-it-works' && <HowItWorksPage onNavigate={handleNavigate} />}

        {/* Global Toast Notifications */}
        <Toaster position="top-right" richColors closeButton />
      </ShaderBackground>
      </DataDirProvider>
    </ThemeProvider>
  )
}