import { motion } from 'motion/react'
import { LayoutDashboard, Wind, FileText, Info, HelpCircle, Home, Sun, Moon } from 'lucide-react'
import airboardLogo from 'figma:asset/bb2587f1136162e67af6f30b9f134dff67fd3295.png'
import { useTheme } from './ThemeContext'

type Page = 'home' | 'dashboard' | 'map' | 'rapports' | 'about-us' | 'how-it-works'

interface TopBarProps {
  currentPage: Page
  onNavigate: (page: Page) => void
}

export default function TopBar({ currentPage, onNavigate }: TopBarProps) {
  const { theme, toggleTheme } = useTheme()
  
  const navItems = [
    { label: 'Accueil', page: 'home' as const, icon: Home },
    { label: 'Dashboard', page: 'dashboard' as const, icon: LayoutDashboard },
    { label: 'WindMap', page: 'map' as const, icon: Wind },
    { label: 'Rapports', page: 'rapports' as const, icon: FileText },
    { label: 'À propos', page: 'about-us' as const, icon: Info },
    { label: 'Fonctionnement', page: 'how-it-works' as const, icon: HelpCircle },
  ]

  return (
    <motion.header
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="fixed top-0 left-0 right-0 h-16 glass-card border-b border-[rgba(47,163,111,0.2)] dark:border-[rgba(14,107,87,0.15)] flex items-center justify-between px-6 z-40"
    >
      {/* Logo */}
      <motion.div
        whileHover={{ scale: 1.03 }}
        whileTap={{ scale: 0.97 }}
        onClick={() => onNavigate('home')}
        className="flex items-center gap-2 cursor-pointer"
      >
        <img src={airboardLogo} alt="Airboard Logo" className="h-8 w-8" />
        <span 
          className="text-xl tracking-tight"
          style={{ 
            fontFamily: "'Playfair Display', serif",
            background: 'linear-gradient(135deg, #0A4D3C 0%, #0E6B57 40%, #2FA36F 70%, #0E6B57 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            fontWeight: 800,
          }}
        >
          AirBoard
        </span>
      </motion.div>

      {/* Navigation Items */}
      <nav className="flex items-center gap-1">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = currentPage === item.page
          
          return (
            <motion.button
              key={item.page}
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onNavigate(item.page)}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300
                ${isActive 
                  ? 'bg-gradient-to-r from-[#0E6B57] via-[#2FA36F] to-[#0E6B57] text-white shadow-md shadow-emerald-500/30' 
                  : 'dark:text-white text-[#1A2A23] hover:bg-emerald-500/10'
                }
              `}
              style={{ fontFamily: 'var(--font-heading)' }}
            >
              <Icon size={18} />
              <span className="text-sm">{item.label}</span>
            </motion.button>
          )
        })}
      </nav>

      {/* Theme Toggle */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={toggleTheme}
        className="flex items-center gap-2 px-3 py-2 rounded-lg glass-card transition-all hover:border-emerald-500/50 dark:text-white text-[#1A2A23]"
        style={{ fontFamily: 'var(--font-heading)' }}
        aria-label={`Passer au mode ${theme === 'light' ? 'sombre' : 'clair'}`}
      >
        {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
      </motion.button>
    </motion.header>
  )
}
