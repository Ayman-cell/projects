import React, { useState } from 'react'
import { motion } from 'motion/react'
import { useDataDir } from '../../contexts/DataDirContext'
import TopBar from '../TopBar'
import { TimeFilterBar } from '../dashboard/TimeFilterBar'
import { LeftPanel } from '../dashboard/LeftPanel'
import { RightPanel } from '../dashboard/RightPanel'

type Page = 'home' | 'dashboard' | 'map' | 'rapports' | 'about-us' | 'how-it-works'

interface DashboardPageProps {
  onNavigate?: (page: Page) => void
}

export default function DashboardPage({ onNavigate }: DashboardPageProps) {
  const selectedStation = 'GP2'; // Station fixe - OCP Safi
  const [selectedDate, setSelectedDate] = useState(new Date())
  const [selectedPeriod, setSelectedPeriod] = useState<'day' | 'month' | 'year'>('day')
  const [selectedHour, setSelectedHour] = useState('09:00')
  const [isLive, setIsLive] = useState(true)
  const { dataDir } = useDataDir()

  return (
    <>
      <TopBar currentPage="dashboard" onNavigate={onNavigate || (() => {})} />
      <div className="h-screen flex flex-col overflow-hidden pt-16">
        {/* Top Filter Bar */}
        <TimeFilterBar
          selectedPeriod={selectedPeriod}
          onPeriodChange={setSelectedPeriod}
          selectedDate={selectedDate}
          onDateChange={setSelectedDate}
          selectedHour={selectedHour}
          onHourChange={setSelectedHour}
          isLive={isLive}
          onLiveChange={setIsLive}
        />

        {/* Main Dashboard */}
        <div className="flex gap-3 px-3 py-2 flex-1 overflow-hidden">
          {/* Left Panel - 30% */}
          <LeftPanel
            selectedStation={selectedStation}
            selectedDate={selectedDate}
            isLive={isLive}
            onNavigateToMap={() => onNavigate?.('map')}
          />

          {/* Right Panel - 70% */}
          <RightPanel
            selectedStation={selectedStation}
            selectedDate={selectedDate}
            selectedPeriod={selectedPeriod}
            selectedHour={selectedHour}
            isLive={isLive}
          />
        </div>
      </div>
    </>
  )
}