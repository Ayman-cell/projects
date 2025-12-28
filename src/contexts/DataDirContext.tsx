import { createContext, useContext, useState, ReactNode } from 'react'

interface DataDirContextType {
  dataDir: string | null
  setDataDir: (dataDir: string | null) => void
}

const DataDirContext = createContext<DataDirContextType | undefined>(undefined)

export function DataDirProvider({ children }: { children: ReactNode }) {
  const [dataDir, setDataDir] = useState<string | null>(null)

  return (
    <DataDirContext.Provider value={{ dataDir, setDataDir }}>
      {children}
    </DataDirContext.Provider>
  )
}

export function useDataDir() {
  const context = useContext(DataDirContext)
  if (context === undefined) {
    throw new Error('useDataDir must be used within a DataDirProvider')
  }
  return context
}

