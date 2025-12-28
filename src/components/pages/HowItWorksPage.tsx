import { motion } from 'motion/react'
import TopBar from '../TopBar'
import { Radio, Brain, Database, Cloud, Bell, TrendingUp, Shield, Map, FileText, MessageSquare, BarChart3, Wind } from 'lucide-react'
import { ImageWithFallback } from '../figma/ImageWithFallback'

type Page = 'home' | 'dashboard' | 'map' | 'rapports' | 'about-us' | 'how-it-works'

interface HowItWorksPageProps {
  onNavigate?: (page: Page) => void
}

export default function HowItWorksPage({ onNavigate }: HowItWorksPageProps = {}) {
  const steps = [
    {
      icon: Radio,
      title: 'Collecte des Données Météorologiques',
      description: 'Le système collecte en temps réel les données météorologiques depuis les fichiers GP2 générés par les stations de mesure.',
      details: [
        'Lecture automatique des fichiers GP2_*.txt dans le dossier data',
        'Données météorologiques : vitesse du vent, direction, température, humidité',
        'Données d\'alimentation (tension)',
        'Mise à jour automatique avec le fichier le plus récent'
      ],
      image: 'https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?w=1080&q=80&auto=format&fit=crop'
    },
    {
      icon: BarChart3,
      title: 'Dashboard en Temps Réel',
      description: 'Visualisation des 10 dernières mesures avec mise à jour automatique toutes les 30 secondes.',
      details: [
        'Affichage des dernières 10 mesures en temps réel',
        'Graphiques de séries temporelles avec échelles adaptatives',
        'Rose des vents interactive avec calcul automatique des scénarios',
        'Tableaux de données horaires avec indicateurs de scénarios (S1, S2, S3, S4)',
        'Métriques clés : température, vitesse vent, direction, humidité, alimentation'
      ],
      image: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1080&q=80&auto=format&fit=crop'
    },
    {
      icon: Map,
      title: 'Windmap - Visualisation Interactive',
      description: 'Carte météorologique interactive intégrant l\'interface Info Windy avec visualisation 2D et 3D du vent.',
      details: [
        'Carte 2D Leaflet avec particules de vent animées',
        'Globe 3D Cesium pour visualisation mondiale',
        'Données météorologiques en temps réel (Open-Meteo API)',
        'Prévisions ML intégrées avec animation temporelle',
        'Chatbot météo intégré pour questions sur les conditions',
        'Fusion Helmholtz pour champs de vent optimisés'
      ],
      image: 'https://images.unsplash.com/photo-1504608524841-42fe6f032b4b?w=1080&q=80&auto=format&fit=crop'
    },
    {
      icon: Brain,
      title: 'Prévisions par Machine Learning',
      description: 'Modèles de Machine Learning génèrent des prévisions météorologiques avec fusion des données Open-Meteo.',
      details: [
        'Modèles LightGBM et XGBoost pour prévisions',
        'Prévisions à court terme (24h, 48h, 72h)',
        'Fusion avec données Open-Meteo pour précision optimale',
        'Intégration dans la timeline du Windmap',
        'Mise à jour automatique toutes les 3 heures'
      ],
      image: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1080&q=80&auto=format&fit=crop'
    },
    {
      icon: Wind,
      title: 'Système de Scénarios Automatiques',
      description: 'Calcul automatique des scénarios opérationnels basés sur la vitesse et la direction du vent.',
      details: [
        'Scénario S1 : Conditions optimales (vent < 12 m/s, direction favorable)',
        'Scénario S2/S2b : Vigilance (vent modéré, direction à surveiller)',
        'Scénario S3/S3b : Attention (vent fort, direction défavorable)',
        'Scénario S4 : Conditions critiques (vent très fort)',
        'Recommandations automatiques par scénario',
        'Affichage en temps réel dans le dashboard et les tableaux'
      ],
      image: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1080&q=80&auto=format&fit=crop'
    },
    {
      icon: MessageSquare,
      title: 'Chatbot Météo Intelligent',
      description: 'Assistant conversationnel intégré utilisant RAG (Retrieval Augmented Generation) avec LLM Cerebras.',
      details: [
        'Interface de chat style Messenger dans le Windmap',
        'Réponses contextuelles basées sur les données météorologiques',
        'Analyse des conditions actuelles et prévisions',
        'Recommandations personnalisées selon les scénarios',
        'Historique de conversation sauvegardé'
      ],
      image: 'https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=1080&q=80&auto=format&fit=crop'
    },
    {
      icon: FileText,
      title: 'Génération de Rapports PDF',
      description: 'Système complet de génération de rapports d\'analyse avec graphiques et analyses détaillées.',
      details: [
        'Sélection de période personnalisée (date début/fin)',
        'Analyse automatique des données avec LLM (Gemini/Cerebras)',
        'Génération de graphiques : séries temporelles, corrélations, distributions, boxplots, rose des vents',
        'Rapport Markdown structuré avec analyses détaillées',
        'Export PDF professionnel avec tous les graphiques intégrés',
        'Adaptation du contenu selon l\'audience (Opérateurs, Management, Ingénieurs)'
      ],
      image: 'https://images.unsplash.com/photo-1450101499163-c8848c66ca85?w=1080&q=80&auto=format&fit=crop'
    }
  ]

  return (
    <>
      <TopBar currentPage="how-it-works" onNavigate={onNavigate || (() => {})} />
      <div className="min-h-screen p-8 pt-24">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-16"
          >
            <div className="inline-flex items-center gap-2 px-5 py-2 rounded-full bg-emerald-500/10 border border-emerald-400/30 backdrop-blur-sm mb-6">
              <Shield size={18} className="text-emerald-400" />
              <span className="text-sm font-bold text-emerald-400" style={{ fontFamily: 'var(--font-heading)' }}>
                Fonctionnement
              </span>
            </div>
            
            <h1 
              className="gradient-text mb-4"
              style={{ 
                fontFamily: 'var(--font-heading)',
                fontSize: 'clamp(2.5rem, 5vw, 4rem)',
                fontWeight: 800,
              }}
            >
              Comment le système fonctionne ?
            </h1>
            
            <p className="text-lg max-w-3xl mx-auto dark:text-white/70 text-[#1A2A23]/70" style={{ fontFamily: 'var(--font-body)' }}>
              Découvrez toutes les fonctionnalités du système de monitoring météorologique : dashboard en temps réel, visualisation interactive du vent, génération de rapports et chatbot intelligent
            </p>
          </motion.div>

          {/* Steps */}
          <div className="space-y-12 mb-16">
            {steps.map((step, index) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="glass-card rounded-3xl p-8"
              >
                <div className="flex flex-col md:flex-row items-start gap-6">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-500/30 to-cyan-500/30 flex items-center justify-center border border-emerald-500/40 flex-shrink-0">
                    <step.icon size={32} className="text-emerald-600 dark:text-emerald-400" />
                  </div>
                  
                  <div className="flex-1">
                    <div className="mb-4">
                      <div className="text-sm font-bold text-emerald-600 dark:text-emerald-400 mb-1">
                        Étape {index + 1}
                      </div>
                      <h3 className="text-2xl dark:text-white text-[#1A2A23] mb-2" style={{ fontFamily: 'var(--font-heading)' }}>
                        {step.title}
                      </h3>
                      <p className="dark:text-white/70 text-[#1A2A23]/70" style={{ fontFamily: 'var(--font-body)' }}>
                        {step.description}
                      </p>
                    </div>
                    
                    <ul className="space-y-2 mb-4">
                      {step.details.map((detail, i) => (
                        <li key={i} className="flex items-start gap-3 dark:text-white/60 text-[#1A2A23]/60">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-2 flex-shrink-0" />
                          <span className="text-sm" style={{ fontFamily: 'var(--font-body)' }}>{detail}</span>
                        </li>
                      ))}
                    </ul>

                    {/* Image Placeholder */}
                    <div className="w-full h-64 rounded-xl overflow-hidden border border-emerald-500/20 mt-4">
                      <ImageWithFallback
                        src={step.image}
                        alt={step.title}
                        className="w-full h-full object-cover"
                      />
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Tech Stack */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="glass-card rounded-3xl p-8 md:p-12"
          >
            <h2 
              className="text-3xl dark:text-white text-[#1A2A23] mb-8 text-center"
              style={{ fontFamily: 'var(--font-heading)' }}
            >
              Technologies Utilisées
            </h2>
            
            <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
              {[
                { category: 'Frontend', tech: 'React, TypeScript, Tailwind CSS, Motion, Plotly, Recharts' },
                { category: 'Backend', tech: 'Python, Flask, Pandas, Plotly, FPDF2, LLMs (Cerebras, Gemini)' },
                { category: 'Visualisation', tech: 'Leaflet.js, CesiumJS, Open-Meteo API, Helmholtz Fusion' }
              ].map((item, i) => (
                <div key={i} className="text-center">
                  <div className="font-bold text-emerald-600 dark:text-emerald-400 mb-2" style={{ fontFamily: 'var(--font-heading)' }}>
                    {item.category}
                  </div>
                  <div className="text-sm dark:text-white/70 text-[#1A2A23]/70" style={{ fontFamily: 'var(--font-body)' }}>
                    {item.tech}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </>
  )
}