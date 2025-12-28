# Airboard — OCP Safi Environmental Monitoring Dashboard

A real-time environmental monitoring and emissions control system for the OCP Safi industrial site, featuring ML-based weather forecasting, automated scenario generation, and interactive visualizations.

## 🎯 Features

- **Real-time Dashboard**: Live monitoring of ~50 sensors across the site
- **ML Weather Forecasting**: Predictions every 3 hours using SARIMA, XGBoost, and LSTM models
- **Automated Scenarios**: Green/Yellow/Red status with actionable recommendations
- **Interactive Maps**: Windy-style weather and emissions visualization
- **Dark/Light Themes**: Theme toggle with localStorage persistence
- **Animated Background**: MeshGradient shader-based background (theme-aware)
- **Accessibility**: WCAG AA+ compliant, keyboard navigation, reduced-motion support
- **Responsive Design**: Mobile-first approach with Tailwind CSS

## 📁 File Structure

```
/
├── App.tsx                          # Main entry with state-based routing
├── components/
│   ├── ThemeContext.tsx             # Dark/Light theme provider
│   ├── ShaderBackground.tsx         # Animated mesh gradient background
│   ├── NavigationBar.tsx            # Fixed top navigation
│   ├── LaunchDialog.tsx             # Modal for quick navigation
│   ├── HomePage.tsx                 # Landing page container
│   ├── sections/                    # 11 Homepage sections
│   │   ├── HeroSection.tsx
│   │   ├── MissionSection.tsx
│   │   ├── HowItWorksSection.tsx
│   │   ├── ForecastingSection.tsx
│   │   ├── DataIngestionSection.tsx
│   │   ├── ScenariosSection.tsx
│   │   ├── InterfaceSection.tsx
│   │   ├── ArchitectureSection.tsx
│   │   ├── SecuritySection.tsx
│   │   ├── KPISection.tsx
│   │   └── TeamSection.tsx
│   ├── pages/                       # Sub-pages
│   │   ├── DashboardPage.tsx
│   │   ├── MapPage.tsx
│   │   └── DecisionsPage.tsx
│   └── ui/                          # shadcn/ui components
└── styles/
    └── globals.css                  # Global styles & CSS variables
```

## 🎨 Design System

### Color Palette

**Light Mode:**
- Background: `#F5F5F0` (beige)
- Text: `#1F2D24` (dark green-black)
- Primary: `#2FA36F` (mid-green)
- Secondary: `#79D6A3` (mint green)
- Accent: `#85D5FF` (sky blue)

**Dark Mode:**
- Background: `#0B0F0C` (deep dark green)
- Text: `#E9F3EE` (light cream)
- Primary: `#51C57B` (bright green)
- Secondary: `#85D5FF` (sky blue)
- Accent: `#B7EFB9` (light mint)

**Scenario Colors:**
- Green (Normal): `#51C57B`
- Yellow (Attention): `#F9C74F`
- Red (Critical): `#FF6B6B`

### Typography

- **Headings**: Inter (600-800 weight)
- **Body**: IBM Plex Sans / Figtree (400-500 weight)
- **Base Size**: 16px
- **Line Height**: 1.5 for body, 1.1-1.2 for headings
- **Max Width**: 68ch for optimal readability

## 🚀 Navigation Flow

### Entry Points

1. **Hamburger Menu** (left) → Opens side panel with 3 module links
2. **Launch App Button** (right) → Opens modal dialog with 3 option cards
3. **Hero CTAs** → Smooth scroll to #mission or #team sections

### Pages

- **Home** (`/`) — Landing page with 11 sections
- **Dashboard** — Real-time emissions monitoring
- **Map** — Interactive weather visualization
- **Decisions** — AI scenarios and recommendations

All sub-pages have a back button (top-left) to return to home.

## ⚙️ Key Technologies

- **React** — UI components and state management
- **Tailwind CSS v4** — Utility-first styling
- **Motion (Framer Motion)** — Smooth animations
- **@paper-design/shaders-react** — MeshGradient background
- **shadcn/ui** — Pre-built accessible components
- **Lucide React** — Icon library

## 🔧 Implementation Notes

### Reduced Motion Support

The app respects `prefers-reduced-motion` media query:
- Animations disabled when reduced motion is preferred
- Background shader animations paused
- Only essential transitions remain

### Theme Switching

Theme state is stored in `localStorage` with key `"airboard-theme"`:
- Default: Light mode
- Toggle via NavigationBar button (Sun/Moon icon)
- `.dark` class applied to `<html>` element
- CSS variables update automatically

### Smooth Scrolling

Hero section CTAs use smooth scrolling to target sections:
- "Explorer le système" → scrolls to `#mission`
- "Contacter l'équipe HSE" → scrolls to `#team`

### Performance Optimizations

- GPU-accelerated transforms (scale, translate, rotate)
- `viewport={{ once: true }}` prevents re-animation on scroll
- Animations use `transform` and `opacity` only
- No heavy box-shadow animations

## 📱 Responsive Behavior

### Breakpoints

- `sm`: 640px+ (small tablets)
- `md`: 768px+ (tablets)
- `lg`: 1024px+ (laptops)
- `xl`: 1280px+ (desktops)

### Mobile Adaptations

- Logo text hidden on small screens
- 3-column grids collapse to single column
- Navigation menu full-width on mobile
- CTA buttons stack vertically
- Touch-friendly tap targets (min 44×44px)

## 🔒 Security & Compliance

- **IT/OT Separation**: Network isolation for security
- **Authentication**: SSO with RBAC
- **Compliance**: ISO 14001, RGPD, OCP internal norms
- **Data**: Encryption at-rest and in-transit
- **Backups**: Daily automated backups

## 📊 KPIs

- **Frequency**: Real-time scenario generation
- **Automation**: >70% target (Phase 1)
- **Precision**: 95%+ proactive detection
- **Reduction**: -30% environmental impact
- **API Latency**: <200ms P95
- **Uptime**: 99.5%+ SLA

## 👥 Contact

**M. Hicham Smaiti**  
Responsable Projet HSE — OCP Safi  
Email: hse@ocp-safi.ma

---

**Airboard** — Copyright © 2025 — OCP Safi  
Powered by FastAPI, React, PostgreSQL/PostGIS & Machine Learning
