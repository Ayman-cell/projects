import React, { Component, ReactNode } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { Button } from './ui/button';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log to error reporting service (e.g., Sentry)
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-[#0B0F0C] dark:to-[#0E1411] p-4">
          <div className="glass-card p-8 md:p-12 max-w-md text-center">
            <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-br from-red-500/30 to-pink-500/30 flex items-center justify-center">
              <AlertCircle size={32} className="text-red-600 dark:text-red-400" />
            </div>

            <h1 
              className="text-2xl md:text-3xl font-bold dark:text-white text-[#1A2A23] mb-4"
              style={{ fontFamily: "'Playfair Display', serif" }}
            >
              Oups ! Une erreur s'est produite
            </h1>

            <p className="dark:text-[#EAF7F0]/80 text-[#1A2A23]/80 mb-6 leading-relaxed">
              Nous sommes désolés, quelque chose s'est mal passé. L'équipe a été notifiée et travaille sur une solution.
            </p>

            {process.env.NODE_ENV === 'development' && this.state.error && (
              <div className="mb-6 p-4 rounded-lg bg-red-50 dark:bg-red-900/10 border border-red-200 dark:border-red-800/20 text-left">
                <p className="text-xs font-mono text-red-800 dark:text-red-200 break-all">
                  {this.state.error.toString()}
                </p>
              </div>
            )}

            <Button
              onClick={this.handleReset}
              className="bg-gradient-to-r from-[#0E6B57] to-[#2FA36F] text-white hover:opacity-90 transition-opacity"
            >
              <RefreshCw size={18} className="mr-2" />
              Retour à l'accueil
            </Button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
