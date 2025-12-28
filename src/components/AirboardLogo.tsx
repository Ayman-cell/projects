import { Wind } from 'lucide-react';

interface AirboardLogoProps {
  size?: 'sm' | 'md' | 'lg';
  showText?: boolean;
}

export function AirboardLogo({ size = 'md', showText = true }: AirboardLogoProps) {
  const sizes = {
    sm: { icon: 20, text: 'text-lg' },
    md: { icon: 24, text: 'text-xl' },
    lg: { icon: 32, text: 'text-2xl' },
  };

  const currentSize = sizes[size];

  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <Wind
          className="text-primary"
          size={currentSize.icon}
          strokeWidth={2.5}
        />
      </div>
      {showText && (
        <span className={`${currentSize.text} tracking-tight`}>
          <span style={{ color: '#2FA36F' }}>A</span>
          <span style={{ color: '#0E6B57' }}>i</span>
          <span style={{ color: '#79D6A3' }}>r</span>
          <span style={{ color: '#2FA36F' }}>b</span>
          <span style={{ color: '#85D5FF' }}>o</span>
          <span style={{ color: '#0E6B57' }}>a</span>
          <span style={{ color: '#79D6A3' }}>r</span>
          <span style={{ color: '#2FA36F' }}>d</span>
        </span>
      )}
    </div>
  );
}