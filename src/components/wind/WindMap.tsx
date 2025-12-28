import { useState, useRef, useEffect } from 'react';
import {
  calculateWindSpeed,
  calculateWindDirection,
  interpolateColor,
  GridPoint,
  GP2_STATION,
  GP1_STATION,
} from '../../lib/wind-utils';

interface WindMapProps {
  weatherData: GridPoint[];
  tempMin: number;
  tempMax: number;
  showTempBackground: boolean;
  tempBackgroundOpacity: number;
  showParticles: boolean;
  particleOpacity: number;
  particleDensity: string;
  showHeatmap: boolean;
  heatmapOpacity: number;
  showLabels: boolean;
  showCursorInfo: boolean;
  onCursorMove?: (data: {
    temp: number;
    windSpeed: number;
    windDirection: number;
  }) => void;
  onRecenterRequest?: (recenter: () => void) => void;
}

// Map utilities
function latLonToPixel(
  lat: number,
  lon: number,
  zoom: number
): { x: number; y: number } {
  const scale = 256 * Math.pow(2, zoom);
  const x = ((lon + 180) / 360) * scale;
  const y =
    ((1 -
      Math.log(
        Math.tan((lat * Math.PI) / 180) + 1 / Math.cos((lat * Math.PI) / 180)
      ) /
        Math.PI) /
      2) *
    scale;
  return { x, y };
}

function pixelToLatLon(
  x: number,
  y: number,
  zoom: number
): { lat: number; lon: number } {
  const scale = 256 * Math.pow(2, zoom);
  const lon = (x / scale) * 360 - 180;
  const n = Math.PI - (2 * Math.PI * y) / scale;
  const lat = (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
  return { lat, lon };
}

// Bilinear interpolation for smooth temperature gradient
function bilinearInterpolate(
  x: number,
  y: number,
  grid: GridPoint[],
  gridSize: number
): { temp: number; u: number; v: number } | null {
  // Find the grid square containing this point
  const minLat = Math.min(...grid.map((p) => p.lat));
  const maxLat = Math.max(...grid.map((p) => p.lat));
  const minLon = Math.min(...grid.map((p) => p.lon));
  const maxLon = Math.max(...grid.map((p) => p.lon));

  const latRange = maxLat - minLat;
  const lonRange = maxLon - minLon;

  const normalizedLat = (y - minLat) / latRange;
  const normalizedLon = (x - minLon) / lonRange;

  const gridX = normalizedLon * (gridSize - 1);
  const gridY = normalizedLat * (gridSize - 1);

  const x1 = Math.floor(gridX);
  const x2 = Math.min(x1 + 1, gridSize - 1);
  const y1 = Math.floor(gridY);
  const y2 = Math.min(y1 + 1, gridSize - 1);

  if (x1 < 0 || x2 >= gridSize || y1 < 0 || y2 >= gridSize) {
    return null;
  }

  // Get the four corner points
  const getGridPoint = (gx: number, gy: number) => {
    const index = gy * gridSize + gx;
    if (index < 0 || index >= grid.length) {
      return null;
    }
    return grid[index];
  };

  const p11 = getGridPoint(x1, y1);
  const p12 = getGridPoint(x1, y2);
  const p21 = getGridPoint(x2, y1);
  const p22 = getGridPoint(x2, y2);

  // Check if all grid points exist
  if (!p11 || !p12 || !p21 || !p22) {
    return null;
  }

  // Interpolation weights
  const wx = gridX - x1;
  const wy = gridY - y1;

  // Interpolate temperature
  const temp =
    p11.temp * (1 - wx) * (1 - wy) +
    p21.temp * wx * (1 - wy) +
    p12.temp * (1 - wx) * wy +
    p22.temp * wx * wy;

  // Interpolate wind components
  const u =
    p11.u * (1 - wx) * (1 - wy) +
    p21.u * wx * (1 - wy) +
    p12.u * (1 - wx) * wy +
    p22.u * wx * wy;

  const v =
    p11.v * (1 - wx) * (1 - wy) +
    p21.v * wx * (1 - wy) +
    p12.v * (1 - wx) * wy +
    p22.v * wx * wy;

  return { temp, u, v };
}

export function WindMap({
  weatherData,
  tempMin,
  tempMax,
  showTempBackground,
  tempBackgroundOpacity,
  showParticles,
  particleOpacity,
  particleDensity,
  showHeatmap,
  heatmapOpacity,
  showLabels,
  showCursorInfo,
  onCursorMove,
  onRecenterRequest,
}: WindMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const tilesCanvasRef = useRef<HTMLCanvasElement>(null);
  const tempBgCanvasRef = useRef<HTMLCanvasElement>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement>(null);
  const particlesCanvasRef = useRef<HTMLCanvasElement>(null);
  const labelsRef = useRef<HTMLDivElement>(null);

  const [zoom, setZoom] = useState(13);
  const [center, setCenter] = useState({
    lat: GP2_STATION.lat,
    lon: GP2_STATION.lon,
  });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  const particlesRef = useRef<
    Array<{ x: number; y: number; vx: number; vy: number; age: number; speed: number; temp: number }>
  >([]);
  const animationRef = useRef<number>();

  // Recenter function
  useEffect(() => {
    const recenter = () => {
      setCenter({ lat: GP2_STATION.lat, lon: GP2_STATION.lon });
      setZoom(13);
      setOffset({ x: 0, y: 0 });
    };
    onRecenterRequest?.(recenter);
  }, [onRecenterRequest]);

  // Draw tiles
  useEffect(() => {
    const canvas = tilesCanvasRef.current;
    if (!canvas || !containerRef.current) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const container = containerRef.current;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    const centerPixel = latLonToPixel(center.lat, center.lon, zoom);
    const tileSize = 256;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const startTileX = Math.floor(
      (centerPixel.x - canvas.width / 2 - offset.x) / tileSize
    );
    const startTileY = Math.floor(
      (centerPixel.y - canvas.height / 2 - offset.y) / tileSize
    );
    const endTileX = Math.ceil(
      (centerPixel.x + canvas.width / 2 - offset.x) / tileSize
    );
    const endTileY = Math.ceil(
      (centerPixel.y + canvas.height / 2 - offset.y) / tileSize
    );

    for (let tileX = startTileX; tileX <= endTileX; tileX++) {
      for (let tileY = startTileY; tileY <= endTileY; tileY++) {
        const maxTile = Math.pow(2, zoom);
        if (tileX < 0 || tileX >= maxTile || tileY < 0 || tileY >= maxTile)
          continue;

        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = `https://tile.openstreetmap.org/${zoom}/${tileX}/${tileY}.png`;

        const x =
          tileX * tileSize -
          centerPixel.x +
          canvas.width / 2 +
          offset.x;
        const y =
          tileY * tileSize -
          centerPixel.y +
          canvas.height / 2 +
          offset.y;

        img.onload = () => {
          ctx.drawImage(img, x, y, tileSize, tileSize);
        };
      }
    }
  }, [center, zoom, offset]);

  // Draw temperature background (continuous gradient)
  useEffect(() => {
    if (!showTempBackground) return;

    const canvas = tempBgCanvasRef.current;
    if (!canvas || !containerRef.current) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const container = containerRef.current;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const centerPixel = latLonToPixel(center.lat, center.lon, zoom);
    const gridSize = 21; // Same as weather data grid

    // Create a dense pixel grid for smooth gradient
    const pixelStep = 8; // Draw every 8 pixels for performance
    
    for (let py = 0; py < canvas.height; py += pixelStep) {
      for (let px = 0; px < canvas.width; px += pixelStep) {
        // Convert pixel to lat/lon
        const worldX = px - canvas.width / 2 - offset.x + centerPixel.x;
        const worldY = py - canvas.height / 2 - offset.y + centerPixel.y;
        const latLon = pixelToLatLon(worldX, worldY, zoom);

        // Interpolate temperature at this point
        const interpolated = bilinearInterpolate(
          latLon.lon,
          latLon.lat,
          weatherData,
          gridSize
        );

        if (interpolated) {
          const color = interpolateColor(interpolated.temp, tempMin, tempMax);
          ctx.fillStyle = color;
          ctx.globalAlpha = tempBackgroundOpacity / 100;
          ctx.fillRect(px, py, pixelStep, pixelStep);
        }
      }
    }
  }, [
    showTempBackground,
    tempBackgroundOpacity,
    weatherData,
    tempMin,
    tempMax,
    center,
    zoom,
    offset,
  ]);

  // Draw heatmap (optional contrast boost)
  useEffect(() => {
    if (!showHeatmap) return;

    const canvas = heatmapCanvasRef.current;
    if (!canvas || !containerRef.current) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const container = containerRef.current;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const centerPixel = latLonToPixel(center.lat, center.lon, zoom);

    weatherData.forEach((point) => {
      const pixel = latLonToPixel(point.lat, point.lon, zoom);
      const x = pixel.x - centerPixel.x + canvas.width / 2 + offset.x;
      const y = pixel.y - centerPixel.y + canvas.height / 2 + offset.y;

      const color = interpolateColor(point.temp, tempMin, tempMax);
      const rgb = color.match(/\d+/g);
      if (!rgb) return;

      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 40);
      gradient.addColorStop(
        0,
        `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${heatmapOpacity / 100})`
      );
      gradient.addColorStop(
        1,
        `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0)`
      );

      ctx.fillStyle = gradient;
      ctx.fillRect(x - 40, y - 40, 80, 80);
    });
  }, [
    showHeatmap,
    weatherData,
    tempMin,
    tempMax,
    heatmapOpacity,
    center,
    zoom,
    offset,
  ]);

  // Animate particles as lines
  useEffect(() => {
    if (!showParticles) return;

    const canvas = particlesCanvasRef.current;
    if (!canvas || !containerRef.current) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const container = containerRef.current;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    // Particle count based on density
    const densityMap = {
      low: 1000,
      medium: 2000,
      high: 3500,
    };
    const numParticles = densityMap[particleDensity as keyof typeof densityMap] || 2000;

    if (particlesRef.current.length === 0 || particlesRef.current.length !== numParticles) {
      particlesRef.current = Array.from({ length: numParticles }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: 0,
        vy: 0,
        age: Math.random() * 100,
        speed: 0,
        temp: 0,
      }));
    }

    const centerPixel = latLonToPixel(center.lat, center.lon, zoom);
    const gridSize = 21;

    const getWindAtPosition = (x: number, y: number) => {
      const worldX = x - canvas.width / 2 - offset.x + centerPixel.x;
      const worldY = y - canvas.height / 2 - offset.y + centerPixel.y;
      const latLon = pixelToLatLon(worldX, worldY, zoom);

      const interpolated = bilinearInterpolate(
        latLon.lon,
        latLon.lat,
        weatherData,
        gridSize
      );

      if (interpolated) {
        return { u: interpolated.u, v: interpolated.v, temp: interpolated.temp };
      }

      return { u: 0, v: 0, temp: tempMin };
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particlesRef.current.forEach((particle) => {
        const wind = getWindAtPosition(particle.x, particle.y);

        particle.vx = wind.u * 0.8;
        particle.vy = -wind.v * 0.8;
        particle.speed = Math.sqrt(particle.vx * particle.vx + particle.vy * particle.vy);
        particle.temp = wind.temp;

        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.age++;

        if (
          particle.x < 0 ||
          particle.x > canvas.width ||
          particle.y < 0 ||
          particle.y > canvas.height ||
          particle.age > 100
        ) {
          particle.x = Math.random() * canvas.width;
          particle.y = Math.random() * canvas.height;
          particle.age = 0;
        }

        // Draw as line (trail)
        const alpha = Math.max(0, 1 - particle.age / 100) * (particleOpacity / 100);
        
        // Line length proportional to speed (plus longues et plus fluides: 15-25px range)
        const lineLength = Math.min(Math.max(particle.speed * 3, 8), 25);
        
        // Calculate line direction
        const angle = Math.atan2(particle.vy, particle.vx);
        const endX = particle.x + Math.cos(angle) * lineLength;
        const endY = particle.y + Math.sin(angle) * lineLength;

        // Line thickness (1.5-3px, thicker for high speed)
        const lineWidth = particle.speed > 10 ? 3 : 2;

        // Get color based on temperature
        const colorRgb = interpolateColor(particle.temp, tempMin, tempMax);
        const rgb = colorRgb.match(/\d+/g);
        
        if (rgb) {
          ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
        } else {
          ctx.strokeStyle = `rgba(31, 41, 55, ${alpha})`;
        }
        
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        
        ctx.beginPath();
        ctx.moveTo(particle.x, particle.y);
        ctx.lineTo(endX, endY);
        ctx.stroke();
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [
    showParticles,
    particleOpacity,
    particleDensity,
    weatherData,
    center,
    zoom,
    offset,
    tempMin,
    tempMax,
  ]);

  // Mouse handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }

    if (showCursorInfo && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const centerPixel = latLonToPixel(center.lat, center.lon, zoom);
      const worldX = x - rect.width / 2 - offset.x + centerPixel.x;
      const worldY = y - rect.height / 2 - offset.y + centerPixel.y;
      const latLon = pixelToLatLon(worldX, worldY, zoom);

      const gridSize = 21;
      const interpolated = bilinearInterpolate(
        latLon.lon,
        latLon.lat,
        weatherData,
        gridSize
      );

      if (interpolated) {
        const windSpeed = calculateWindSpeed(interpolated.u, interpolated.v);
        const windDirection = calculateWindDirection(interpolated.u, interpolated.v);

        onCursorMove?.({
          temp: interpolated.temp,
          windSpeed,
          windDirection,
        });
      }
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -1 : 1;
    // Limite de zoom: min 11 (frontières de Safi), max 15 (très proche)
    setZoom((prev) => Math.max(11, Math.min(15, prev + delta)));
  };

  // Draw GP2 and GP1 markers and labels
  useEffect(() => {
    const canvas = particlesCanvasRef.current;
    const labelContainer = labelsRef.current;
    if (!canvas || !labelContainer || !containerRef.current) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const centerPixel = latLonToPixel(center.lat, center.lon, zoom);
    
    // Draw GP2 Station
    const gp2Pixel = latLonToPixel(GP2_STATION.lat, GP2_STATION.lon, zoom);
    const x2 = gp2Pixel.x - centerPixel.x + canvas.width / 2 + offset.x;
    const y2 = gp2Pixel.y - centerPixel.y + canvas.height / 2 + offset.y;

    // Draw GP2 halo
    ctx.beginPath();
    ctx.arc(x2, y2, 16, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(89, 134, 75, 0.3)';
    ctx.fill();

    // Draw GP2 outer circle (white border)
    ctx.beginPath();
    ctx.arc(x2, y2, 12, 0, Math.PI * 2);
    ctx.fillStyle = 'white';
    ctx.fill();

    // Draw GP2 inner circle (green)
    ctx.beginPath();
    ctx.arc(x2, y2, 9, 0, Math.PI * 2);
    ctx.fillStyle = '#59864B';
    ctx.fill();

    // Draw GP2 station label
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = 'white';
    ctx.strokeStyle = '#59864B';
    ctx.lineWidth = 3;
    ctx.textAlign = 'center';
    ctx.strokeText('GP2', x2, y2 + 30);
    ctx.fillText('GP2', x2, y2 + 30);
    
    // Draw GP1 Station (with red/orange color to differentiate)
    const gp1Pixel = latLonToPixel(GP1_STATION.lat, GP1_STATION.lon, zoom);
    const x1 = gp1Pixel.x - centerPixel.x + canvas.width / 2 + offset.x;
    const y1 = gp1Pixel.y - centerPixel.y + canvas.height / 2 + offset.y;

    // Draw GP1 halo
    ctx.beginPath();
    ctx.arc(x1, y1, 16, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';
    ctx.fill();

    // Draw GP1 outer circle (white border)
    ctx.beginPath();
    ctx.arc(x1, y1, 12, 0, Math.PI * 2);
    ctx.fillStyle = 'white';
    ctx.fill();

    // Draw GP1 inner circle (red/orange)
    ctx.beginPath();
    ctx.arc(x1, y1, 9, 0, Math.PI * 2);
    ctx.fillStyle = '#EF4444';
    ctx.fill();

    // Draw GP1 station label
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = 'white';
    ctx.strokeStyle = '#EF4444';
    ctx.lineWidth = 3;
    ctx.textAlign = 'center';
    ctx.strokeText('GP1', x1, y1 - 18);
    ctx.fillText('GP1', x1, y1 - 18);

    // Temperature labels
    if (showLabels) {
      labelContainer.innerHTML = '';
      const step = 3;

      for (let i = 0; i < weatherData.length; i += step) {
        const point = weatherData[i];
        const pixel = latLonToPixel(point.lat, point.lon, zoom);
        const px = pixel.x - centerPixel.x + canvas.width / 2 + offset.x;
        const py = pixel.y - centerPixel.y + canvas.height / 2 + offset.y;

        const label = document.createElement('div');
        label.style.position = 'absolute';
        label.style.left = `${px}px`;
        label.style.top = `${py - 20}px`;
        label.style.transform = 'translateX(-50%)';
        label.style.fontSize = '11px';
        label.style.fontFamily = 'monospace';
        label.style.color = '#0f172a';
        label.style.backgroundColor = 'rgba(255,255,255,0.9)';
        label.style.padding = '2px 4px';
        label.style.borderRadius = '4px';
        label.style.whiteSpace = 'nowrap';
        label.style.pointerEvents = 'none';
        label.textContent = `${point.temp.toFixed(1)}°`;
        labelContainer.appendChild(label);
      }
    } else {
      labelContainer.innerHTML = '';
    }
  }, [center, zoom, offset, showLabels, weatherData]);

  return (
    <div
      ref={containerRef}
      className="flex-1 relative overflow-hidden bg-muted cursor-move"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
    >
      <canvas
        ref={tilesCanvasRef}
        className="absolute inset-0"
        style={{ zIndex: 1 }}
      />
      <canvas
        ref={tempBgCanvasRef}
        className="absolute inset-0 pointer-events-none"
        style={{ 
          zIndex: 2, 
          display: showTempBackground ? 'block' : 'none',
          mixBlendMode: 'multiply'
        }}
      />
      <canvas
        ref={heatmapCanvasRef}
        className="absolute inset-0 pointer-events-none"
        style={{ 
          zIndex: 3, 
          display: showHeatmap ? 'block' : 'none',
          mixBlendMode: 'multiply'
        }}
      />
      <canvas
        ref={particlesCanvasRef}
        className="absolute inset-0 pointer-events-none"
        style={{ zIndex: 4 }}
      />
      <div
        ref={labelsRef}
        className="absolute inset-0 pointer-events-none"
        style={{ zIndex: 5 }}
      />

      {/* Attribution */}
      <div className="absolute bottom-2 right-2 text-xs text-muted-foreground bg-card/90 px-2 py-1 rounded z-10">
        © OpenStreetMap
      </div>

      {/* Zoom Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 z-10">
        <button
          onClick={() => setZoom((prev) => Math.min(15, prev + 1))}
          className="w-10 h-10 rounded-lg bg-card/90 backdrop-blur-sm hover:bg-card flex items-center justify-center transition-all border border-border shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={zoom >= 15}
          aria-label="Zoomer"
        >
          <span className="text-xl">+</span>
        </button>
        <button
          onClick={() => setZoom((prev) => Math.max(11, prev - 1))}
          className="w-10 h-10 rounded-lg bg-card/90 backdrop-blur-sm hover:bg-card flex items-center justify-center transition-all border border-border shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={zoom <= 11}
          aria-label="Dézoomer"
        >
          <span className="text-xl">−</span>
        </button>
      </div>
    </div>
  );
}