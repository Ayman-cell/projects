"use client";

import type React from "react";
import { useEffect, useRef, useState } from "react";
import { MeshGradient } from "@paper-design/shaders-react";
import { useTheme } from "./ThemeContext";
import { motion, useScroll, useTransform } from "motion/react";

interface ShaderBackgroundProps {
  children?: React.ReactNode;
}

export default function ShaderBackground({
  children,
}: ShaderBackgroundProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [prefersReducedMotion, setPrefersReducedMotion] =
    useState(false);
  const { theme } = useTheme();

  const { scrollYProgress } = useScroll();
  const backgroundY = useTransform(
    scrollYProgress,
    [0, 1],
    ["0%", "20%"],
  );

  useEffect(() => {
    const mediaQuery = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    );
    setPrefersReducedMotion(mediaQuery.matches);

    const handleChange = (e: MediaQueryListEvent) => {
      setPrefersReducedMotion(e.matches);
    };

    mediaQuery.addEventListener("change", handleChange);
    return () =>
      mediaQuery.removeEventListener("change", handleChange);
  }, []);

  // Light mode - vibrant gradient colors
  const lightColors = [
    "#CFF5E4",
    "#79D6A3",
    "#85D5FF",
    "#A8E6CF",
    "#2FA36F",
  ];
  const lightColors2 = [
    "#79D6A3",
    "#85D5FF",
    "#CFF5E4",
    "#6BCDDA",
  ];
  const lightColors3 = ["#85D5FF", "#A8E6CF", "#79D6A3"];

  // Dark mode - Plus foncé
 const darkColors = ["#020403", "#050807", "#07110E", "#0A1E1A", "#000000"]
const darkColors2 = ["#0", "#0", "#0", "#1FA48A"]
const darkColors3 = ["#07110E", "#146A56", "#34E0FF"]


  
  const colors = theme === "light" ? lightColors : darkColors;
  const colors2 =
    theme === "light" ? lightColors2 : darkColors2;
  const colors3 =
    theme === "light" ? lightColors3 : darkColors3;
  const bgColor = theme === "light" ? "#CFF5E4" : "#0B0F0C";

  return (
    <div
      ref={containerRef}
      className="min-h-screen relative"
      style={{
        backgroundColor: bgColor,
        background:
          theme === "light"
            ? "linear-gradient(180deg, #CFF5E4 0%, #79D6A3 50%, #85D5FF 100%)"
            : "linear-gradient(180deg, #0B0F0C 0%, #0E1411 50%, #0A1912 100%)",
      }}
    >
      {/* Background shader layer - fixed behind everything */}
      <div className="fixed inset-0 z-0 overflow-hidden">
        {!prefersReducedMotion && (
          <>
            {/* SVG Filters for turbulence */}
            <svg className="absolute inset-0 w-0 h-0 pointer-events-none">
              <defs>
                <filter
                  id="fluid-turbulence"
                  x="-50%"
                  y="-50%"
                  width="200%"
                  height="200%"
                >
                  <feTurbulence
                    type="fractalNoise"
                    baseFrequency={
                      theme === "light"
                        ? "0.004 0.007"
                        : "0.003 0.006"
                    }
                    numOctaves="4"
                    result="turbulence"
                  >
                    <animate
                      attributeName="baseFrequency"
                      dur="60s"
                      values={
                        theme === "light"
                          ? "0.004 0.007;0.006 0.009;0.004 0.007"
                          : "0.003 0.006;0.005 0.008;0.003 0.006"
                      }
                      repeatCount="indefinite"
                    />
                  </feTurbulence>
                  <feDisplacementMap
                    in="SourceGraphic"
                    in2="turbulence"
                    scale="40"
                  />
                  <feGaussianBlur stdDeviation="1" />
                </filter>
              </defs>
            </svg>

            {/* Main gradient layer - always visible at 100% opacity */}
            <motion.div
              className="absolute inset-0 w-full h-full"
              style={{ y: backgroundY }}
            >
              <MeshGradient
                className="absolute inset-0 w-full h-full"
                colors={colors}
                speed={0.8}
                backgroundcolor={bgColor}
              />
            </motion.div>

            {/* Secondary layer for depth */}
            <MeshGradient
              className="absolute inset-0 w-full h-full opacity-60"
              colors={colors2}
              speed={0.6}
              backgroundcolor="transparent"
            />

            {/* Tertiary accent layer */}
            <MeshGradient
              className="absolute inset-0 w-full h-full opacity-30"
              colors={colors3}
              speed={0.4}
              wireframe="true"
              backgroundcolor="transparent"
            />

            {/* Overlay gradient for depth */}
            <div
              className="absolute inset-0 pointer-events-none opacity-40"
              style={{
                background:
                  theme === "light"
                    ? "linear-gradient(180deg, rgba(255, 255, 255, 0.3) 0%, rgba(0, 0, 0, 0.05) 100%)"
                    : "linear-gradient(180deg, rgba(255, 255, 255, 0.05) 0%, rgba(0, 0, 0, 0.3) 100%)",
              }}
            />
          </>
        )}
      </div>

      {/* Content layer - on top with transparent background */}
      <div
        className="relative z-10"
        style={{ background: "transparent" }}
      >
        {children}
      </div>
    </div>
  );
}