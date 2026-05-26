'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import { FaEye, FaBolt, FaChartLine, FaMicrochip, FaBrain } from 'react-icons/fa'

export default function Home() {
  return (
    <main className="h-screen flex flex-col overflow-hidden">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass border-b border-glass-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <motion.div 
              className="flex items-center space-x-2"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <FaEye className="text-primary text-2xl" />
              <span className="text-xl font-bold gradient-text">LenScope AI</span>
            </motion.div>
            
            <div className="flex items-center space-x-6">
              <Link href="/detect" className="text-gray-300 hover:text-primary transition-colors hidden md:block">
                Detection
              </Link>
              <Link href="/webcam" className="text-gray-300 hover:text-primary transition-colors hidden md:block">
                Webcam
              </Link>
              <Link href="/analytics" className="text-gray-300 hover:text-primary transition-colors hidden md:block">
                Analytics
              </Link>
              <Link href="/models" className="text-gray-300 hover:text-primary transition-colors hidden md:block">
                Models
              </Link>
              <Link href="/admin" className="text-gray-300 hover:text-primary transition-colors hidden lg:block">
                Admin
              </Link>
              <Link href="/ai-features" className="text-gray-300 hover:text-primary transition-colors hidden md:block">
                AI Features
              </Link>
              <Link 
                href="/detect" 
                className="btn-primary text-sm"
              >
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content - Compact Layout */}
      <div className="flex-1 flex items-center justify-center pt-16 pb-8 px-4">
        <div className="max-w-7xl mx-auto w-full">
          <div className="relative z-10 text-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <h1 className="text-5xl md:text-6xl font-bold mb-4">
                <span className="gradient-text">AI-Powered</span>
                <br />
                <span className="text-white">Object Detection</span>
              </h1>
              
              <p className="text-lg md:text-xl text-gray-400 mb-6 max-w-3xl mx-auto">
                Advanced computer vision dashboard with real-time object detection,
                multi-model comparison, and comprehensive analytics.
              </p>

              <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
                <Link href="/detect" className="btn-primary text-lg px-8 py-3">
                  Start Detecting
                  <FaBolt className="ml-2 inline" />
                </Link>
                <Link href="/webcam" className="btn-secondary text-lg px-8 py-3">
                  Live Webcam
                  <FaEye className="ml-2 inline" />
                </Link>
              </div>
            </motion.div>

            {/* Feature Cards - Compact Grid */}
            <motion.div 
              className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.8 }}
            >
              <FeatureCard
                icon={<FaBolt className="text-2xl text-primary" />}
                title="Real-Time Detection"
                description="Process images with lightning-fast inference"
              />
              <FeatureCard
                icon={<FaChartLine className="text-2xl text-secondary" />}
                title="Model Comparison"
                description="Compare YOLOv8 variants for optimal performance"
              />
              <FeatureCard
                icon={<FaMicrochip className="text-2xl text-accent-green" />}
                title="GPU Acceleration"
                description="CUDA acceleration with intelligent CPU fallback"
              />
            </motion.div>

            {/* Stats - Compact */}
            <motion.div 
              className="grid grid-cols-2 md:grid-cols-4 gap-3"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6, duration: 0.8 }}
            >
              <StatCard value="80+" label="Classes" />
              <StatCard value="<30ms" label="Inference" />
              <StatCard value="5+" label="Models" />
              <StatCard value="99%" label="Uptime" />
            </motion.div>
          </div>
        </div>
      </div>

      {/* Footer with Copyright */}
      <footer className="glass border-t border-glass-border py-2">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-gray-500 text-xs">
            © 2026 LenScope AI. Developed by Archisman Das. All rights reserved.
          </p>
        </div>
      </footer>
    </main>
  )
}

function FeatureCard({ icon, title, description }: { 
  icon: React.ReactNode
  title: string
  description: string
}) {
  return (
    <motion.div 
      className="card p-4"
      whileHover={{ scale: 1.03, y: -3 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <div className="mb-2">{icon}</div>
      <h3 className="text-lg font-semibold mb-1 text-white">{title}</h3>
      <p className="text-sm text-gray-400">{description}</p>
    </motion.div>
  )
}

function StatCard({ value, label }: { value: string; label: string }) {
  return (
    <div className="glass rounded-lg p-3 text-center">
      <div className="text-2xl font-bold neon-text mb-0">{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  )
}