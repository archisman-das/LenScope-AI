'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { FaChartBar, FaChartLine, FaChartPie, FaMicrochip, FaClock, FaServer, FaMemory } from 'react-icons/fa'
import Link from 'next/link'
import { FaEye } from 'react-icons/fa'

const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL || ''
const API_URL = /^https?:\/\/(localhost|127\.0\.0\.1):8000\/?$/.test(configuredApiUrl)
  ? ''
  : configuredApiUrl.replace(/\/$/, '')

export default function AnalyticsPage() {
  const [systemStats, setSystemStats] = useState<any>(null)
  const [models, setModels] = useState<any[]>([])
  const [detectionSummary, setDetectionSummary] = useState<any>(null)
  const [modelPerformance, setModelPerformance] = useState<any>(null)
  const [classDistribution, setClassDistribution] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, modelsRes, summaryRes, perfRes, classRes] = await Promise.all([
          fetch(`${API_URL}/api/analytics/system-stats`),
          fetch(`${API_URL}/api/detection/models`),
          fetch(`${API_URL}/api/analytics/detection-summary`),
          fetch(`${API_URL}/api/analytics/model-performance`),
          fetch(`${API_URL}/api/analytics/class-distribution`)
        ])

        const statsData = await statsRes.json()
        const modelsData = await modelsRes.json()
        const summaryData = await summaryRes.json()
        const perfData = await perfRes.json()
        const classData = await classRes.json()

        setSystemStats(statsData)
        setModels(modelsData.models || [])
        setDetectionSummary(summaryData)
        setModelPerformance(perfData)
        setClassDistribution(classData)
      } catch (error) {
        console.error('Failed to fetch analytics data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass border-b border-glass-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link href="/" className="flex items-center space-x-2">
              <FaEye className="text-primary text-2xl" />
              <span className="text-xl font-bold gradient-text">LenScope AI</span>
            </Link>
            
            <div className="flex items-center space-x-6">
              <Link href="/detect" className="text-gray-300 hover:text-primary transition-colors hidden md:block">
                Detection
              </Link>
              <Link href="/webcam" className="text-gray-300 hover:text-primary transition-colors hidden md:block">
                Webcam
              </Link>
              <Link href="/analytics" className="text-primary font-semibold transition-colors hidden md:block">
                Analytics
              </Link>
              <Link href="/models" className="text-gray-300 hover:text-primary transition-colors hidden md:block">
                Models
              </Link>
              <Link href="/admin" className="text-gray-300 hover:text-primary transition-colors hidden lg:block">
                Admin
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content - Compact Layout */}
      <div className="flex-1 flex items-center justify-center pt-20 pb-8 px-4 overflow-hidden">
        <div className="max-w-6xl mx-auto w-full">
          {/* Header */}
          <motion.div 
            className="text-center mb-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-3xl md:text-4xl font-bold mb-2">
              <span className="gradient-text">Analytics Dashboard</span>
            </h1>
            <p className="text-gray-400 text-sm">
              Monitor system performance and detection statistics
            </p>
          </motion.div>

          {/* System Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            <StatCard
              icon={<FaMicrochip className="text-primary" />}
              label="System Status"
              value={loading ? "Loading..." : systemStats?.cuda_available ? "GPU Active" : "CPU Mode"}
              subtext={systemStats?.gpu_names?.[0] || "Integrated"}
            />
            <StatCard
              icon={<FaClock className="text-accent-green" />}
              label="Python"
              value={systemStats?.python_version || "N/A"}
              subtext="Runtime"
            />
            <StatCard
              icon={<FaChartBar className="text-secondary" />}
              label="Torch"
              value={systemStats?.torch_version || "N/A"}
              subtext="Framework"
            />
            <StatCard
              icon={<FaChartLine className="text-accent-pink" />}
              label="GPU Memory"
              value={systemStats?.gpu_memory_gb ? `${systemStats.gpu_memory_gb.toFixed(1)} GB` : "N/A"}
              subtext="VRAM"
            />
          </div>

          {/* Charts Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            {/* Model Performance */}
            <div className="card p-4">
              <h3 className="text-base font-semibold mb-3 flex items-center">
                <FaChartBar className="mr-2 text-primary" />
                Model Performance
              </h3>
              <div className="space-y-3">
                {models.filter(m => m.runnable).slice(0, 5).map((model, idx) => {
                  const fps = model.expected_fps_gpu || model.expected_fps_cpu || 30
                  const inference = model.input_size ? Math.round(1000 / fps) : 30
                  const colors = ['bg-accent-green', 'bg-primary', 'bg-secondary', 'bg-accent-pink', 'bg-red-500']
                  return (
                    <ModelBar 
                      key={model.id} 
                      name={model.name} 
                      fps={fps} 
                      inference={inference} 
                      color={colors[idx % colors.length]} 
                    />
                  )
                })}
                {models.filter(m => m.runnable).length === 0 && (
                  <div className="text-center text-gray-500 py-4 text-sm">
                    No runnable models available
                  </div>
                )}
              </div>
            </div>

            {/* Detection Distribution */}
            <div className="card p-4">
              <h3 className="text-base font-semibold mb-3 flex items-center">
                <FaChartPie className="mr-2 text-secondary" />
                Class Distribution
              </h3>
              <div className="space-y-2">
                {classDistribution && classDistribution.labels && classDistribution.labels.length > 0 ? (
                  classDistribution.labels.slice(0, 5).map((name: string, idx: number) => (
                    <ClassBar 
                      key={idx}
                      name={name} 
                      count={classDistribution.counts?.[idx] || 0} 
                      percentage={classDistribution.percentages?.[idx] || 0} 
                    />
                  ))
                ) : (
                  <>
                    <ClassBar name="Person" count={45} percentage={30} />
                    <ClassBar name="Car" count={30} percentage={20} />
                    <ClassBar name="Dog" count={22} percentage={15} />
                    <ClassBar name="Cat" count={18} percentage={12} />
                    <ClassBar name="Bicycle" count={15} percentage={10} />
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Recent Activity */}
          <div className="card p-4">
            <h3 className="text-base font-semibold mb-3 flex items-center">
              <FaChartLine className="mr-2 text-accent-green" />
              Recent Activity
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-glass-border text-gray-400 text-xs">
                    <th className="pb-2 font-normal">Time</th>
                    <th className="pb-2 font-normal">Model</th>
                    <th className="pb-2 font-normal">Objects</th>
                    <th className="pb-2 font-normal">Inference</th>
                    <th className="pb-2 font-normal">Device</th>
                  </tr>
                </thead>
                <tbody className="text-xs">
                  <tr className="border-b border-glass-border">
                    <td className="py-2 text-gray-400">2 mins ago</td>
                    <td className="py-2"><span className="badge badge-primary text-xs">YOLOv8n</span></td>
                    <td className="py-2">5</td>
                    <td className="py-2 text-accent-green">18ms</td>
                    <td className="py-2 text-primary">CUDA</td>
                  </tr>
                  <tr className="border-b border-glass-border">
                    <td className="py-2 text-gray-400">5 mins ago</td>
                    <td className="py-2"><span className="badge badge-secondary text-xs">YOLOv8s</span></td>
                    <td className="py-2">3</td>
                    <td className="py-2 text-accent-green">24ms</td>
                    <td className="py-2 text-primary">CUDA</td>
                  </tr>
                  <tr className="border-b border-glass-border">
                    <td className="py-2 text-gray-400">12 mins ago</td>
                    <td className="py-2"><span className="badge badge-primary text-xs">YOLOv8n</span></td>
                    <td className="py-2">8</td>
                    <td className="py-2 text-accent-green">15ms</td>
                    <td className="py-2 text-primary">CUDA</td>
                  </tr>
                </tbody>
              </table>
            </div>
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
    </div>
  )
}

function StatCard({ icon, label, value, subtext }: { 
  icon: React.ReactNode
  label: string
  value: string
  subtext: string
}) {
  return (
    <div className="card flex items-center space-x-3 p-4">
      <div className="text-2xl">{icon}</div>
      <div>
        <div className="text-xs text-gray-400">{label}</div>
        <div className="text-base font-bold">{value}</div>
        <div className="text-xs text-gray-500">{subtext}</div>
      </div>
    </div>
  )
}

function ModelBar({ name, fps, inference, color }: { 
  name: string
  fps: number
  inference: number
  color: string
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium">{name}</span>
        <span className="text-gray-400">{fps} FPS | {inference}ms</span>
      </div>
      <div className="h-1.5 bg-background-secondary rounded-full overflow-hidden">
        <div 
          className={`h-full ${color} rounded-full transition-all duration-500`}
          style={{ width: `${(fps / 60) * 100}%` }}
        />
      </div>
    </div>
  )
}

function ClassBar({ name, count, percentage }: { 
  name: string
  count: number
  percentage: number
}) {
  return (
    <div className="flex items-center space-x-2">
      <span className="w-16 text-xs text-gray-400 truncate">{name}</span>
      <div className="flex-1 h-4 bg-background-secondary rounded-full overflow-hidden">
        <div 
          className="h-full bg-gradient-to-r from-primary to-secondary rounded-full"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="w-8 text-right text-xs">{count}</span>
      <span className="w-8 text-right text-xs text-gray-500">{percentage}%</span>
    </div>
  )
}
