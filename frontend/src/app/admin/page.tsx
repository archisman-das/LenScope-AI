'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  FaUsers, 
  FaDatabase, 
  FaFileAlt, 
  FaCogs, 
  FaChartBar,
  FaClock,
  FaTrash,
  FaDownload,
  FaSearch,
  FaServer,
  FaMemory,
  FaMicrochip
} from 'react-icons/fa'
import Link from 'next/link'
import { FaEye } from 'react-icons/fa'

const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL || ''
const API_URL = /^https?:\/\/(localhost|127\.0\.0\.1):8000\/?$/.test(configuredApiUrl)
  ? ''
  : configuredApiUrl.replace(/\/$/, '')

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState('overview')
  const [systemStats, setSystemStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    const fetchSystemStats = async () => {
      try {
        const response = await fetch(`${API_URL}/api/detection/system-info`)
        const data = await response.json()
        setSystemStats(data)
      } catch (error) {
        console.error('Failed to fetch system stats:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchSystemStats()
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview', icon: <FaChartBar /> },
    { id: 'detections', label: 'History', icon: <FaFileAlt /> },
    { id: 'datasets', label: 'Datasets', icon: <FaDatabase /> },
    { id: 'logs', label: 'Logs', icon: <FaClock /> },
    { id: 'users', label: 'Users', icon: <FaUsers /> },
    { id: 'settings', label: 'Settings', icon: <FaCogs /> },
  ]

  const recentDetections = [
    { id: 1, model: 'YOLOv8n', objects: 5, time: '2m ago', user: 'admin', confidence: 92 },
    { id: 2, model: 'YOLOv8s', objects: 3, time: '5m ago', user: 'user1', confidence: 87 },
    { id: 3, model: 'YOLOv8n', objects: 8, time: '12m ago', user: 'admin', confidence: 95 },
    { id: 4, model: 'YOLOv8m', objects: 2, time: '18m ago', user: 'user2', confidence: 78 },
  ]

  const systemLogs = [
    { id: 1, level: 'info', message: 'Model yolov8n loaded', time: '10:30' },
    { id: 2, level: 'info', message: 'DB connection OK', time: '10:30' },
    { id: 3, level: 'warning', message: 'High memory (85%)', time: '10:28' },
    { id: 4, level: 'info', message: 'Admin logged in', time: '10:25' },
  ]

  const datasets = [
    { id: 1, name: 'COCO', images: 118287, classes: 80, size: '22GB', status: 'active' },
    { id: 2, name: 'Custom', images: 1250, classes: 12, size: '450MB', status: 'active' },
    { id: 3, name: 'Test', images: 500, classes: 5, size: '180MB', status: 'archived' },
  ]

  const users = [
    { id: 1, name: 'Admin', email: 'admin@lenscope.ai', role: 'Admin', detections: 1250, lastActive: 'Now' },
    { id: 2, name: 'User 1', email: 'user1@lenscope.ai', role: 'User', detections: 340, lastActive: '5m ago' },
    { id: 3, name: 'User 2', email: 'user2@lenscope.ai', role: 'User', detections: 89, lastActive: '1h ago' },
  ]

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
              <Link href="/analytics" className="text-gray-300 hover:text-primary transition-colors hidden md:block">
                Analytics
              </Link>
              <Link href="/models" className="text-gray-300 hover:text-primary transition-colors hidden md:block">
                Models
              </Link>
              <Link href="/admin" className="text-primary font-semibold transition-colors hidden lg:block">
                Admin
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content - Compact Layout */}
      <div className="flex-1 flex flex-col pt-20 pb-8 px-4 overflow-hidden">
        <div className="max-w-6xl mx-auto w-full flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <motion.div 
            className="text-center mb-3"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-2xl md:text-3xl font-bold mb-1">
              <span className="gradient-text">Admin Dashboard</span>
            </h1>
            <p className="text-gray-400 text-xs">
              System management and monitoring
            </p>
          </motion.div>

          {/* System Health Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 mb-3">
            <StatCard
              icon={<FaMicrochip className="text-primary" />}
              label="CPU"
              value={systemStats?.memory_usage?.cpu_percent ? `${systemStats.memory_usage.cpu_percent}%` : 'N/A'}
              subtext="Usage"
            />
            <StatCard
              icon={<FaMemory className="text-secondary" />}
              label="RAM"
              value={systemStats?.memory_usage?.ram_percent ? `${systemStats.memory_usage.ram_percent.toFixed(1)}%` : 'N/A'}
              subtext="Usage"
            />
            <StatCard
              icon={<FaServer className="text-accent-green" />}
              label="Models"
              value={systemStats?.loaded_models?.length || 0}
              subtext="Loaded"
            />
            <StatCard
              icon={<FaDatabase className="text-accent-pink" />}
              label="Detections"
              value="1,691"
              subtext="Total"
            />
          </div>

          <div className="flex-1 flex gap-3 overflow-hidden">
            {/* Sidebar Navigation */}
            <div className="w-32 flex-shrink-0">
              <div className="card p-2 h-full">
                <nav className="space-y-1">
                  {tabs.map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`w-full flex items-center space-x-2 px-3 py-2 rounded-lg transition-all text-xs ${
                        activeTab === tab.id
                          ? 'bg-primary/20 text-primary border border-primary/30'
                          : 'text-gray-400 hover:text-white hover:bg-white/5'
                      }`}
                    >
                      <span className="text-sm">{tab.icon}</span>
                      <span className="font-medium hidden xl:block">{tab.label}</span>
                    </button>
                  ))}
                </nav>
              </div>
            </div>

            {/* Main Content Area */}
            <div className="flex-1 overflow-hidden">
              <div className="card h-full p-4 overflow-auto">
                {/* Overview Tab */}
                {activeTab === 'overview' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <h2 className="text-lg font-bold mb-4">System Overview</h2>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                      <QuickStat label="Uptime" value="24h" />
                      <QuickStat label="Req/min" value="42" />
                      <QuickStat label="Avg Resp" value="156ms" />
                      <QuickStat label="Errors" value="0.02%" />
                    </div>

                    <h3 className="text-sm font-semibold mb-2">Recent Detections</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr className="border-b border-glass-border text-gray-400">
                            <th className="pb-2 font-normal">Model</th>
                            <th className="pb-2 font-normal">Objects</th>
                            <th className="pb-2 font-normal">Confidence</th>
                            <th className="pb-2 font-normal">User</th>
                            <th className="pb-2 font-normal">Time</th>
                          </tr>
                        </thead>
                        <tbody>
                          {recentDetections.map((det) => (
                            <tr key={det.id} className="border-b border-glass-border">
                              <td className="py-2">
                                <span className="badge badge-primary text-xs">{det.model}</span>
                              </td>
                              <td className="py-2">{det.objects}</td>
                              <td className="py-2 text-accent-green">{det.confidence}%</td>
                              <td className="py-2 text-gray-400">{det.user}</td>
                              <td className="py-2 text-gray-500">{det.time}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </motion.div>
                )}

                {/* Detection History Tab */}
                {activeTab === 'detections' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-lg font-bold">Detection History</h2>
                      <button className="btn-secondary text-xs flex items-center">
                        <FaDownload className="mr-1" />
                        Export
                      </button>
                    </div>

                    <div className="relative mb-4">
                      <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500 text-xs" />
                      <input
                        type="text"
                        placeholder="Search..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="input pl-10 text-sm"
                      />
                    </div>

                    <div className="text-center text-gray-500 py-8 text-sm">
                      <FaFileAlt className="text-2xl mx-auto mb-2 opacity-50" />
                      <p>History will be displayed here</p>
                    </div>
                  </motion.div>
                )}

                {/* Datasets Tab */}
                {activeTab === 'datasets' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-lg font-bold">Datasets</h2>
                      <button className="btn-primary text-xs">+ Upload</button>
                    </div>

                    <div className="space-y-2">
                      {datasets.map((dataset) => (
                        <div key={dataset.id} className="glass rounded-lg p-3 flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <FaDatabase className="text-primary text-xl" />
                            <div>
                              <h3 className="text-sm font-semibold">{dataset.name}</h3>
                              <p className="text-xs text-gray-400">
                                {dataset.images.toLocaleString()} images • {dataset.classes} classes
                              </p>
                            </div>
                          </div>
                          <span className={`badge text-xs ${dataset.status === 'active' ? 'badge-primary' : 'badge-secondary'}`}>
                            {dataset.status}
                          </span>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}

                {/* Logs Tab */}
                {activeTab === 'logs' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-lg font-bold">System Logs</h2>
                      <button className="btn-secondary text-xs flex items-center">
                        <FaDownload className="mr-1" />
                        Export
                      </button>
                    </div>

                    <div className="space-y-1">
                      {systemLogs.map((log) => (
                        <div key={log.id} className="glass rounded-lg p-2 flex items-center space-x-3 text-xs">
                          <span className={`w-2 h-2 rounded-full ${
                            log.level === 'info' ? 'bg-accent-green' :
                            log.level === 'warning' ? 'bg-yellow-500' :
                            'bg-red-500'
                          }`} />
                          <span className="text-gray-500 font-mono">{log.time}</span>
                          <span className={`uppercase ${
                            log.level === 'info' ? 'text-accent-green' :
                            log.level === 'warning' ? 'text-yellow-500' :
                            'text-red-500'
                          }`}>
                            {log.level}
                          </span>
                          <span className="text-gray-300 flex-1">{log.message}</span>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}

                {/* Users Tab */}
                {activeTab === 'users' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-lg font-bold">Users</h2>
                      <button className="btn-primary text-xs">+ Add</button>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr className="border-b border-glass-border text-gray-400">
                            <th className="pb-2 font-normal">User</th>
                            <th className="pb-2 font-normal">Role</th>
                            <th className="pb-2 font-normal">Detections</th>
                            <th className="pb-2 font-normal">Last Active</th>
                            <th className="pb-2 font-normal">Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {users.map((user) => (
                            <tr key={user.id} className="border-b border-glass-border">
                              <td className="py-2">
                                <div className="font-medium">{user.name}</div>
                                <div className="text-gray-500 text-xs">{user.email}</div>
                              </td>
                              <td className="py-2">
                                <span className="badge badge-primary text-xs">{user.role}</span>
                              </td>
                              <td className="py-2">{user.detections.toLocaleString()}</td>
                              <td className="py-2 text-gray-500">{user.lastActive}</td>
                              <td className="py-2">
                                <button className="text-gray-400 hover:text-white mr-2">
                                  <FaCogs />
                                </button>
                                <button className="text-gray-400 hover:text-red-500">
                                  <FaTrash />
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </motion.div>
                )}

                {/* Settings Tab */}
                {activeTab === 'settings' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <h2 className="text-lg font-bold mb-4">Settings</h2>

                    <div className="space-y-4">
                      <div className="glass rounded-lg p-4">
                        <h3 className="text-sm font-semibold mb-3">General</h3>
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <div>
                              <div className="text-sm font-medium">Debug Mode</div>
                              <div className="text-xs text-gray-500">Enable detailed logging</div>
                            </div>
                            <label className="relative inline-flex items-center cursor-pointer">
                              <input type="checkbox" className="sr-only peer" />
                              <div className="w-9 h-5 bg-gray-700 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary"></div>
                            </label>
                          </div>
                          <div className="flex items-center justify-between">
                            <div>
                              <div className="text-sm font-medium">Auto Model Loading</div>
                              <div className="text-xs text-gray-500">Load models on startup</div>
                            </div>
                            <label className="relative inline-flex items-center cursor-pointer">
                              <input type="checkbox" className="sr-only peer" defaultChecked />
                              <div className="w-9 h-5 bg-gray-700 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary"></div>
                            </label>
                          </div>
                        </div>
                      </div>

                      <div className="glass rounded-lg p-4">
                        <h3 className="text-sm font-semibold mb-3">Performance</h3>
                        <div className="space-y-3">
                          <div>
                            <label className="block text-xs text-gray-400 mb-1">Default Model</label>
                            <select className="input text-sm">
                              <option>yolov8n (Nano)</option>
                              <option>yolov8s (Small)</option>
                              <option>yolov8m (Medium)</option>
                            </select>
                          </div>
                        </div>
                      </div>

                      <button className="btn-primary text-sm">Save Settings</button>
                    </div>
                  </motion.div>
                )}
              </div>
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
    <div className="card flex items-center space-x-3 p-3">
      <div className="text-xl">{icon}</div>
      <div>
        <div className="text-xs text-gray-400">{label}</div>
        <div className="text-base font-bold">{value}</div>
        <div className="text-xs text-gray-500">{subtext}</div>
      </div>
    </div>
  )
}

function QuickStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass rounded-lg p-3 text-center">
      <div className="text-xl font-bold text-primary">{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  )
}
