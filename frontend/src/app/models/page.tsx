'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { FaMicrochip, FaTachometerAlt, FaCrosshairs, FaCheckCircle, FaTimesCircle, FaBalanceScale } from 'react-icons/fa'
import Link from 'next/link'
import { FaEye } from 'react-icons/fa'

const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL || ''
const API_URL = /^https?:\/\/(localhost|127\.0\.0\.1):8000\/?$/.test(configuredApiUrl)
  ? ''
  : configuredApiUrl.replace(/\/$/, '')

export default function ModelsPage() {
  const [models, setModels] = useState<any[]>([])
  const [systemInfo, setSystemInfo] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [comparisonMode, setComparisonMode] = useState(false)

  const fetchModels = async () => {
    setLoading(true)
    setError(null)
    try {
      const endpoints = [`${API_URL}/api/detection/models`, `${API_URL}/api/models`]
      let data: any = null

      for (const endpoint of endpoints) {
        try {
          const response = await fetch(endpoint, { cache: 'no-store' })
          if (!response.ok) continue
          const payload = await response.json()
          if (Array.isArray(payload.models)) {
            data = payload
            break
          }
        } catch (err) {
          console.warn('Model endpoint failed:', endpoint, err)
        }
      }

      if (!data) {
        throw new Error('Could not load models from the backend')
      }

      setModels(data.models || [])
      setSystemInfo(data.system || null)
    } catch (err: any) {
      console.error('Failed to fetch models:', err)
      setError(err.message || 'Failed to load models')
      setModels([])
      setSystemInfo(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchModels()
  }, [])

  const getModelTypeColor = (type: string) => {
    switch (type) {
      case 'realtime':
        return 'text-accent-green bg-accent-green/20 border-accent-green/30'
      case 'research':
        return 'text-secondary bg-secondary/20 border-secondary/30'
      case 'openvocabulary':
        return 'text-accent-pink bg-accent-pink/20 border-accent-pink/30'
      default:
        return 'text-primary bg-primary/20 border-primary/30'
    }
  }

  const getModelIcon = (type: string) => {
    switch (type) {
      case 'realtime':
        return <FaTachometerAlt className="text-accent-green" />
      case 'research':
        return <FaCrosshairs className="text-secondary" />
      case 'openvocabulary':
        return <FaMicrochip className="text-accent-pink" />
      default:
        return <FaMicrochip className="text-primary" />
    }
  }

  const toggleModelSelection = (modelId: string) => {
    setSelectedModels(prev => {
      if (prev.includes(modelId)) {
        return prev.filter(id => id !== modelId)
      }
      if (prev.length >= 3) {
        return prev
      }
      return [...prev, modelId]
    })
  }

  const runnableModels = models.filter((item) => item.runnable)
  const catalogOnlyModels = models.filter((item) => !item.runnable)
  const modelFamilies = Array.from(new Set(models.map((item) => item.family || item.type))).sort()

  const selectedModelsData = models.filter(m => selectedModels.includes(m.id))

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400 text-sm">Loading models...</p>
        </div>
      </div>
    )
  }

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
              <Link href="/models" className="text-primary font-semibold transition-colors hidden md:block">
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
      <div className="flex-1 flex flex-col pt-20 pb-8 px-4 overflow-hidden">
        <div className="max-w-6xl mx-auto w-full flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <motion.div 
            className="text-center mb-3"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-2xl md:text-3xl font-bold mb-1">
              <span className="gradient-text">Model Comparison</span>
            </h1>
            <p className="text-gray-400 text-xs">
              Select up to 3 models to compare their performance
            </p>
          </motion.div>

          {/* System Info Banner */}
          {systemInfo && (
            <motion.div 
              className="card mb-3 py-2 px-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center space-x-3">
                  <div className={`w-2 h-2 rounded-full ${systemInfo.cuda_available ? 'bg-accent-green animate-pulse' : 'bg-gray-500'}`} />
                  <div>
                    <div className="text-xs text-gray-400">System</div>
                    <div className="text-sm font-semibold">
                      {systemInfo.cuda_available ? (
                        <span className="text-accent-green">GPU ({systemInfo.device})</span>
                      ) : (
                        <span className="text-gray-400">CPU</span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-xs text-gray-400">Recommended</div>
                  <div className="text-sm font-semibold text-primary">
                    {systemInfo.recommended_model || 'yolov8n'}
                  </div>
                </div>
                <div className="flex gap-2">
                  <span className="text-xs text-gray-400">{models.length} Total</span>
                  <span className="text-xs text-accent-green">{runnableModels.length} Runnable</span>
                </div>
              </div>
            </motion.div>
          )}

          {/* Comparison Mode Toggle */}
          <div className="flex items-center justify-between mb-3">
            <button
              onClick={() => setComparisonMode(!comparisonMode)}
              className={`flex items-center px-3 py-2 rounded-lg text-sm transition-all ${
                comparisonMode ? 'bg-primary/20 text-primary border border-primary/30' : 'glass text-gray-400 hover:text-white'
              }`}
            >
              <FaBalanceScale className="mr-2" />
              {comparisonMode ? 'Exit Comparison' : 'Compare Models'}
            </button>
            {selectedModels.length > 0 && (
              <span className="text-xs text-gray-400">
                Selected: {selectedModels.length}/3
              </span>
            )}
          </div>

          {error && (
            <div className="card mb-3 p-4 text-center">
              <p className="text-sm text-red-400">{error}</p>
              <p className="mt-1 text-xs text-gray-500">Make sure the backend is running on port 8000.</p>
              <button onClick={fetchModels} className="btn-secondary mt-3 text-sm">
                Retry
              </button>
            </div>
          )}

          {/* Comparison View */}
          {comparisonMode && selectedModelsData.length > 0 && (
            <motion.div 
              className="card mb-3 p-4 overflow-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h2 className="text-base font-semibold mb-3">Model Comparison</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-glass-border text-gray-400 text-xs">
                      <th className="pb-2 font-normal">Model</th>
                      <th className="pb-2 font-normal">FPS (GPU)</th>
                      <th className="pb-2 font-normal">Inference</th>
                      <th className="pb-2 font-normal">Input Size</th>
                      <th className="pb-2 font-normal">Accuracy</th>
                      <th className="pb-2 font-normal">Status</th>
                    </tr>
                  </thead>
                  <tbody className="text-xs">
                    {selectedModelsData.map((model) => (
                      <tr key={model.id} className="border-b border-glass-border">
                        <td className="py-2 font-medium">
                          <div className="flex items-center space-x-2">
                            {getModelIcon(model.type)}
                            <span>{model.name}</span>
                          </div>
                        </td>
                        <td className="py-2 text-accent-green">
                          {model.expected_fps_gpu || 'N/A'} FPS
                        </td>
                        <td className="py-2 text-primary">
                          {model.input_size ? Math.round(1000 / (model.expected_fps_gpu || 30)) : 'N/A'}ms
                        </td>
                        <td className="py-2 text-gray-400">
                          {model.input_size || 'N/A'}
                        </td>
                        <td className="py-2 text-secondary">
                          {model.accuracy_score ? `${Math.round(model.accuracy_score * 100)}%` : 'N/A'}
                        </td>
                        <td className="py-2">
                          {model.runnable ? (
                            <span className="text-accent-green">✓ Runnable</span>
                          ) : (
                            <span className="text-gray-500">Catalog</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              {/* Visual Comparison Bars */}
              <div className="mt-4 space-y-3">
                <h3 className="text-xs text-gray-400">Performance Visualization</h3>
                {selectedModelsData.map((model) => {
                  const fps = model.expected_fps_gpu || model.expected_fps_cpu || 30
                  const maxFps = Math.max(...selectedModelsData.map(m => m.expected_fps_gpu || m.expected_fps_cpu || 30))
                  return (
                    <div key={model.id} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-medium">{model.name}</span>
                        <span className="text-gray-400">{fps} FPS</span>
                      </div>
                      <div className="h-2 bg-background-secondary rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-primary to-accent-green rounded-full transition-all duration-500"
                          style={{ width: `${(fps / maxFps) * 100}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </motion.div>
          )}

          {/* Models Grid - Scrollable */}
          <div className="flex-1 overflow-y-auto mb-3">
            {models.length === 0 && !error ? (
              <div className="card p-6 text-center text-sm text-gray-400">
                No models returned by the backend.
              </div>
            ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {models.map((model, index) => (
                <motion.div
                  key={model.id}
                  className={`card cursor-pointer transition-all duration-300 p-4 ${
                    selectedModels.includes(model.id) ? 'border-primary shadow-neon' : ''
                  }`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.02 }}
                  whileHover={{ scale: 1.02 }}
                  onClick={() => comparisonMode && toggleModelSelection(model.id)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <div className="text-2xl">{getModelIcon(model.type)}</div>
                      <div>
                        <h3 className="text-base font-bold">{model.name}</h3>
                        <div className="flex flex-wrap gap-1 mt-1">
                          <span className={`text-xs px-2 py-0.5 rounded-full border ${getModelTypeColor(model.type)}`}>
                            {model.type}
                          </span>
                        </div>
                      </div>
                    </div>
                    {model.is_loaded ? (
                      <FaCheckCircle className="text-accent-green text-lg" />
                    ) : (
                      <FaTimesCircle className="text-gray-600 text-lg" />
                    )}
                  </div>

                  <div className="space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">Input Size</span>
                      <span className="text-white font-mono">{model.input_size}px</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">FPS (GPU)</span>
                      <span className="text-accent-green">{model.expected_fps_gpu || 'N/A'}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">Status</span>
                      <span className={model.runnable ? 'text-accent-green' : 'text-gray-500'}>
                        {model.runnable ? 'Runnable' : 'Catalog'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">Accuracy</span>
                      <span className="text-secondary">
                        {model.accuracy_score ? `${Math.round(model.accuracy_score * 100)}%` : 'N/A'}
                      </span>
                    </div>
                  </div>

                  {comparisonMode && (
                    <div className="mt-2 pt-2 border-t border-glass-border">
                      <span className={`text-xs ${
                        selectedModels.includes(model.id) ? 'text-primary' : 'text-gray-500'
                      }`}>
                        {selectedModels.includes(model.id) ? '✓ Selected' : 'Click to compare'}
                      </span>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
            )}
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
