'use client'

import { useState, useCallback, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion } from 'framer-motion'
import { FaUpload, FaImage, FaSpinner, FaDownload, FaTrash } from 'react-icons/fa'
import Link from 'next/link'
import { FaEye, FaBolt } from 'react-icons/fa'

const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL || ''
const API_URL = /^https?:\/\/(localhost|127\.0\.0\.1):8000\/?$/.test(configuredApiUrl)
  ? ''
  : configuredApiUrl.replace(/\/$/, '')

const boxColors = ['#00d4ff', '#22c55e', '#f97316', '#a855f7', '#ec4899', '#eab308']

type ModelComparisonRow = {
  model_id: string
  name: string
  family: string
  type: string
  num_detections: number
  inference_time_ms: number
  fps: number
  avg_confidence: number
  classes: string[]
  error?: string | null
}

function DetectionImage({ src, result }: { src: string; result: any }) {
  const width = result?.original_width || 1
  const height = result?.original_height || 1
  const detections = result?.results || []

  return (
    <div className="relative overflow-hidden rounded-lg border border-glass-border bg-black">
      <img
        src={src}
        alt="Detection Result"
        className="block w-full"
      />
      {detections.length > 0 && (
        <svg
          className="absolute inset-0 h-full w-full pointer-events-none"
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
        >
          {detections.map((det: any, idx: number) => {
            const color = boxColors[idx % boxColors.length]
            const x = Math.max(0, det.bbox?.x1 || 0)
            const y = Math.max(0, det.bbox?.y1 || 0)
            const w = Math.max(0, (det.bbox?.x2 || 0) - x)
            const h = Math.max(0, (det.bbox?.y2 || 0) - y)
            const labelY = y > 22 ? y - 4 : y + 18

            return (
              <g key={`${det.class_name}-${idx}`}>
                <rect
                  x={x}
                  y={y}
                  width={w}
                  height={h}
                  stroke={color}
                  strokeWidth={Math.max(width, height) * 0.004}
                  fill="none"
                />
                <rect
                  x={x}
                  y={Math.max(0, labelY - 16)}
                  width={Math.min(width - x, Math.max(90, det.class_name.length * 9 + 46))}
                  height="18"
                  fill={color}
                  opacity="0.92"
                />
                <text
                  x={x + 5}
                  y={Math.max(13, labelY - 3)}
                  fill="#031018"
                  fontSize={Math.max(width, height) * 0.018}
                  fontWeight="700"
                >
                  {det.class_name} {(det.confidence * 100).toFixed(0)}%
                </text>
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}

function ComparisonDashboard({ comparison, comparing }: { comparison: any; comparing: boolean }) {
  const models: ModelComparisonRow[] = comparison?.models || []
  const successful = models.filter((item) => !item.error)
  const highestConfidence = successful.reduce<ModelComparisonRow | null>((best, item) => (
    !best || item.avg_confidence > best.avg_confidence ? item : best
  ), null)
  const byId = new Map<string, ModelComparisonRow>(models.map((item) => [item.model_id, item]))
  const bestDetection = byId.get(comparison?.summary?.best_detection_model)
  const fastest = byId.get(comparison?.summary?.fastest_model)

  return (
    <motion.div
      className="card mt-6"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold">All-Model Comparison</h2>
          <p className="text-xs text-gray-500">Same image analyzed across every runnable model.</p>
        </div>
        {comparing && (
          <span className="flex items-center gap-1 text-xs text-primary">
            <FaSpinner className="animate-spin" />
            Analyzing all models
          </span>
        )}
      </div>

      {models.length > 0 ? (
        <div className="space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <div className="glass rounded-lg p-3">
              <div className="text-xs text-gray-400">Most Objects</div>
              <div className="mt-1 text-base font-semibold text-primary">{bestDetection?.name || '-'}</div>
              <div className="text-xs text-gray-500">{bestDetection ? `${bestDetection.num_detections} detections` : 'No successful run'}</div>
            </div>
            <div className="glass rounded-lg p-3">
              <div className="text-xs text-gray-400">Fastest</div>
              <div className="mt-1 text-base font-semibold text-accent-green">{fastest?.name || '-'}</div>
              <div className="text-xs text-gray-500">{fastest ? `${fastest.inference_time_ms?.toFixed(1)}ms` : 'No timing data'}</div>
            </div>
            <div className="glass rounded-lg p-3">
              <div className="text-xs text-gray-400">Highest Avg Confidence</div>
              <div className="mt-1 text-base font-semibold text-secondary">{highestConfidence?.name || '-'}</div>
              <div className="text-xs text-gray-500">
                {highestConfidence ? `${(highestConfidence.avg_confidence * 100).toFixed(1)}%` : 'No detections'}
              </div>
            </div>
          </div>

          <div className="max-h-72 overflow-y-auto rounded-lg border border-glass-border">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-gray-950/95 text-gray-400">
                <tr>
                  <th className="px-2 py-2 text-left font-medium">Model</th>
                  <th className="px-2 py-2 text-right font-medium">Objects</th>
                  <th className="px-2 py-2 text-right font-medium">Avg Conf</th>
                  <th className="px-2 py-2 text-right font-medium">Time</th>
                  <th className="px-2 py-2 text-right font-medium">FPS</th>
                  <th className="px-2 py-2 text-left font-medium">Classes</th>
                </tr>
              </thead>
              <tbody>
                {models.map((item) => (
                  <tr key={item.model_id} className="border-t border-white/10">
                    <td className="px-2 py-2">
                      <div className="font-semibold text-white">{item.name}</div>
                      <div className="text-gray-500">{item.family}</div>
                    </td>
                    <td className="px-2 py-2 text-right text-primary font-semibold">
                      {item.error ? '-' : item.num_detections}
                    </td>
                    <td className="px-2 py-2 text-right text-gray-300">
                      {item.error ? '-' : `${(item.avg_confidence * 100).toFixed(1)}%`}
                    </td>
                    <td className="px-2 py-2 text-right text-gray-300">
                      {item.error ? 'Error' : `${item.inference_time_ms?.toFixed(1)}ms`}
                    </td>
                    <td className="px-2 py-2 text-right text-gray-300">
                      {item.error ? '-' : item.fps?.toFixed(1)}
                    </td>
                    <td className="px-2 py-2 text-gray-400">
                      {item.error ? item.error : (item.classes || []).slice(0, 6).join(', ') || 'None'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="glass rounded-lg p-4 text-center text-xs text-gray-500">
          {comparing ? 'Running comparison...' : 'Run detection to compare all models'}
        </div>
      )}
    </motion.div>
  )
}

export default function DetectionPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [detecting, setDetecting] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [comparison, setComparison] = useState<any>(null)
  const [comparing, setComparing] = useState(false)
  const [model, setModel] = useState('auto')
  const [models, setModels] = useState<any[]>([])
  const [confidence, setConfidence] = useState(0.5)
  const [error, setError] = useState<string | null>(null)
  const [modelRecommendation, setModelRecommendation] = useState<any>(null)
  const [selectingModel, setSelectingModel] = useState(false)
  const [apiStatus, setApiStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking')

  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const response = await fetch(`${API_URL}/api/detection/health`)
        if (response.ok) {
          setApiStatus('connected')
        } else {
          setApiStatus('disconnected')
        }
      } catch (err) {
        setApiStatus('disconnected')
      }
    }

    const fetchModels = async () => {
      try {
        const response = await fetch(`${API_URL}/api/detection/models`)
        const data = await response.json()
        const runnableModels = (data.models || []).filter((item: any) => item.runnable)
        setModels(runnableModels)
        if (apiStatus !== 'connected') {
          setApiStatus('connected')
        }
      } catch (err) {
        console.error('Failed to fetch models:', err)
        setApiStatus('disconnected')
      }
    }

    checkApiStatus()
    fetchModels()
  }, [apiStatus])

  const recommendModelForFile = async (file: File) => {
    setSelectingModel(true)
    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_URL}/api/detection/recommend/image`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Could not select the best model')
      }

      const data = await response.json()
      setModelRecommendation(data)
      setModel(data.recommended_model || 'auto')
      setApiStatus('connected')
    } catch (err: any) {
      console.error('Failed to recommend model:', err)
      setModelRecommendation(null)
      setModel('auto')
      setError(err.message || 'Could not select the best model')
    } finally {
      setSelectingModel(false)
    }
  }

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setSelectedFile(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
      setComparison(null)
      setError(null)
      setModelRecommendation(null)
      recommendModelForFile(file)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.webp']
    },
    maxFiles: 1
  })

  const handleDetect = async () => {
    if (!selectedFile) return

    setDetecting(true)
    setError(null)
    setComparison(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('model_name', modelRecommendation?.fallback_model ? 'auto' : model)
      formData.append('confidence_threshold', confidence.toString())

      const response = await fetch(`${API_URL}/api/detection/detect/image`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Detection failed')
      }

      const data = await response.json()
      setResult(data)
      setDetecting(false)
      await handleCompareAll(selectedFile)
    } catch (err: any) {
      setError(err.message || 'An error occurred')
    } finally {
      setDetecting(false)
    }
  }

  const handleCompareAll = async (file: File) => {
    setComparing(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('confidence_threshold', confidence.toString())

      const response = await fetch(`${API_URL}/api/detection/detect/compare-all`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Model comparison failed')
      }

      const data = await response.json()
      setComparison(data)
    } catch (err: any) {
      console.error('Comparison error:', err)
      setError(err.message || 'Model comparison failed')
    } finally {
      setComparing(false)
    }
  }

  const handleDownload = () => {
    if (result?.output_image_base64) {
      const link = document.createElement('a')
      link.href = result.output_image_base64
      link.download = `detection_${selectedFile?.name || 'result'}.jpg`
      link.click()
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass border-b border-glass-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link href="/" className="flex items-center space-x-2">
              <FaEye className="text-primary text-2xl" />
              <span className="text-xl font-bold gradient-text">LenScope AI</span>
            </Link>
            
            <div className="flex items-center space-x-6">
              <Link href="/detect" className="text-primary font-semibold transition-colors hidden md:block">
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
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content - Compact Layout */}
      <div className="flex-1 pt-20 pb-8 px-4">
        <div className="max-w-6xl mx-auto w-full">
          {/* Header */}
          <motion.div 
            className="text-center mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-3xl md:text-4xl font-bold mb-2">
              <span className="gradient-text">Object Detection</span>
            </h1>
            <p className="text-gray-400 text-sm">
              Upload an image to detect objects using AI models
            </p>
            {/* API Status Indicator */}
            <div className="mt-3 flex items-center justify-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                apiStatus === 'connected' ? 'bg-accent-green animate-pulse' : 
                apiStatus === 'disconnected' ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'
              }`} />
              <span className={`text-xs ${
                apiStatus === 'connected' ? 'text-accent-green' : 
                apiStatus === 'disconnected' ? 'text-red-400' : 'text-yellow-400'
              }`}>
                {apiStatus === 'connected' ? 'Backend Connected' : 
                 apiStatus === 'disconnected' ? 'Backend Disconnected' : 'Checking...'}
              </span>
            </div>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Upload Section */}
            <motion.div 
              className="card"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <h2 className="text-lg font-semibold mb-3 flex items-center">
                <FaImage className="mr-2 text-primary" />
                Upload Image
              </h2>

              {/* Dropzone */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                  isDragActive 
                    ? 'border-primary bg-primary/10' 
                    : 'border-glass-border hover:border-primary/50'
                }`}
              >
                <input {...getInputProps()} />
                <FaUpload className="text-3xl text-gray-500 mx-auto mb-3" />
                {isDragActive ? (
                  <p className="text-primary text-sm">Drop the image here...</p>
                ) : (
                  <p className="text-gray-400 text-sm">
                    Drag & drop an image, or click to select
                  </p>
                )}
              </div>

              {/* Preview */}
              {preview && (
                <div className="mt-3">
                  <img 
                    src={preview} 
                    alt="Preview" 
                    className="w-full h-48 object-cover rounded-lg"
                  />
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-xs text-gray-400 truncate">{selectedFile?.name}</span>
                    <button
                      onClick={() => {
                        setSelectedFile(null)
                        setPreview(null)
                        setResult(null)
                        setComparison(null)
                        setModel('auto')
                        setModelRecommendation(null)
                      }}
                      className="text-red-400 hover:text-red-300"
                    >
                      <FaTrash />
                    </button>
                  </div>
                </div>
              )}

              {/* Settings */}
              <div className="mt-4 space-y-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Model</label>
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="input text-sm py-2"
                    disabled={selectingModel}
                  >
                    <option value="auto">Auto (Best Model Selection)</option>
                    {(models.length > 0 ? models : [{ id: 'yolov8n', name: 'YOLOv8n', family: 'YOLOv8' }]).map((item: any) => (
                      <option key={item.id} value={item.id}>
                        {item.name} ({item.family || item.type})
                      </option>
                    ))}
                  </select>
                  <div className="mt-1 text-xs text-gray-500">
                    {selectingModel
                      ? 'Selecting best model...'
                      : modelRecommendation
                        ? `Selected ${modelRecommendation.recommended_model}${modelRecommendation.fallback_model ? `, fallback ${modelRecommendation.fallback_model}` : ''}`
                        : 'Auto mode will choose at detection time'}
                  </div>
                </div>

                <div>
                  <label className="block text-xs text-gray-400 mb-1">
                    Confidence Threshold: <span className="text-primary font-semibold">{(confidence * 100).toFixed(0)}%</span>
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="1"
                    step="0.05"
                    value={confidence}
                    onChange={(e) => setConfidence(parseFloat(e.target.value))}
                    className="w-full accent-primary"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>0.1</span>
                    <span>0.5</span>
                    <span>1.0</span>
                  </div>
                </div>

                <button
                  onClick={handleDetect}
                  disabled={!selectedFile || detecting || selectingModel}
                  className="btn-primary w-full flex items-center justify-center text-sm py-2"
                >
                  {detecting || selectingModel ? (
                    <>
                      <FaSpinner className="animate-spin mr-2" />
                      {selectingModel ? 'Selecting Model...' : 'Detecting...'}
                    </>
                  ) : (
                    <>
                      <FaImage className="mr-2" />
                      Run Detection
                    </>
                  )}
                </button>
              </div>

              {error && (
                <div className="mt-3 p-2 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-xs">
                  {error}
                </div>
              )}
            </motion.div>

            {/* Results Section */}
            <motion.div 
              className="card"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <h2 className="text-lg font-semibold mb-3">Results</h2>

              {result ? (
                <div className="space-y-3">
                  {/* Output Image */}
                  {(result.output_image_base64 || preview) && (
                    <div className="relative">
                      <DetectionImage src={result.output_image_base64 || preview} result={result} />
                      <button
                        onClick={handleDownload}
                        className="absolute top-2 right-2 btn-secondary text-xs"
                        disabled={!result.output_image_base64}
                      >
                        <FaDownload className="mr-1" />
                        Download
                      </button>
                    </div>
                  )}

                  {/* Stats */}
                  <div className="grid grid-cols-3 gap-2">
                    <div className="glass rounded-lg p-2 text-center">
                      <div className="text-xl font-bold text-primary">{result.num_detections}</div>
                      <div className="text-xs text-gray-400">Objects</div>
                    </div>
                    <div className="glass rounded-lg p-2 text-center">
                      <div className="text-xl font-bold text-accent-green">{result.inference_time_ms?.toFixed(1)}ms</div>
                      <div className="text-xs text-gray-400">Inference</div>
                    </div>
                    <div className="glass rounded-lg p-2 text-center">
                      <div className="text-xl font-bold text-secondary">{result.fps?.toFixed(1)}</div>
                      <div className="text-xs text-gray-400">FPS</div>
                    </div>
                  </div>

                  {result.auto_selection && (
                    <div className="glass rounded-lg p-2 text-xs text-gray-300">
                      <span className="text-primary font-semibold">{result.auto_selection.selected_model}</span>
                      <span className="text-gray-500"> · {result.auto_selection.reason}</span>
                    </div>
                  )}

                  {/* Detections List */}
                  {result.results && result.results.length > 0 && (
                    <div>
                      <h3 className="text-xs text-gray-400 mb-2">Detected Objects</h3>
                      <div className="space-y-1 max-h-32 overflow-y-auto">
                        {result.results.map((det: any, idx: number) => (
                          <div key={idx} className="glass rounded-lg p-2 flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <span className="badge badge-primary text-xs">{det.class_name}</span>
                              <span className="text-xs text-gray-400">
                                {(det.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-48 text-gray-500">
                  <FaImage className="text-3xl mb-3 opacity-50" />
                  <p className="text-sm">Upload an image and run detection</p>
                </div>
              )}
            </motion.div>
          </div>

          {(result || comparing || comparison) && (
            <ComparisonDashboard comparison={comparison} comparing={comparing} />
          )}
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
