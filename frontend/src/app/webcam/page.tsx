'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { motion } from 'framer-motion'
import { FaCamera, FaStop, FaDownload, FaImage } from 'react-icons/fa'
import Link from 'next/link'
import { FaEye } from 'react-icons/fa'

const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL || ''
const API_URL = /^https?:\/\/(localhost|127\.0\.0\.1):8000\/?$/.test(configuredApiUrl)
  ? ''
  : configuredApiUrl.replace(/\/$/, '')

export default function WebcamPage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isCameraReady, setIsCameraReady] = useState(false)
  const [detections, setDetections] = useState<any[]>([])
  const [fps, setFps] = useState(0)
  const [model, setModel] = useState('yolov8m')
  const [models, setModels] = useState<any[]>([])
  const [confidence, setConfidence] = useState(0.35)
  const [error, setError] = useState<string | null>(null)
  const [cameraMessage, setCameraMessage] = useState('Click Start Webcam to turn on the camera')
  const [objectCounts, setObjectCounts] = useState<Record<string, number>>({})
  const [movementCounts, setMovementCounts] = useState({ up: 0, down: 0 })
  const [frameSize, setFrameSize] = useState({ width: 640, height: 480 })
  const animationRef = useRef<number>()
  const lastFrameTime = useRef<number>(0)
  const detectingRef = useRef(false)
  const movementTracksRef = useRef<Map<string, { x: number; y: number; direction: 'up' | 'down' | null }>>(new Map())
  const [snapshots, setSnapshots] = useState<string[]>([])
  const [showSnapshots, setShowSnapshots] = useState(false)
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
        const preferredOrder = ['yolov8m', 'yolov8s', 'yolov8l', 'yolov8x', 'yolov8s-world', 'yolov8n']
        const runnableModels = (data.models || [])
          .filter((item: any) => item.runnable && preferredOrder.includes(item.id))
          .sort((a: any, b: any) => preferredOrder.indexOf(a.id) - preferredOrder.indexOf(b.id))
        setModels(runnableModels)
        setApiStatus('connected')
      } catch (err) {
        console.error('Failed to fetch models:', err)
        setApiStatus('disconnected')
      }
    }

    checkApiStatus()
    fetchModels()
  }, [])

  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  const updateVideoSize = useCallback(() => {
    const video = videoRef.current
    if (!video) return

    const width = video.videoWidth || 640
    const height = video.videoHeight || 480
    setFrameSize({ width, height })
  }, [])

  const getScreenshot = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current

    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null
    }

    const width = video.videoWidth || frameSize.width || 640
    const height = video.videoHeight || frameSize.height || 480
    canvas.width = width
    canvas.height = height

    const context = canvas.getContext('2d')
    if (!context) return null

    context.drawImage(video, 0, 0, width, height)
    return canvas.toDataURL('image/jpeg', 0.92)
  }, [frameSize.height, frameSize.width])

  const captureAndDetect = useCallback(async () => {
    if (!videoRef.current) {
      detectingRef.current = false
      return
    }

    const screenshot = getScreenshot()
    if (!screenshot) {
      setError('Camera frame is not ready yet')
      detectingRef.current = false
      return
    }

    try {
      const response = await fetch(screenshot)
      const blob = await response.blob()
      
      const formData = new FormData()
      formData.append('file', blob, 'webcam_capture.jpg')
      formData.append('model_name', model === 'auto' ? 'yolov8m' : model)
      formData.append('confidence_threshold', confidence.toString())
      if (model === 'yolov8s-world') {
        formData.append(
          'open_vocab_prompt',
          'person, phone, laptop, keyboard, mouse, book, bottle, cup, bag, backpack, chair, table, monitor, screen, paper, box, clothing, object, electronic device'
        )
      }

      const fetchResponse = await fetch(`${API_URL}/api/detection/detect/image`, {
        method: 'POST',
        body: formData,
      })

      if (fetchResponse.ok) {
        const data = await fetchResponse.json()
        const results = data.results || []
        setDetections(results)
        if (data.original_width && data.original_height) {
          setFrameSize({ width: data.original_width, height: data.original_height })
        }
        setError(null)
        
        const counts: Record<string, number> = {}
        results.forEach((det: any) => {
          counts[det.class_name] = (counts[det.class_name] || 0) + 1
        })
        setObjectCounts(counts)
        updateMovementCounters(results, data.original_height || frameSize.height)
      } else {
        setError('Webcam detection request failed')
      }
    } catch (err) {
      console.error('Detection error:', err)
      setError('Webcam detection failed')
    } finally {
      detectingRef.current = false
    }
  }, [getScreenshot, model, confidence, frameSize.height])

  const updateMovementCounters = (results: any[], originalHeight: number) => {
    const threshold = Math.max(18, originalHeight * 0.04)
    const seenKeys = new Set<string>()
    const sorted = [...results].sort((a, b) => {
      const classCompare = String(a.class_name).localeCompare(String(b.class_name))
      if (classCompare !== 0) return classCompare
      return (a.bbox.x1 + a.bbox.x2) - (b.bbox.x1 + b.bbox.x2)
    })

    setMovementCounts((previousCounts) => {
      const nextCounts = { ...previousCounts }

      sorted.forEach((det, index) => {
        const centerY = (det.bbox.y1 + det.bbox.y2) / 2
        const centerX = (det.bbox.x1 + det.bbox.x2) / 2
        const key = `${det.class_name}-${index}`
        seenKeys.add(key)
        const previousTrack = movementTracksRef.current.get(key)

        if (!previousTrack) {
          movementTracksRef.current.set(key, { x: centerX, y: centerY, direction: null })
          return
        }

        const dy = centerY - previousTrack.y
        const dx = Math.abs(centerX - previousTrack.x)

        if (Math.abs(dy) >= threshold && dx < Math.max(80, threshold * 3)) {
          const direction = dy < 0 ? 'up' : 'down'
          if (previousTrack.direction !== direction) {
            nextCounts[direction] += 1
          }
          movementTracksRef.current.set(key, { x: centerX, y: centerY, direction })
        } else {
          movementTracksRef.current.set(key, { ...previousTrack, x: centerX, y: centerY })
        }
      })

      Array.from(movementTracksRef.current.keys()).forEach((key) => {
        if (!seenKeys.has(key)) {
          movementTracksRef.current.delete(key)
        }
      })

      return nextCounts
    })
  }

  const resetMovementCounters = () => {
    movementTracksRef.current.clear()
    setMovementCounts({ up: 0, down: 0 })
  }

  const beginDetectionLoop = useCallback(() => {
    setIsStreaming(true)
    setError(null)
    setCameraMessage('Camera live. Running detection.')
    lastFrameTime.current = 0
    
    const detect = async () => {
      const now = performance.now()
      const elapsed = now - lastFrameTime.current
      
      if (elapsed >= 900 && !detectingRef.current) {
        detectingRef.current = true
        lastFrameTime.current = now
        await captureAndDetect()
        setFps(Math.round(1000 / Math.max(elapsed, 1)))
      }
      
      animationRef.current = requestAnimationFrame(detect)
    }
    
    detect()
  }, [captureAndDetect])

  const startStream = useCallback(async () => {
    if (isStreaming) return

    try {
      setCameraMessage('Starting camera. Allow permission if your browser asks.')
      setError(null)

      if (!streamRef.current) {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('Camera API is not available in this browser')
        }

        streamRef.current = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user',
          },
        })
      }

      const video = videoRef.current
      if (!video) {
        throw new Error('Camera preview is not available')
      }

      video.srcObject = streamRef.current
      video.muted = true
      video.playsInline = true
      await video.play()
      updateVideoSize()
      setIsCameraReady(true)
      beginDetectionLoop()
    } catch (err: any) {
      console.error('Camera error:', err)
      setIsCameraReady(false)
      setIsStreaming(false)
      streamRef.current?.getTracks().forEach((track) => track.stop())
      streamRef.current = null
      setCameraMessage('Camera blocked or unavailable')
      setError(err?.message || 'Camera access failed. Allow camera permission and close other apps using the camera.')
    }
  }, [beginDetectionLoop, isStreaming, updateVideoSize])

  const stopStream = useCallback(() => {
    setIsStreaming(false)
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }
    setDetections([])
    setObjectCounts({})
    resetMovementCounters()
    setFps(0)
    setCameraMessage(isCameraReady ? 'Camera ready' : 'Click Start Webcam to turn on the camera')
    detectingRef.current = false
  }, [isCameraReady])

  const totalObjects = Object.values(objectCounts).reduce((a, b) => a + b, 0)
  const totalMovements = movementCounts.up + movementCounts.down

  const captureSnapshot = useCallback(async () => {
    if (!videoRef.current || !isCameraReady) {
      setError('Camera is not ready for a snapshot yet')
      return
    }

    const screenshot = getScreenshot()
    if (!screenshot) {
      setError('Camera frame is not ready yet')
      return
    }

    try {
      const response = await fetch(screenshot)
      const blob = await response.blob()
      
      const formData = new FormData()
      formData.append('file', blob, `snapshot_${Date.now()}.jpg`)
      formData.append('model_name', model === 'auto' ? 'yolov8m' : model)
      formData.append('confidence_threshold', confidence.toString())
      if (model === 'yolov8s-world') {
        formData.append(
          'open_vocab_prompt',
          'person, phone, laptop, keyboard, mouse, book, bottle, cup, bag, backpack, chair, table, monitor, screen, paper, box, clothing, object, electronic device'
        )
      }

      const fetchResponse = await fetch(`${API_URL}/api/detection/detect/image`, {
        method: 'POST',
        body: formData,
      })

      if (fetchResponse.ok) {
        const data = await fetchResponse.json()
        
        if (data.output_image_base64) {
          setSnapshots(prev => [data.output_image_base64, ...prev].slice(0, 10))
        } else {
          setSnapshots(prev => [screenshot, ...prev].slice(0, 10))
        }
        
        setError(null)
      }
    } catch (err) {
      console.error('Snapshot error:', err)
      setError('Failed to capture snapshot')
    }
  }, [getScreenshot, model, confidence, isCameraReady])

  const availableModels = models.length > 0 ? models : [
    { id: 'yolov8m', name: 'YOLOv8m', family: 'YOLOv8', best_for: 'Balanced webcam accuracy' },
    { id: 'yolov8s', name: 'YOLOv8s', family: 'YOLOv8', best_for: 'Efficient webcam detection' },
    { id: 'yolov8l', name: 'YOLOv8l', family: 'YOLOv8', best_for: 'Higher accuracy' },
    { id: 'yolov8x', name: 'YOLOv8x', family: 'YOLOv8', best_for: 'Maximum accuracy' },
    { id: 'yolov8s-world', name: 'YOLOv8s-World', family: 'YOLO-World', best_for: 'Broad objects' },
    { id: 'yolov8n', name: 'YOLOv8n', family: 'YOLOv8', best_for: 'Fastest' },
  ]

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
              <Link href="/detect" className="text-gray-300 hover:text-primary transition-colors hidden md:block">
                Detection
              </Link>
              <Link href="/webcam" className="text-primary font-semibold transition-colors hidden md:block">
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
      <div className="flex-1 flex items-center justify-center pt-20 pb-8 px-4">
        <div className="max-w-6xl mx-auto w-full">
          {/* Header */}
          <motion.div 
            className="text-center mb-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-3xl md:text-4xl font-bold mb-2">
              <span className="gradient-text">Real-Time Webcam Detection</span>
            </h1>
            <div className="flex items-center justify-center gap-2">
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

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Webcam Feed */}
            <div className="lg:col-span-2">
              <div className="card p-4">
                <div className="relative">
                  <video
                    ref={videoRef}
                    onLoadedMetadata={updateVideoSize}
                    className="aspect-[4/3] w-full rounded-lg bg-black object-contain"
                  />
                  <canvas ref={canvasRef} className="hidden" />
                  
                  {/* Detection overlay */}
                  {detections.length > 0 && (
                    <svg
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      viewBox={`0 0 ${frameSize.width} ${frameSize.height}`}
                      preserveAspectRatio="none"
                    >
                      {detections.map((det, idx) => (
                        <g key={idx}>
                          <rect
                            x={det.bbox.x1}
                            y={det.bbox.y1}
                            width={det.bbox.x2 - det.bbox.x1}
                            height={det.bbox.y2 - det.bbox.y1}
                            stroke="#00d4ff"
                            strokeWidth={Math.max(frameSize.width, frameSize.height) * 0.004}
                            fill="none"
                          />
                          <rect
                            x={det.bbox.x1}
                            y={Math.max(0, det.bbox.y1 - 18)}
                            width={Math.min(frameSize.width - det.bbox.x1, Math.max(90, det.class_name.length * 9 + 44))}
                            height="18"
                            fill="#00d4ff"
                            opacity="0.9"
                          />
                          <text
                            x={det.bbox.x1 + 5}
                            y={Math.max(13, det.bbox.y1 - 5)}
                            fill="#031018"
                            fontSize={Math.max(frameSize.width, frameSize.height) * 0.018}
                            fontWeight="bold"
                          >
                            {det.class_name} {(det.confidence * 100).toFixed(0)}%
                          </text>
                        </g>
                      ))}
                    </svg>
                  )}

                  {/* FPS Counter */}
                  {isStreaming && (
                    <div className="absolute top-2 left-2 glass px-2 py-1 rounded-lg">
                      <span className="text-accent-green font-mono text-xs">{fps} FPS</span>
                    </div>
                  )}

                  {/* Status indicator */}
                  <div className="absolute top-2 right-2 flex items-center space-x-2">
                    {isStreaming ? (
                      <>
                        <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                        <span className="text-xs text-red-400">LIVE</span>
                      </>
                    ) : (
                      <span className={`text-xs ${isCameraReady ? 'text-accent-green' : 'text-yellow-400'}`}>
                        {isCameraReady ? 'READY' : 'WAITING'}
                      </span>
                    )}
                  </div>
                </div>
                <div className="mt-2 text-center text-xs text-gray-400">
                  {cameraMessage}
                </div>

                {/* Controls */}
                <div className="mt-3 flex items-center justify-center space-x-3 flex-wrap gap-2">
                  {!isStreaming ? (
                    <button
                      onClick={startStream}
                      className="btn-primary flex items-center text-sm py-2 px-4"
                    >
                      <FaCamera className="mr-2" />
                      {isCameraReady ? 'Start Detection' : 'Start Webcam'}
                    </button>
                  ) : (
                    <button
                      onClick={stopStream}
                      className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg flex items-center transition-colors text-sm"
                    >
                      <FaStop className="mr-2" />
                      Stop
                    </button>
                  )}
                  
                  <button
                    onClick={captureSnapshot}
                    className="btn-secondary flex items-center text-sm py-2 px-4"
                    disabled={!isStreaming}
                  >
                    <FaImage className="mr-2" />
                    Snapshot
                  </button>
                  
                  {snapshots.length > 0 && (
                    <button
                      onClick={() => setShowSnapshots(!showSnapshots)}
                      className="px-3 py-2 glass text-white rounded-lg flex items-center transition-colors hover:border-primary/50 text-sm"
                    >
                      <FaDownload className="mr-1" />
                      ({snapshots.length})
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Sidebar */}
            <div className="space-y-4">
              {/* Settings */}
              <div className="card p-4">
                <h3 className="text-base font-semibold mb-3">Settings</h3>
                
                <div className="space-y-3">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Model</label>
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="input text-sm py-2"
                      disabled={isStreaming}
                    >
                      {availableModels.map((item: any) => (
                        <option key={item.id} value={item.id}>
                          {item.name} ({item.best_for || item.family || item.type})
                        </option>
                      ))}
                    </select>
                    <div className="mt-1 text-xs text-gray-500">
                      Use YOLOv8m or YOLOv8s-World when non-person objects are missed.
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Confidence: <span className="text-primary font-semibold">{(confidence * 100).toFixed(0)}%</span>
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.05"
                      value={confidence}
                      onChange={(e) => setConfidence(parseFloat(e.target.value))}
                      className="w-full accent-primary"
                      disabled={isStreaming}
                    />
                  </div>
                </div>
              </div>

              {/* Object Counter */}
              <div className="card p-4">
                <h3 className="text-base font-semibold mb-3">Objects</h3>
                
                <div className="text-center mb-3">
                  <div className="text-3xl font-bold text-primary">{totalObjects}</div>
                  <div className="text-xs text-gray-400">Total Detected</div>
                </div>

                {Object.keys(objectCounts).length > 0 ? (
                  <div className="space-y-1 max-h-32 overflow-y-auto">
                    {Object.entries(objectCounts).map(([className, count]) => (
                      <div key={className} className="flex items-center justify-between glass rounded-lg p-2">
                        <span className="badge badge-primary text-xs">{className}</span>
                        <span className="font-bold text-sm">{count}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-4 text-sm">
                    No objects detected
                  </div>
                )}
              </div>

              {/* Stats */}
              <div className="card p-4">
                <h3 className="text-base font-semibold mb-3">Performance</h3>
                
                <div className="grid grid-cols-2 gap-2">
                  <div className="glass rounded-lg p-2 text-center">
                    <div className="text-lg font-bold text-accent-green">{fps}</div>
                    <div className="text-xs text-gray-400">FPS</div>
                  </div>
                  <div className="glass rounded-lg p-2 text-center">
                    <div className="text-lg font-bold text-secondary">{detections.length}</div>
                    <div className="text-xs text-gray-400">Detections</div>
                  </div>
                  <div className="glass rounded-lg p-2 text-center">
                    <div className="text-lg font-bold text-primary">{snapshots.length}</div>
                    <div className="text-xs text-gray-400">Snaps</div>
                  </div>
                  <div className="glass rounded-lg p-2 text-center">
                    <div className="text-lg font-bold text-accent-pink">{totalObjects}</div>
                    <div className="text-xs text-gray-400">Objects</div>
                  </div>
                  <div className="glass rounded-lg p-2 text-center">
                    <div className="text-lg font-bold text-primary">{movementCounts.up}</div>
                    <div className="text-xs text-gray-400">Up</div>
                  </div>
                  <div className="glass rounded-lg p-2 text-center">
                    <div className="text-lg font-bold text-secondary">{movementCounts.down}</div>
                    <div className="text-xs text-gray-400">Down</div>
                  </div>
                  <div className="glass rounded-lg p-2 text-center col-span-2">
                    <div className="text-lg font-bold text-accent-green">{totalMovements}</div>
                    <div className="text-xs text-gray-400">Total Movement</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Snapshots Gallery Modal */}
      {showSnapshots && snapshots.length > 0 && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={() => setShowSnapshots(false)}>
          <div className="card max-w-4xl w-full mx-4 max-h-[80vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold">Snapshots ({snapshots.length})</h2>
              <button 
                onClick={() => setShowSnapshots(false)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {snapshots.map((snapshot, idx) => (
                <div key={idx} className="relative group">
                  <img 
                    src={snapshot} 
                    alt={`Snapshot ${idx + 1}`}
                    className="w-full rounded-lg border border-glass-border"
                  />
                  <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded-lg">
                    <a 
                      href={snapshot} 
                      download={`snapshot_${snapshots.length - idx}.jpg`}
                      className="btn-primary text-xs"
                    >
                      <FaDownload className="mr-1 inline" />
                      Download
                    </a>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-4 flex justify-end">
              <button
                onClick={() => setSnapshots([])}
                className="px-4 py-2 text-red-400 hover:text-red-300 transition-colors text-sm"
              >
                Clear All
              </button>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="fixed bottom-12 left-1/2 transform -translate-x-1/2 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm z-50">
          {error}
        </div>
      )}

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
