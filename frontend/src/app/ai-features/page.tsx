'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  FaBrain, 
  FaHistory, 
  FaProjectDiagram, 
  FaChartLine, 
  FaLightbulb,
  FaSync,
  FaMicrochip,
  FaNetworkWired,
  FaMagic,
  FaCommentDots,
  FaCheckCircle,
  FaExclamationTriangle,
  FaInfoCircle,
  FaChevronDown,
  FaChevronUp
} from 'react-icons/fa'
import Link from 'next/link'
import { FaEye } from 'react-icons/fa'

const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL || ''
const API_URL = /^https?:\/\/(localhost|127\.0\.0\.1):8000\/?$/.test(configuredApiUrl)
  ? ''
  : configuredApiUrl.replace(/\/$/, '')

type TabType = 'adaptive' | 'recommender' | 'timeline' | 'graph' | 'predictions' | 'explanations' | 'full-analysis'

export default function AIFeaturesPage() {
  const [activeTab, setActiveTab] = useState<TabType>('full-analysis')
  const [loading, setLoading] = useState(false)
  const [fullAnalysis, setFullAnalysis] = useState<any>(null)
  const [adaptiveStatus, setAdaptiveStatus] = useState<any>(null)
  const [sceneStats, setSceneStats] = useState<any>(null)
  const [graphOverview, setGraphOverview] = useState<any>(null)
  const [predictions, setPredictions] = useState<any>(null)
  const [explanations, setExplanations] = useState<any>(null)
  const [recommendation, setRecommendation] = useState<any>(null)
  const [hardwareResources, setHardwareResources] = useState<any>(null)
  const [selectedSceneType, setSelectedSceneType] = useState<string>('auto')
  const [selectedPriority, setSelectedPriority] = useState<string>('auto')
  const [apiStatus, setApiStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking')
  const [error, setError] = useState<string | null>(null)

  const fetchJson = async (url: string) => {
    const response = await fetch(url)

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`)
    }

    setApiStatus('connected')
    setError(null)
    return response.json()
  }

  const fetchRecommendation = async (sceneType?: string, priority?: string) => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (sceneType && sceneType !== 'auto') params.append('scene_type', sceneType)
      if (priority && priority !== 'auto') params.append('priority', priority)
      const url = `${API_URL}/api/ai/recommend/model?${params.toString()}`
      const data = await fetchJson(url)
      setRecommendation(data)
    } catch (err: any) {
      console.error('Error fetching recommendation:', err)
      setApiStatus('disconnected')
      setError(err.message || 'Could not connect to backend')
    } finally {
      setLoading(false)
    }
  }

  const fetchHardwareResources = async () => {
    try {
      const data = await fetchJson(`${API_URL}/api/ai/hardware/resources`)
      setHardwareResources(data)
    } catch (err: any) {
      console.error('Error fetching hardware resources:', err)
      setApiStatus('disconnected')
      setError(err.message || 'Could not connect to backend')
    }
  }

  const fetchData = async () => {
    setLoading(true)
    try {
      switch (activeTab) {
        case 'full-analysis':
          const fullData = await fetchJson(`${API_URL}/api/ai/analysis/full`)
          setFullAnalysis(fullData)
          break
        case 'adaptive':
          const adaptiveData = await fetchJson(`${API_URL}/api/ai/adaptive/status`)
          setAdaptiveStatus(adaptiveData)
          break
        case 'recommender':
          await fetchRecommendation(selectedSceneType !== 'auto' ? selectedSceneType : undefined, selectedPriority !== 'auto' ? selectedPriority : undefined)
          await fetchHardwareResources()
          break
        case 'timeline':
          const sceneData = await fetchJson(`${API_URL}/api/ai/scene/statistics`)
          setSceneStats(sceneData)
          break
        case 'graph':
          const graphData = await fetchJson(`${API_URL}/api/ai/graph/overview`)
          setGraphOverview(graphData)
          break
        case 'predictions':
          const predData = await fetchJson(`${API_URL}/api/ai/predictions/summary`)
          setPredictions(predData)
          break
        case 'explanations':
          const explData = await fetchJson(`${API_URL}/api/ai/explanations/history?limit=10`)
          setExplanations(explData)
          break
      }
    } catch (err: any) {
      console.error('Error fetching data:', err)
      setApiStatus('disconnected')
      setError(err.message || 'Could not connect to backend')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [activeTab])

  const handleRefresh = () => {
    fetchData()
  }

  const tabs = [
    { id: 'full-analysis', label: 'Full', icon: FaBrain },
    { id: 'adaptive', label: 'Adaptive', icon: FaMicrochip },
    { id: 'recommender', label: 'Recommender', icon: FaMagic },
    { id: 'timeline', label: 'Timeline', icon: FaHistory },
    { id: 'graph', label: 'Graph', icon: FaProjectDiagram },
    { id: 'predictions', label: 'Predict', icon: FaChartLine },
    { id: 'explanations', label: 'Explain', icon: FaLightbulb },
  ]

  return (
    <div className="h-dvh flex flex-col overflow-hidden">
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
              <Link href="/admin" className="text-gray-300 hover:text-primary transition-colors hidden lg:block">
                Admin
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content - Compact Layout */}
      <div className="flex-1 flex flex-col pt-[4.5rem] pb-2 px-3 overflow-hidden">
        <div className="max-w-7xl mx-auto w-full flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <motion.div 
            className="text-center mb-2"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-xl md:text-2xl font-bold mb-1">
              <span className="gradient-text">AI Features</span>
            </h1>
            <div className="flex items-center justify-center gap-2">
              <div className={`h-2 w-2 rounded-full ${
                apiStatus === 'connected' ? 'bg-accent-green animate-pulse' :
                apiStatus === 'disconnected' ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'
              }`} />
              <span className={`text-xs ${
                apiStatus === 'connected' ? 'text-accent-green' :
                apiStatus === 'disconnected' ? 'text-red-400' : 'text-yellow-400'
              }`}>
                {apiStatus === 'connected' ? 'Backend Connected' :
                 apiStatus === 'disconnected' ? 'Backend Disconnected' : 'Checking backend...'}
              </span>
            </div>
          </motion.div>

          {/* Tabs */}
          <div className="flex flex-wrap gap-1 mb-2 justify-center">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as TabType)}
                className={`px-2.5 py-1.5 rounded-lg flex items-center gap-1 transition-all text-xs ${
                  activeTab === tab.id
                    ? 'bg-primary text-white'
                    : 'glass text-gray-400 hover:text-white hover:bg-white/10'
                }`}
              >
                <tab.icon className="text-sm" />
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            ))}
            <button
              onClick={handleRefresh}
              className="px-2.5 py-1.5 rounded-lg glass text-gray-400 hover:text-white hover:bg-white/10 transition-all text-xs"
              title="Refresh"
            >
              <FaSync className={loading ? 'animate-spin' : ''} />
            </button>
          </div>

          {error && (
            <div className="mb-2 flex items-center justify-center gap-2 rounded-lg border border-red-500/40 bg-red-500/15 px-3 py-2 text-xs text-red-300">
              <FaExclamationTriangle />
              <span>{error}. Make sure the FastAPI backend is running on port 8000.</span>
            </div>
          )}

          {/* Content - Scrollable */}
          <div className="flex-1 overflow-auto">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className="h-full"
              >
                {loading ? (
                  <div className="card flex items-center justify-center py-8">
                    <FaSync className="text-2xl text-primary animate-spin" />
                    <p className="ml-3 text-gray-400 text-sm">Loading...</p>
                  </div>
                ) : (
                  <>
                    {activeTab === 'full-analysis' && <FullAnalysisView data={fullAnalysis} />}
                    {activeTab === 'adaptive' && <AdaptiveAIView data={adaptiveStatus} />}
                    {activeTab === 'recommender' && <ModelRecommenderView recommendation={recommendation} hardware={hardwareResources} />}
                    {activeTab === 'timeline' && <SceneMemoryView data={sceneStats} />}
                    {activeTab === 'graph' && <ObjectGraphView data={graphOverview} />}
                    {activeTab === 'predictions' && <PredictionsView data={predictions} />}
                    {activeTab === 'explanations' && <ExplanationsView data={explanations} />}
                  </>
                )}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Footer with Copyright */}
      <footer className="glass border-t border-glass-border py-1.5">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-gray-500 text-xs">
            © 2026 LenScope AI. Developed by Archisman Das. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  )
}

// Full Analysis View
function FullAnalysisView({ data }: { data: any }) {
  if (!data) {
    return (
      <div className="card flex items-center justify-center py-8">
        <p className="text-gray-400 text-sm">No data available. Run some detections first.</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
        <StatCard icon={FaMicrochip} label="Model" value={data.adaptive_ai?.current_model || 'None'} color="text-primary" />
        <StatCard icon={FaHistory} label="Frames" value={data.scene_memory?.statistics?.total_frames || 0} color="text-accent-green" />
        <StatCard icon={FaNetworkWired} label="Tracked" value={data.scene_memory?.statistics?.total_objects_tracked || 0} color="text-secondary" />
        <StatCard icon={FaChartLine} label="Edges" value={data.object_graph?.statistics?.total_edges || 0} color="text-accent-purple" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        <div className="card p-3">
          <h3 className="text-sm font-semibold mb-2 flex items-center">
            <FaMicrochip className="mr-2 text-primary" />
            Adaptive AI
          </h3>
          {data.adaptive_ai?.performance_stats && Object.keys(data.adaptive_ai.performance_stats).length > 0 ? (
            <div className="space-y-2">
              {Object.entries(data.adaptive_ai.performance_stats).map(([modelId, stats]: [string, any]) => (
                <div key={modelId} className="glass rounded-lg p-2">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm font-medium">{stats.model_name}</span>
                    <span className="text-xs text-gray-400">{stats.total_detections} det</span>
                  </div>
                  <div className="flex gap-3 text-xs">
                    <span className="text-gray-400">Avg: {stats.avg_inference_time_ms?.toFixed(1)}ms</span>
                    <span className="text-gray-400">Success: {(stats.success_rate * 100)?.toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-400 text-xs">No model performance data yet</p>
          )}
        </div>

        <div className="card p-3">
          <h3 className="text-sm font-semibold mb-2 flex items-center">
            <FaHistory className="mr-2 text-accent-green" />
            Scene Memory
          </h3>
          {data.scene_memory?.statistics ? (
            <div className="space-y-2">
              <div className="glass rounded-lg p-2">
                <div className="text-xs text-gray-400">Top Classes</div>
                <div className="flex flex-wrap gap-1 mt-1">
                  {Object.entries(data.scene_memory.statistics.class_frequency || {})
                    .sort((a, b) => (b[1] as number) - (a[1] as number))
                    .slice(0, 6)
                    .map(([className, count]) => (
                      <span key={className} className="badge badge-primary text-xs">
                        {className}: {count as number}
                      </span>
                    ))}
                </div>
              </div>
              <div className="glass rounded-lg p-2">
                <div className="text-xs text-gray-400">Sessions</div>
                <div className="text-lg font-semibold mt-1">
                  {data.scene_memory.statistics.session_count || 0}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-gray-400 text-xs">No scene data yet</p>
          )}
        </div>

        <div className="card p-3">
          <h3 className="text-sm font-semibold mb-2 flex items-center">
            <FaProjectDiagram className="mr-2 text-secondary" />
            Object Graph
          </h3>
          {data.object_graph?.statistics ? (
            <div className="grid grid-cols-2 gap-2">
              <div className="glass rounded-lg p-2 text-center">
                <div className="text-xl font-bold text-primary">{data.object_graph.statistics.total_nodes || 0}</div>
                <div className="text-xs text-gray-400">Nodes</div>
              </div>
              <div className="glass rounded-lg p-2 text-center">
                <div className="text-xl font-bold text-secondary">{data.object_graph.statistics.total_edges || 0}</div>
                <div className="text-xs text-gray-400">Edges</div>
              </div>
            </div>
          ) : (
            <p className="text-gray-400 text-xs">No relationship data yet</p>
          )}
        </div>

        <div className="card p-3">
          <h3 className="text-sm font-semibold mb-2 flex items-center">
            <FaChartLine className="mr-2 text-accent-purple" />
            Predictions
          </h3>
          {data.predictions?.next_frame?.complexity_prediction ? (
            <div className="glass rounded-lg p-2 text-center">
              <div className="text-lg font-semibold capitalize">
                {data.predictions.next_frame.complexity_prediction.predicted_complexity || 'unknown'}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Confidence: {((data.predictions.next_frame.complexity_prediction.confidence || 0) * 100).toFixed(0)}%
              </div>
            </div>
          ) : (
            <p className="text-gray-400 text-xs">No predictions available yet</p>
          )}
        </div>
      </div>
    </div>
  )
}

// Adaptive AI View
function AdaptiveAIView({ data }: { data: any }) {
  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <FaMicrochip className="mr-2 text-primary" />
        Adaptive AI Model Switching
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass rounded-lg p-4">
          <h3 className="text-xs text-gray-400 mb-2">Current Status</h3>
          <div className="space-y-2">
            <div>
              <span className="text-gray-500 text-xs">Model:</span>
              <div className="text-base font-semibold mt-1">{data?.current_model || 'Not set'}</div>
            </div>
            <div>
              <span className="text-gray-500 text-xs">Switches:</span>
              <div className="text-base font-semibold mt-1">{data?.total_switches || 0}</div>
            </div>
            <div>
              <span className="text-gray-500 text-xs">Analyses:</span>
              <div className="text-base font-semibold mt-1">{data?.scene_analyses || 0}</div>
            </div>
          </div>
        </div>

        <div className="glass rounded-lg p-4">
          <h3 className="text-xs text-gray-400 mb-2">Recent Switches</h3>
          <p className="text-gray-400 text-xs">Switch history will appear here</p>
        </div>
      </div>

      <div className="mt-4 p-3 glass rounded-lg">
        <h3 className="text-xs text-gray-400 mb-1">How It Works</h3>
        <p className="text-xs text-gray-300">
          The Adaptive AI system analyzes scene complexity in real-time and automatically selects the optimal model.
        </p>
      </div>
    </div>
  )
}

// Scene Memory View
function SceneMemoryView({ data }: { data: any }) {
  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <FaHistory className="mr-2 text-accent-green" />
        Scene Memory Timeline
      </h2>

      {data ? (
        <div className="grid grid-cols-3 gap-3 mb-4">
          <StatCard icon={FaHistory} label="Frames" value={data.total_frames || 0} color="text-primary" />
          <StatCard icon={FaProjectDiagram} label="Tracked" value={data.total_objects_tracked || 0} color="text-secondary" />
          <StatCard icon={FaNetworkWired} label="Sessions" value={data.session_count || 0} color="text-accent-green" />
        </div>
      ) : (
        <p className="text-gray-400 text-sm">No scene memory data available</p>
      )}

      <div className="mt-4 p-3 glass rounded-lg">
        <h3 className="text-xs text-gray-400 mb-1">How It Works</h3>
        <p className="text-xs text-gray-300">
          The Scene Memory Timeline maintains a temporal history of all detected objects, tracking their movements across frames.
        </p>
      </div>
    </div>
  )
}

// Object Graph View
function ObjectGraphView({ data }: { data: any }) {
  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <FaProjectDiagram className="mr-2 text-secondary" />
        Object Relationship Graph
      </h2>

      {data ? (
        <div className="grid grid-cols-3 gap-3 mb-4">
          <StatCard icon={FaNetworkWired} label="Nodes" value={data.total_nodes || 0} color="text-primary" />
          <StatCard icon={FaProjectDiagram} label="Edges" value={data.total_edges || 0} color="text-secondary" />
          <StatCard icon={FaChartLine} label="Density" value={(data.graph_density || 0).toFixed(3)} color="text-accent-green" />
        </div>
      ) : (
        <p className="text-gray-400 text-sm">No graph data available</p>
      )}

      <div className="mt-4 p-3 glass rounded-lg">
        <h3 className="text-xs text-gray-400 mb-1">How It Works</h3>
        <p className="text-xs text-gray-300">
          The Object Relationship Graph analyzes spatial proximity and co-occurrence patterns between detected objects.
        </p>
      </div>
    </div>
  )
}

// Predictions View
function PredictionsView({ data }: { data: any }) {
  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <FaChartLine className="mr-2 text-accent-purple" />
        Prediction Engine
      </h2>

      {data?.next_frame?.complexity_prediction ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="glass rounded-lg p-4 text-center">
            <h3 className="text-xs text-gray-400 mb-2">Scene Complexity</h3>
            <div className="text-2xl font-bold capitalize mb-2">
              {data.next_frame.complexity_prediction.predicted_complexity || 'unknown'}
            </div>
            <span className="badge badge-primary">
              Confidence: {((data.next_frame.complexity_prediction.confidence || 0) * 100).toFixed(0)}%
            </span>
          </div>

          <div className="glass rounded-lg p-4">
            <h3 className="text-xs text-gray-400 mb-2">Class Predictions</h3>
            {data.next_frame.class_predictions && data.next_frame.class_predictions.length > 0 ? (
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {data.next_frame.class_predictions.slice(0, 5).map((pred: any, idx: number) => (
                  <div key={idx} className="flex items-center justify-between text-xs">
                    <span>{pred.class_name}</span>
                    <span className="badge badge-secondary text-xs">{((pred.confidence || 0) * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-400 text-xs">No class predictions available</p>
            )}
          </div>
        </div>
      ) : (
        <p className="text-gray-400 text-sm">No prediction data available</p>
      )}

      <div className="mt-4 p-3 glass rounded-lg">
        <h3 className="text-xs text-gray-400 mb-1">How It Works</h3>
        <p className="text-xs text-gray-300">
          The Prediction Engine uses trajectory analysis and scene history to forecast future object positions.
        </p>
      </div>
    </div>
  )
}

// Explanations View
function ExplanationsView({ data }: { data: any }) {
  const [expandedId, setExpandedId] = useState<string | null>(null)

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <FaLightbulb className="mr-2 text-yellow-400" />
        AI Decision Explanation
      </h2>

      {data?.explanations && data.explanations.length > 0 ? (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {data.explanations.map((item: any, idx: number) => (
            <div key={idx} className="glass rounded-lg overflow-hidden">
              <button
                onClick={() => setExpandedId(expandedId === item.type ? null : item.type)}
                className="w-full p-3 flex items-center justify-between text-left hover:bg-white/5 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <span className="badge badge-primary text-xs capitalize">{item.type}</span>
                  <span className="text-xs text-gray-400">{new Date(item.timestamp).toLocaleTimeString()}</span>
                </div>
                {expandedId === item.type ? <FaChevronUp /> : <FaChevronDown />}
              </button>
              
              {expandedId === item.type && (
                <div className="p-3 border-t border-white/10">
                  <pre className="text-xs text-gray-300 whitespace-pre-wrap">
                    {JSON.stringify(item.explanation, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8">
          <FaCommentDots className="text-2xl text-gray-600 mx-auto mb-2" />
          <p className="text-gray-400 text-sm">No explanations available yet</p>
        </div>
      )}

      <div className="mt-4 p-3 glass rounded-lg">
        <h3 className="text-xs text-gray-400 mb-1">How It Works</h3>
        <p className="text-xs text-gray-300">
          The Explanation Engine provides human-readable explanations for AI decisions, including confidence factors and model selection reasoning.
        </p>
      </div>
    </div>
  )
}

// Model Recommender View
function ModelRecommenderView({ recommendation, hardware }: { recommendation: any; hardware: any }) {
  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <FaMagic className="mr-2 text-accent-pink" />
        Model Recommender
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass rounded-lg p-4">
          <h3 className="text-xs text-gray-400 mb-2">Recommendation</h3>
          {recommendation?.recommended_model ? (
            <div>
              <div className="text-xl font-bold text-primary">{recommendation.recommended_model}</div>
              <div className="text-xs text-gray-400 mt-1">
                {recommendation.reason || 'Based on scene analysis'}
              </div>
            </div>
          ) : (
            <p className="text-gray-400 text-sm">Set scene parameters to get a recommendation</p>
          )}
        </div>

        <div className="glass rounded-lg p-4">
          <h3 className="text-xs text-gray-400 mb-2">Hardware Resources</h3>
          {hardware ? (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">GPU:</span>
                <span>{hardware.gpu_available ? 'Available' : 'Not Available'}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Memory:</span>
                <span>{hardware.memory_gb ? `${hardware.memory_gb}GB` : 'N/A'}</span>
              </div>
            </div>
          ) : (
            <p className="text-gray-400 text-sm">No hardware data available</p>
          )}
        </div>
      </div>

      <div className="mt-4 p-3 glass rounded-lg">
        <h3 className="text-xs text-gray-400 mb-1">How It Works</h3>
        <p className="text-xs text-gray-300">
          The Model Recommender analyzes scene type, priority, and hardware resources to suggest the optimal model for your detection task.
        </p>
      </div>
    </div>
  )
}

// Stat Card Component
function StatCard({ icon, label, value, color }: { icon: any; label: string; value: string | number; color: string }) {
  const Icon = icon
  return (
    <div className="glass rounded-lg p-3 text-center">
      <Icon className={`text-xl mx-auto mb-1 ${color}`} />
      <div className="text-xs text-gray-400">{label}</div>
      <div className="text-lg font-bold">{value}</div>
    </div>
  )
}
