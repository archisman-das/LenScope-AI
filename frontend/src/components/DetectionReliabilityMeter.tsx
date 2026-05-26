'use client'

import { useState } from 'react'
import { 
  FaCheckCircle, 
  FaExclamationTriangle, 
  FaTimesCircle,
  FaInfoCircle,
  FaBolt,
  FaEye,
  FaSun,
  FaExpandArrowsAlt,
  FaCrosshairs
} from 'react-icons/fa'

interface DetectionReliabilityProps {
  detection: {
    class_name: string
    confidence: number
    bbox?: { x1: number; y1: number; x2: number; y2: number }
    image_dimensions?: { width: number; height: number }
  }
  showDetails?: boolean
}

interface ReliabilityMetrics {
  confidence: number
  stability: number
  uncertainty: number
  reliability: 'High' | 'Medium' | 'Low' | 'Very Low'
  issues: string[]
  recommendations: string[]
}

export default function DetectionReliabilityMeter({ detection, showDetails = true }: DetectionReliabilityProps) {
  const [expanded, setExpanded] = useState(false)

  // Calculate reliability metrics
  const calculateMetrics = (): ReliabilityMetrics => {
    const { confidence, bbox, image_dimensions } = detection
    const issues: string[] = []
    const recommendations: string[] = []

    // Calculate object size relative to image
    let relativeSize = 0
    let sizeCategory = 'unknown'
    if (bbox && image_dimensions) {
      const objectArea = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
      const imageArea = image_dimensions.width * image_dimensions.height
      relativeSize = objectArea / imageArea
      sizeCategory = relativeSize > 0.1 ? 'large' : relativeSize > 0.02 ? 'medium' : 'small'
    }

    // Stability score based on various factors
    let stability = 0.8 // Base stability
    let uncertainty = 0.2 // Base uncertainty

    // Factor 1: Confidence impact
    if (confidence < 0.5) {
      stability -= 0.3
      uncertainty += 0.2
      issues.push('Low confidence detection')
      recommendations.push('Consider manual verification')
    } else if (confidence < 0.7) {
      stability -= 0.15
      uncertainty += 0.1
      issues.push('Moderate confidence - may require verification')
    }

    // Factor 2: Object size impact
    if (sizeCategory === 'small') {
      stability -= 0.2
      uncertainty += 0.15
      issues.push('Small object detected - higher uncertainty')
      recommendations.push('Consider using higher resolution or zoom')
    } else if (sizeCategory === 'large') {
      stability += 0.1
      uncertainty -= 0.05
    }

    // Factor 3: Position impact
    if (bbox && image_dimensions) {
      const centerX = (bbox.x1 + bbox.x2) / 2 / image_dimensions.width
      const centerY = (bbox.y1 + bbox.y2) / 2 / image_dimensions.height
      
      const nearEdge = centerX < 0.15 || centerX > 0.85 || centerY < 0.15 || centerY > 0.85
      if (nearEdge) {
        stability -= 0.15
        uncertainty += 0.1
        issues.push('Object near image edge - partial occlusion possible')
      }
    }

    // Clamp values
    stability = Math.max(0.1, Math.min(1, stability))
    uncertainty = Math.max(0.05, Math.min(0.95, uncertainty))

    // Determine overall reliability
    let reliability: ReliabilityMetrics['reliability'] = 'High'
    if (confidence < 0.5 || stability < 0.5) {
      reliability = 'Very Low'
    } else if (confidence < 0.7 || stability < 0.65) {
      reliability = 'Low'
    } else if (confidence < 0.85 || stability < 0.75) {
      reliability = 'Medium'
    }

    return {
      confidence,
      stability,
      uncertainty,
      reliability,
      issues,
      recommendations
    }
  }

  const metrics = calculateMetrics()

  const getReliabilityColor = (level: string) => {
    switch (level) {
      case 'High': return 'text-green-400'
      case 'Medium': return 'text-yellow-400'
      case 'Low': return 'text-orange-400'
      case 'Very Low': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  const getReliabilityBg = (level: string) => {
    switch (level) {
      case 'High': return 'bg-green-500/20 border-green-500/40'
      case 'Medium': return 'bg-yellow-500/20 border-yellow-500/40'
      case 'Low': return 'bg-orange-500/20 border-orange-500/40'
      case 'Very Low': return 'bg-red-500/20 border-red-500/40'
      default: return 'bg-gray-500/20 border-gray-500/40'
    }
  }

  const getReliabilityIcon = (level: string) => {
    switch (level) {
      case 'High': return FaCheckCircle
      case 'Medium': return FaInfoCircle
      case 'Low': return FaExclamationTriangle
      case 'Very Low': return FaTimesCircle
      default: return FaInfoCircle
    }
  }

  const ReliabilityIcon = getReliabilityIcon(metrics.reliability)

  // Gauge component for metrics
  const MetricGauge = ({ 
    value, 
    label, 
    color, 
    icon: Icon 
  }: { 
    value: number
    label: string
    color: string
    icon: any
  }) => (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className={`text-sm ${color}`} />
          <span className="text-sm text-gray-400">{label}</span>
        </div>
        <span className={`text-sm font-semibold ${color}`}>
          {(value * 100).toFixed(0)}%
        </span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div 
          className={`h-full rounded-full transition-all duration-500 ${color.replace('text-', 'bg-')}`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  )

  return (
    <div className={`glass rounded-lg overflow-hidden ${expanded ? 'ring-1 ring-primary/30' : ''}`}>
      {/* Header */}
      <div 
        className="p-4 cursor-pointer hover:bg-white/5 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Object Class */}
            <div className="flex items-center gap-2">
              <span className="text-lg font-semibold capitalize">{detection.class_name}</span>
            </div>
            
            {/* Confidence Badge */}
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              metrics.confidence > 0.8 ? 'bg-green-500/20 text-green-400' :
              metrics.confidence > 0.6 ? 'bg-yellow-500/20 text-yellow-400' :
              'bg-red-500/20 text-red-400'
            }`}>
              {(metrics.confidence * 100).toFixed(1)}% confidence
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Reliability Badge */}
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium ${getReliabilityBg(metrics.reliability)}`}>
              <ReliabilityIcon className={getReliabilityColor(metrics.reliability)} />
              <span className={getReliabilityColor(metrics.reliability)}>
                {metrics.reliability} Reliability
              </span>
            </div>

            {/* Expand/Collapse */}
            <div className={`transform transition-transform ${expanded ? 'rotate-180' : ''}`}>
              <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Expanded Details */}
      {expanded && showDetails && (
        <div className="px-4 pb-4 space-y-4 border-t border-white/10 pt-4">
          {/* Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="glass rounded-lg p-3">
              <MetricGauge 
                value={metrics.confidence} 
                label="Confidence" 
                color={
                  metrics.confidence > 0.8 ? 'text-green-400' :
                  metrics.confidence > 0.6 ? 'text-yellow-400' :
                  'text-red-400'
                }
                icon={FaCrosshairs}
              />
            </div>
            <div className="glass rounded-lg p-3">
              <MetricGauge 
                value={metrics.stability} 
                label="Stability" 
                color={
                  metrics.stability > 0.75 ? 'text-green-400' :
                  metrics.stability > 0.6 ? 'text-yellow-400' :
                  'text-red-400'
                }
                icon={FaBolt}
              />
            </div>
            <div className="glass rounded-lg p-3">
              <MetricGauge 
                value={1 - metrics.uncertainty} 
                label="Certainty" 
                color={
                  metrics.uncertainty < 0.25 ? 'text-green-400' :
                  metrics.uncertainty < 0.4 ? 'text-yellow-400' :
                  'text-red-400'
                }
                icon={FaEye}
              />
            </div>
          </div>

          {/* Issues */}
          {metrics.issues.length > 0 && (
            <div className="glass rounded-lg p-3 bg-red-500/10 border border-red-500/20">
              <div className="flex items-center gap-2 mb-2">
                <FaExclamationTriangle className="text-red-400" />
                <span className="text-sm font-medium text-red-400">Possible Issues</span>
              </div>
              <ul className="space-y-1">
                {metrics.issues.map((issue, idx) => (
                  <li key={idx} className="text-sm text-red-300 flex items-start gap-2">
                    <span className="text-red-400 mt-1">•</span>
                    {issue}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Recommendations */}
          {metrics.recommendations.length > 0 && (
            <div className="glass rounded-lg p-3 bg-blue-500/10 border border-blue-500/20">
              <div className="flex items-center gap-2 mb-2">
                <FaInfoCircle className="text-blue-400" />
                <span className="text-sm font-medium text-blue-400">Recommendations</span>
              </div>
              <ul className="space-y-1">
                {metrics.recommendations.map((rec, idx) => (
                  <li key={idx} className="text-sm text-blue-300 flex items-start gap-2">
                    <span className="text-blue-400 mt-1">•</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Technical Details */}
          <div className="grid grid-cols-2 gap-4">
            <div className="glass rounded-lg p-3">
              <div className="text-xs text-gray-500 mb-1">Uncertainty Score</div>
              <div className="text-lg font-semibold">
                <span className={
                  metrics.uncertainty < 0.25 ? 'text-green-400' :
                  metrics.uncertainty < 0.4 ? 'text-yellow-400' :
                  'text-red-400'
                }>
                  {(metrics.uncertainty * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="glass rounded-lg p-3">
              <div className="text-xs text-gray-500 mb-1">Detection Quality</div>
              <div className="text-lg font-semibold">
                <span className={
                  metrics.reliability === 'High' ? 'text-green-400' :
                  metrics.reliability === 'Medium' ? 'text-yellow-400' :
                  'text-red-400'
                }>
                  {metrics.reliability}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}