import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'LenScope AI - Object Detection Dashboard',
  description: 'AI-powered Object Detection Dashboard with YOLOv8 - Real-time and image-based detection with multi-model comparison',
  keywords: ['AI', 'Object Detection', 'YOLO', 'Computer Vision', 'Deep Learning'],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-background text-white min-h-screen`}>
        {/* Scanline effect overlay */}
        <div className="scanline" />
        
        {/* Background grid pattern */}
        <div className="fixed inset-0 grid-bg pointer-events-none opacity-50" />
        
        {/* Main content */}
        <div className="relative z-10">
          {children}
        </div>
      </body>
    </html>
  )
}