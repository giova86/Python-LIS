import { useState, useEffect, useRef, useCallback } from 'react'
import Camera from './components/Camera.jsx'
import PredictionSidebar from './components/PredictionSidebar.jsx'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'

export default function App() {
  const [connected, setConnected] = useState(false)
  const [data, setData] = useState(null)
  const wsRef = useRef(null)
  const readyRef = useRef(true)

  useEffect(() => {
    let ws

    const connect = () => {
      ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        readyRef.current = true
      }

      ws.onmessage = (e) => {
        setData(JSON.parse(e.data))
        readyRef.current = true
      }

      ws.onclose = () => {
        setConnected(false)
        setTimeout(connect, 2000)
      }

      ws.onerror = () => ws.close()
    }

    connect()
    return () => ws?.close()
  }, [])

  const sendFrame = useCallback((frameBase64) => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN && readyRef.current) {
      readyRef.current = false
      ws.send(frameBase64)
    }
  }, [])

  return (
    <div className="shell">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark"><em>L</em>IS</span>
          <span className="brand-divider" />
          <span className="brand-meta">
            <span className="brand-title">Riconoscimento Alfabeto</span>
            <span className="brand-sub">Lingua dei Segni · Real-time</span>
          </span>
        </div>
        <div className={`conn ${connected ? 'on' : 'off'}`}>
          <span className="conn-led" />
          {connected ? 'Online' : 'Offline'}
        </div>
      </header>

      <main className="app-main">
        <Camera onFrame={sendFrame} data={data} />
        <PredictionSidebar data={data} />
      </main>
    </div>
  )
}
