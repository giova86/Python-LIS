import { useEffect, useRef, useState } from 'react'

const STATUS = {
  searching: 'Ricerca…',
  detected:  'Rilevato',
  maybe:     'Forse',
  uncertain: 'Incerto',
}

function getStatus(data) {
  if (!data?.has_hand) return 'searching'
  if (data.confidence > 0.5) return 'detected'
  if (data.confidence > 0.3) return 'maybe'
  return 'uncertain'
}

export default function Camera({ onFrame, data }) {
  const videoRef    = useRef(null)
  const canvasRef   = useRef(null)
  const [camError, setCamError] = useState(null)

  useEffect(() => {
    let stream   = null
    let interval = null

    const start = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        })
        const video = videoRef.current
        if (!video) return
        video.srcObject = stream

        // Capture and send frames — no new frame sent until backend replies
        interval = setInterval(() => {
          const v = videoRef.current
          const c = canvasRef.current
          if (!v || !c || v.readyState < 2) return
          c.width  = v.videoWidth
          c.height = v.videoHeight
          c.getContext('2d').drawImage(v, 0, 0)
          onFrame(c.toDataURL('image/jpeg', 0.75))
        }, 80) // max ~12 fps; actual rate gated by backend ACK via readyRef in App
      } catch (err) {
        setCamError(err.message || 'Camera non disponibile')
      }
    }

    start()
    return () => {
      clearInterval(interval)
      stream?.getTracks().forEach(t => t.stop())
    }
  }, [onFrame])

  const hasHand    = data?.has_hand
  const prediction = data?.prediction
  const confidence = data?.confidence ?? 0
  const status     = getStatus(data)

  return (
    <section className="camera-section">
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {camError ? (
        <div className="camera-error">
          <div className="camera-error-icon">📷</div>
          <h3>Camera non accessibile</h3>
          <p>{camError}</p>
          <p style={{ marginTop: 8, fontSize: 12, opacity: 0.5 }}>
            Controlla i permessi del browser e ricarica la pagina.
          </p>
        </div>
      ) : (
        <video ref={videoRef} autoPlay playsInline muted className="camera-feed" />
      )}

      {!camError && (
        <>
          {/* Hand indicator */}
          <div className={`hand-pill ${hasHand ? 'has-hand' : 'no-hand'}`}>
            <span className="hand-dot" />
            {hasHand ? 'Mano destra' : 'Nessuna mano'}
          </div>

          {/* Bottom overlay */}
          <div className="pred-overlay">
            <div className={`pred-badge ${status}`}>
              {STATUS[status]}
            </div>

            {prediction ? (
              <>
                <div key={prediction} className="pred-letter">
                  {prediction.toUpperCase()}
                </div>
                <div className="pred-conf">
                  Confidenza: {Math.round(confidence * 100)}%
                </div>
              </>
            ) : (
              <p className="pred-prompt">
                Mostra la mano destra alla telecamera…
              </p>
            )}
          </div>
        </>
      )}
    </section>
  )
}
