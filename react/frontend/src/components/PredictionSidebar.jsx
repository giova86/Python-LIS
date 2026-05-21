import LetterBar from './LetterBar.jsx'

export default function PredictionSidebar({ data }) {
  const probabilities = data?.probabilities ?? {}
  const prediction    = data?.prediction
  const labels        = Object.keys(probabilities).sort()

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <div className="sidebar-title">Confidenza Lettere</div>
        <div className="sidebar-sub">Alfabeto LIS — tutte le lettere</div>
      </div>

      {labels.length === 0 ? (
        <div className="sidebar-empty">
          <span style={{ fontSize: 32 }}>⏳</span>
          In attesa del backend…
        </div>
      ) : (
        <div className="letter-list">
          {labels.map(label => (
            <LetterBar
              key={label}
              letter={label}
              probability={probabilities[label] ?? 0}
              isActive={label === prediction}
            />
          ))}
        </div>
      )}
    </aside>
  )
}
