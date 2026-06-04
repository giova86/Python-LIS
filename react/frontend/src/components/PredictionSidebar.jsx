import LetterBar from './LetterBar.jsx'

export default function PredictionSidebar({ data }) {
  const probabilities = data?.probabilities ?? {}
  const prediction    = data?.prediction
  const labels        = Object.keys(probabilities).sort()

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <div>
          <div className="sidebar-title">Spettro Confidenza</div>
          <div className="sidebar-sub">Alfabeto LIS — tutte le lettere</div>
        </div>
        {labels.length > 0 && (
          <div className="sidebar-count">{labels.length}</div>
        )}
      </div>

      {labels.length === 0 ? (
        <div className="sidebar-empty">
          <span className="dots">In attesa del backend</span>
        </div>
      ) : (
        <div className="spectrum">
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
