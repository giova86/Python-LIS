export default function LetterBar({ letter, probability, isActive }) {
  const pct       = Math.round(probability * 100)
  const fillClass = probability > 0.7 ? 'fill-high' : probability > 0.4 ? 'fill-medium' : 'fill-low'

  return (
    <div className={`lbar ${isActive ? 'active' : ''}`}>
      <span className="lbar-letter">{letter.toUpperCase()}</span>
      <div className="lbar-track">
        <div className={`lbar-fill ${fillClass}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="lbar-pct">{pct}%</span>
    </div>
  )
}
