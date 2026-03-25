import React from 'react';

interface PlayerDot {
  id: string;
  x: number;
  y: number;
  team: 'A' | 'B';
  label?: string;
}

interface Props {
  players?: PlayerDot[];
  ballPosition?: { x: number; y: number } | null;
  ballTrail?: { x: number; y: number }[];
}

// Court is 10m wide × 20m long. SVG viewBox maps court coords to pixels.
// Court coords: x=[0,10], y=[0,20]. y=0 is near baseline, y=20 is far baseline.
// SVG: we add padding and flip y so near baseline is at bottom.
const COURT_W = 10;
const COURT_H = 20;
const PAD = 1.5;
const SVG_W = COURT_W + PAD * 2;
const SVG_H = COURT_H + PAD * 2;

// Convert court coords to SVG coords (flip Y so near=bottom)
const cx = (x: number) => x + PAD;
const cy = (y: number) => SVG_H - (y + PAD);

const CourtMiniMap: React.FC<Props> = ({ players = [], ballPosition, ballTrail = [] }) => (
  <div style={{ width: '100%', height: '100%', background: '#1a2332', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 8 }}>
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      style={{ width: '100%', height: '100%', maxHeight: '100%' }}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Background */}
      <rect x={0} y={0} width={SVG_W} height={SVG_H} fill="#1a2332" />

      {/* Court surface */}
      <rect x={cx(0)} y={cy(COURT_H)} width={COURT_W} height={COURT_H} fill="#1b5e20" rx={0.15} />

      {/* Court lines */}
      {/* Outer boundary */}
      <rect x={cx(0)} y={cy(COURT_H)} width={COURT_W} height={COURT_H} fill="none" stroke="white" strokeWidth={0.08} />

      {/* Net line */}
      <line x1={cx(0)} y1={cy(10)} x2={cx(10)} y2={cy(10)} stroke="#fdcb6e" strokeWidth={0.12} />

      {/* Service lines */}
      <line x1={cx(0)} y1={cy(6.95)} x2={cx(10)} y2={cy(6.95)} stroke="white" strokeWidth={0.05} strokeOpacity={0.6} />
      <line x1={cx(0)} y1={cy(13.05)} x2={cx(10)} y2={cy(13.05)} stroke="white" strokeWidth={0.05} strokeOpacity={0.6} />

      {/* Center service line */}
      <line x1={cx(5)} y1={cy(6.95)} x2={cx(5)} y2={cy(13.05)} stroke="white" strokeWidth={0.05} strokeOpacity={0.6} />

      {/* Center marks */}
      <line x1={cx(5)} y1={cy(0)} x2={cx(5)} y2={cy(0.4)} stroke="white" strokeWidth={0.05} strokeOpacity={0.4} />
      <line x1={cx(5)} y1={cy(19.6)} x2={cx(5)} y2={cy(20)} stroke="white" strokeWidth={0.05} strokeOpacity={0.4} />

      {/* Labels */}
      <text x={cx(5)} y={cy(-0.8)} textAnchor="middle" fill="#74b9ff" fontSize={0.6} fontWeight="bold">TEAM A (Near)</text>
      <text x={cx(5)} y={cy(20.5)} textAnchor="middle" fill="#e17055" fontSize={0.6} fontWeight="bold">TEAM B (Far)</text>

      {/* Ball trail */}
      {ballTrail.map((p, i) => (
        <circle
          key={`trail-${i}`}
          cx={cx(p.x)}
          cy={cy(p.y)}
          r={0.12}
          fill="#fdcb6e"
          opacity={0.15 + (i / Math.max(ballTrail.length - 1, 1)) * 0.5}
        />
      ))}

      {/* Ball position */}
      {ballPosition && (
        <>
          <circle cx={cx(ballPosition.x)} cy={cy(ballPosition.y)} r={0.35} fill="#fdcb6e" opacity={0.2} />
          <circle cx={cx(ballPosition.x)} cy={cy(ballPosition.y)} r={0.18} fill="#fdcb6e" />
        </>
      )}

      {/* Player dots */}
      {players.map(p => (
        <g key={p.id}>
          <circle
            cx={cx(p.x)}
            cy={cy(p.y)}
            r={0.4}
            fill={p.team === 'A' ? '#74b9ff' : '#e17055'}
            opacity={0.85}
          />
          <text
            x={cx(p.x)}
            y={cy(p.y) + 0.18}
            textAnchor="middle"
            fill="white"
            fontSize={0.4}
            fontWeight="bold"
          >
            {p.label || p.id}
          </text>
        </g>
      ))}
    </svg>
  </div>
);

export default CourtMiniMap;
