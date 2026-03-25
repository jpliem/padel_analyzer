import React from 'react';
import type { ScoreData } from '../types';

interface Props {
  score: ScoreData | null;
  teamA?: string;
  teamB?: string;
  variant?: 'overlay' | 'sidebar';
}

const Scoreboard: React.FC<Props> = ({ score, teamA = 'Team A', teamB = 'Team B', variant = 'sidebar' }) => {
  if (!score) return null;

  if (variant === 'overlay') {
    return (
      <div style={{ display: 'flex', background: 'rgba(0,0,0,0.85)', borderRadius: 8, overflow: 'hidden', padding: 2 }}>
        <div style={{ padding: '8px 20px', textAlign: 'center' }}>
          <div style={{ fontSize: 9, color: '#74b9ff', textTransform: 'uppercase' as const }}>{teamA}</div>
          <div style={{ fontSize: 26, fontWeight: 700, color: 'white' }}>{score.score.split(' - ')[0]}</div>
        </div>
        <div style={{ width: 1, background: '#444' }} />
        <div style={{ padding: '8px 20px', textAlign: 'center' }}>
          <div style={{ fontSize: 9, color: '#e17055', textTransform: 'uppercase' as const }}>{teamB}</div>
          <div style={{ fontSize: 26, fontWeight: 700, color: 'white' }}>{score.score.split(' - ')[1]}</div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: '12px 16px', background: 'white', borderBottom: '1px solid #e8e8e8' }}>
      <div className="label">Score</div>
      <div style={{ display: 'flex', justifyContent: 'center', gap: 16, alignItems: 'baseline', marginTop: 8 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 10, color: '#74b9ff' }}>{teamA}</div>
          <div style={{ fontSize: 28, fontWeight: 700 }}>{score.score.split(' - ')[0]}</div>
        </div>
        <div style={{ fontSize: 14, color: '#888' }}>-</div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 10, color: '#e17055' }}>{teamB}</div>
          <div style={{ fontSize: 28, fontWeight: 700 }}>{score.score.split(' - ')[1]}</div>
        </div>
      </div>
      <div style={{ textAlign: 'center', fontSize: 12, color: '#888', marginTop: 4 }}>
        Games: {score.games} | Sets: {score.sets}
      </div>
    </div>
  );
};

export default Scoreboard;
