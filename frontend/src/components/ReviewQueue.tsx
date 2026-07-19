import React from 'react';
import type { ReviewRecord } from '../types';

interface Props {
  reviews: ReviewRecord[];
  onResolve: (id: string, confirmed: boolean, winner?: number) => void;
  onSeek?: (frame: number) => void;
}

const ReviewQueue: React.FC<Props> = ({ reviews, onResolve, onSeek }) => {
  const pending = reviews.filter(r => r.status === 'proposed');
  if (!pending.length) return null;
  return (
    <div style={{ padding: 8, background: '#fff8e1', borderBottom: '1px solid #eed48a' }}>
      <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 6 }}>Needs review ({pending.length})</div>
      {pending.map(record => (
        <div key={record.id} style={{ display: 'flex', gap: 5, alignItems: 'center', fontSize: 11, marginBottom: 5 }}>
          <button onClick={() => onSeek?.(record.frame_number)} style={{ border: 0, background: 'none', cursor: 'pointer' }}>
            f{record.frame_number}: {record.reason} ({Math.round(record.confidence * 100)}%)
          </button>
          <button onClick={() => onResolve(record.id, true, record.winner_team || undefined)}>Confirm</button>
          <button onClick={() => onResolve(record.id, false)}>Reject</button>
          <button onClick={() => onResolve(record.id, true, 1)}>Team A</button>
          <button onClick={() => onResolve(record.id, true, 2)}>Team B</button>
        </div>
      ))}
    </div>
  );
};

export default ReviewQueue;
