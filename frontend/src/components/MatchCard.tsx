import React from 'react';
import { useNavigate } from 'react-router-dom';
import { deleteMatch } from '../api';
import type { MatchSummary } from '../types';

interface Props {
  match: MatchSummary;
  onDeleted?: () => void;
}

const statusBadge: Record<string, string> = {
  created: 'badge-created',
  calibrated: 'badge-calibrated',
  uploaded: 'badge-calibrated',
  processing: 'badge-processing',
  analyzed: 'badge-analyzed',
  live: 'badge-live',
};

const MatchCard: React.FC<Props> = ({ match, onDeleted }) => {
  const navigate = useNavigate();

  const handleClick = () => {
    if (match.status === 'analyzed' || match.status === 'processing') {
      navigate(`/match/${match.match_id}/analyze`);
    } else if (match.status === 'calibrated') {
      return;
    } else {
      navigate(`/match/${match.match_id}/calibrate`);
    }
  };

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm(`Delete match "${match.match_name}" (${match.match_id})?`)) return;
    try {
      await deleteMatch(match.match_id);
      onDeleted?.();
    } catch (err) {
      alert('Failed to delete match');
    }
  };

  return (
    <div
      onClick={handleClick}
      style={{
        background: 'white', border: '1px solid #e8e8e8', borderRadius: 10,
        padding: 20, cursor: 'pointer', transition: 'box-shadow 0.2s', position: 'relative',
      }}
      onMouseOver={e => (e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)')}
      onMouseOut={e => (e.currentTarget.style.boxShadow = 'none')}
    >
      {/* Delete button */}
      <button
        onClick={handleDelete}
        style={{
          position: 'absolute', top: 8, right: 8,
          background: 'none', border: 'none', cursor: 'pointer',
          color: '#ccc', fontSize: 16, padding: '2px 6px', borderRadius: 4,
          transition: 'color 0.2s',
        }}
        onMouseOver={e => (e.currentTarget.style.color = '#e17055')}
        onMouseOut={e => (e.currentTarget.style.color = '#ccc')}
        title="Delete match"
      >
        ×
      </button>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', paddingRight: 20 }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 600 }}>{match.match_name}</div>
          <div style={{ fontSize: 12, color: '#888', marginTop: 4 }}>{match.match_id}</div>
          {match.media && <div style={{ fontSize: 11, color: '#999', marginTop: 7 }}>{match.media.original_name} · {Math.floor(match.media.duration_seconds / 60)}:{String(Math.floor(match.media.duration_seconds % 60)).padStart(2, '0')}</div>}
        </div>
        <span className={`badge ${statusBadge[match.status] || 'badge-created'}`}>
          {match.status}
        </span>
      </div>
      {(match.status === 'calibrated' || match.status === 'uploaded') && (
        <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
          <button className="btn btn-primary" style={{ fontSize: 12 }}
            onClick={e => { e.stopPropagation(); navigate(`/match/${match.match_id}/analyze`); }}>
            Analyze Video
          </button>
          <button className="btn btn-outline" style={{ fontSize: 12 }}
            onClick={e => { e.stopPropagation(); navigate(`/match/${match.match_id}/live`); }}>
            Go Live
          </button>
        </div>
      )}
    </div>
  );
};

export default MatchCard;
