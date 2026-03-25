import React from 'react';
import { useNavigate } from 'react-router-dom';
import type { MatchSummary } from '../types';

interface Props {
  match: MatchSummary;
}

const statusBadge: Record<string, string> = {
  created: 'badge-created',
  calibrated: 'badge-calibrated',
  analyzed: 'badge-analyzed',
  live: 'badge-live',
};

const MatchCard: React.FC<Props> = ({ match }) => {
  const navigate = useNavigate();

  const handleClick = () => {
    if (match.status === 'analyzed') {
      navigate(`/match/${match.match_id}/analyze`);
    } else if (match.status === 'calibrated') {
      return;
    } else {
      navigate(`/match/${match.match_id}/calibrate`);
    }
  };

  return (
    <div
      onClick={handleClick}
      style={{
        background: 'white', border: '1px solid #e8e8e8', borderRadius: 10,
        padding: 20, cursor: 'pointer', transition: 'box-shadow 0.2s',
      }}
      onMouseOver={e => (e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)')}
      onMouseOut={e => (e.currentTarget.style.boxShadow = 'none')}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 600 }}>{match.match_name}</div>
          <div style={{ fontSize: 12, color: '#888', marginTop: 4 }}>{match.match_id}</div>
        </div>
        <span className={`badge ${statusBadge[match.status] || 'badge-created'}`}>
          {match.status}
        </span>
      </div>
      {match.status === 'calibrated' && (
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
