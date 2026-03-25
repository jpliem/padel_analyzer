import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { calibrate, uploadVideo, listTemplates, getTemplate, saveTemplate, CalibrationTemplate } from '../api';
import CalibrationCanvas from '../components/CalibrationCanvas';
import CourtMiniMap from '../components/CourtMiniMap';

type PointMode = 'corners' | 'net';

const POINT_LABELS: Record<PointMode, string[]> = {
  corners: ['Near-Left', 'Near-Right', 'Far-Right', 'Far-Left'],
  net: ['Net-Left', 'Net-Right'],
};

const Calibration: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [corners, setCorners] = useState<number[][]>([]);
  const [netPoints, setNetPoints] = useState<number[][]>([]);
  const [pointMode, setPointMode] = useState<PointMode>('corners');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Templates
  const [templates, setTemplates] = useState<CalibrationTemplate[]>([]);
  const [templateName, setTemplateName] = useState('');
  const [showSaveTemplate, setShowSaveTemplate] = useState(false);

  useEffect(() => {
    listTemplates().then(r => setTemplates(r.templates)).catch(() => {});
  }, []);

  const activePoints = pointMode === 'corners' ? corners : netPoints;
  const maxPoints = pointMode === 'corners' ? 4 : 2;
  const labels = POINT_LABELS[pointMode];

  const handleCornerClick = (x: number, y: number) => {
    if (pointMode === 'corners' && corners.length < 4) {
      setCorners([...corners, [x, y]]);
    } else if (pointMode === 'net' && netPoints.length < 2) {
      setNetPoints([...netPoints, [x, y]]);
    }
  };

  const handleSave = async () => {
    if (!id || corners.length !== 4) return;
    setSaving(true);
    setError(null);
    try {
      await calibrate(id, corners, netPoints.length === 2 ? netPoints : null);
      if (videoFile) {
        await uploadVideo(id, videoFile);
      }
      setSaved(true);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleSaveTemplate = async () => {
    if (!templateName.trim() || corners.length !== 4) return;
    // Capture thumbnail from video
    let thumbnail: string | null = null;
    const canvas = document.querySelector('canvas');
    if (canvas) {
      thumbnail = canvas.toDataURL('image/jpeg', 0.5).split(',')[1];
    }
    try {
      const result = await saveTemplate({
        name: templateName.trim(),
        corners,
        net_points: netPoints.length === 2 ? netPoints : null,
        thumbnail,
      });
      setTemplates([...templates, { id: result.id, name: templateName.trim(), corners, net_points: netPoints.length === 2 ? netPoints : null, has_thumbnail: !!thumbnail }]);
      setShowSaveTemplate(false);
      setTemplateName('');
    } catch (err: any) {
      setError(err.message);
    }
  };

  const loadTemplate = async (templateId: string) => {
    try {
      const t = await getTemplate(templateId);
      setCorners(t.corners);
      if (t.net_points) setNetPoints(t.net_points);
      setPointMode('corners');
    } catch (err: any) {
      setError(err.message);
    }
  };

  // Court preview — show where points are in court coords
  const previewPlayers = corners.length === 4 ? corners.map((c, i) => ({
    id: `C${i + 1}`,
    x: i < 2 ? (i === 0 ? 0 : 10) : (i === 2 ? 10 : 0),
    y: i < 2 ? 0 : 20,
    team: 'A' as const,
    label: `C${i + 1}`,
  })) : [];

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 56px)' }}>
      {/* Left: Video frame */}
      <div style={{ flex: 2 }}>
        <CalibrationCanvas
          videoFile={videoFile}
          corners={[...corners, ...netPoints]}
          onCornerClick={handleCornerClick}
          onReset={() => {
            if (pointMode === 'corners') setCorners([]);
            else setNetPoints([]);
          }}
        />
      </div>

      {/* Right: Controls */}
      <div style={{ flex: 1, minWidth: 300, padding: 16, display: 'flex', flexDirection: 'column', gap: 10, background: '#fafafa', borderLeft: '1px solid #e0e0e0', overflow: 'auto' }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, margin: 0 }}>Court Calibration</h2>

        {/* Templates section */}
        {templates.length > 0 && (
          <div>
            <div className="label">Saved Templates</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {templates.map(t => (
                <button
                  key={t.id}
                  onClick={() => loadTemplate(t.id)}
                  style={{
                    padding: '6px 10px', background: 'white', border: '1px solid #d0d0d0',
                    borderRadius: 6, fontSize: 11, cursor: 'pointer', display: 'flex',
                    alignItems: 'center', gap: 6,
                  }}
                >
                  {t.has_thumbnail && (
                    <img
                      src={`http://localhost:8000/templates/${t.id}/thumbnail`}
                      alt=""
                      style={{ width: 32, height: 20, borderRadius: 2, objectFit: 'cover' }}
                      onError={e => (e.currentTarget.style.display = 'none')}
                    />
                  )}
                  {t.name}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Video source */}
        <div>
          <div className="label">Video Source</div>
          <input type="file" accept="video/*" onChange={e => {
            const file = e.target.files?.[0];
            if (file) { setVideoFile(file); setCorners([]); setNetPoints([]); setSaved(false); }
          }} style={{ fontSize: 12 }} />
        </div>

        {/* Point mode selector */}
        <div>
          <div className="label">Reference Points</div>
          <div style={{ display: 'flex', gap: 4 }}>
            <button
              onClick={() => setPointMode('corners')}
              style={{
                flex: 1, padding: 6, fontSize: 12, borderRadius: 6, cursor: 'pointer',
                border: pointMode === 'corners' ? '2px solid #1a1a2e' : '1px solid #d0d0d0',
                background: pointMode === 'corners' ? '#1a1a2e' : 'white',
                color: pointMode === 'corners' ? 'white' : '#555',
              }}
            >
              4 Corners {corners.length === 4 ? '✓' : `(${corners.length}/4)`}
            </button>
            <button
              onClick={() => setPointMode('net')}
              style={{
                flex: 1, padding: 6, fontSize: 12, borderRadius: 6, cursor: 'pointer',
                border: pointMode === 'net' ? '2px solid #6c5ce7' : '1px solid #d0d0d0',
                background: pointMode === 'net' ? '#6c5ce7' : 'white',
                color: pointMode === 'net' ? 'white' : '#555',
              }}
            >
              Net Posts {netPoints.length === 2 ? '✓' : `(${netPoints.length}/2)`}
            </button>
          </div>
        </div>

        {/* Current point instruction */}
        <div style={{ padding: 8, background: '#f0f4ff', borderRadius: 6, fontSize: 12, color: '#555' }}>
          {pointMode === 'corners' ? (
            corners.length < 4 ? (
              <>Click <strong>{labels[corners.length]}</strong> corner on the video</>
            ) : (
              <span style={{ color: '#00b894' }}>All 4 corners set ✓</span>
            )
          ) : (
            netPoints.length < 2 ? (
              <>Click <strong>{labels[netPoints.length]}</strong> post on the video</>
            ) : (
              <span style={{ color: '#00b894' }}>Both net posts set ✓</span>
            )
          )}
        </div>

        {/* Point list */}
        <div style={{ fontSize: 11, color: '#888' }}>
          {corners.map((c, i) => (
            <div key={`c${i}`}>Corner {i + 1}: ({Math.round(c[0])}, {Math.round(c[1])})</div>
          ))}
          {netPoints.map((c, i) => (
            <div key={`n${i}`} style={{ color: '#6c5ce7' }}>Net {i + 1}: ({Math.round(c[0])}, {Math.round(c[1])})</div>
          ))}
        </div>

        {/* 2D court preview */}
        <div style={{ height: 150 }}><CourtMiniMap players={previewPlayers} /></div>

        {error && <div style={{ padding: 8, background: '#fff3f3', border: '1px solid #e17055', borderRadius: 6, color: '#e17055', fontSize: 12 }}>{error}</div>}

        {/* Actions */}
        <div style={{ display: 'flex', gap: 6 }}>
          <button className="btn btn-outline" style={{ flex: 1, fontSize: 11 }} onClick={() => {
            setCorners([]); setNetPoints([]); setPointMode('corners');
          }}>Reset All</button>
          <button className="btn btn-success" style={{ flex: 1, fontSize: 11 }} onClick={handleSave}
            disabled={corners.length !== 4 || saving}>
            {saving ? 'Saving...' : 'Save & Upload'}
          </button>
        </div>

        {/* Save as template */}
        {corners.length === 4 && !showSaveTemplate && (
          <button className="btn btn-outline" style={{ fontSize: 11 }} onClick={() => setShowSaveTemplate(true)}>
            Save as Template
          </button>
        )}
        {showSaveTemplate && (
          <div style={{ display: 'flex', gap: 6 }}>
            <input type="text" placeholder="Template name..." value={templateName}
              onChange={e => setTemplateName(e.target.value)} style={{ fontSize: 12, padding: 6 }} />
            <button className="btn btn-primary" style={{ fontSize: 11, whiteSpace: 'nowrap' }} onClick={handleSaveTemplate}>Save</button>
          </div>
        )}

        {/* Post-save navigation */}
        {saved && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: 12, background: '#f0fff4', border: '1px solid #00b894', borderRadius: 8 }}>
            <div style={{ fontSize: 13, color: '#00b894', fontWeight: 500 }}>Calibration saved!</div>
            <button className="btn btn-primary" onClick={() => navigate(`/match/${id}/analyze`)}>
              Analyze Video →
            </button>
            <button className="btn btn-outline" onClick={() => navigate(`/match/${id}/live`)}>
              Go Live →
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Calibration;
