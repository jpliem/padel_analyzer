import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { calibrate, uploadVideo, listTemplates, getTemplate, saveTemplate, CalibrationTemplate } from '../api';
import CalibrationCanvas from '../components/CalibrationCanvas';
import CourtMiniMap from '../components/CourtMiniMap';

// 12 keypoints matching the backend KEYPOINT_COURT_COORDS_12
const KEYPOINT_LABELS = [
  'K1: Near-Left Baseline',
  'K2: Near-Right Baseline',
  'K3: Near-Left Service',
  'K4: Near-Center Service',
  'K5: Near-Right Service',
  'K6: Net-Left',
  'K7: Net-Right',
  'K8: Far-Left Service',
  'K9: Far-Center Service',
  'K10: Far-Right Service',
  'K11: Far-Left Baseline',
  'K12: Far-Right Baseline',
];

// Court coords for preview
const KEYPOINT_COURT = [
  [0, 0], [10, 0], [0, 6.95], [5, 6.95], [10, 6.95],
  [0, 10], [10, 10], [0, 13.05], [5, 13.05], [10, 13.05],
  [0, 20], [10, 20],
];

const Calibration: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [keypoints, setKeypoints] = useState<number[][]>([]);
  const [netTopPoints, setNetTopPoints] = useState<number[][]>([]);
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

  const handleClick = (x: number, y: number) => {
    if (keypoints.length < 12) {
      setKeypoints([...keypoints, [x, y]]);
    } else if (netTopPoints.length < 2) {
      // After 12 ground keypoints, collect 2 net top points
      setNetTopPoints([...netTopPoints, [x, y]]);
    }
  };

  const handleSave = async () => {
    if (!id || keypoints.length < 4) return;
    setSaving(true);
    setError(null);
    try {
      // Send ground keypoints + optional net top points for 3D camera model
      await calibrate(id, keypoints, null, netTopPoints.length === 2 ? netTopPoints : null);
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
    if (!templateName.trim() || keypoints.length < 4) return;
    let thumbnail: string | null = null;
    const canvas = document.querySelector('canvas');
    if (canvas) {
      thumbnail = canvas.toDataURL('image/jpeg', 0.5).split(',')[1];
    }
    try {
      const result = await saveTemplate({
        name: templateName.trim(),
        corners: keypoints,
        thumbnail,
      });
      setTemplates([...templates, { id: result.id, name: templateName.trim(), corners: keypoints, has_thumbnail: !!thumbnail }]);
      setShowSaveTemplate(false);
      setTemplateName('');
    } catch (err: any) {
      setError(err.message);
    }
  };

  const loadTemplate = async (templateId: string) => {
    try {
      const t = await getTemplate(templateId);
      setKeypoints(t.corners);
    } catch (err: any) {
      setError(err.message);
    }
  };

  // Preview dots on the 2D court
  const previewDots = keypoints.map((_, i) => ({
    id: `K${i + 1}`,
    x: i < KEYPOINT_COURT.length ? KEYPOINT_COURT[i][0] : 5,
    y: i < KEYPOINT_COURT.length ? KEYPOINT_COURT[i][1] : 10,
    team: 'A' as const,
    label: `${i + 1}`,
  }));

  const currentLabel = keypoints.length < 12 ? KEYPOINT_LABELS[keypoints.length] : 'All 12 set!';
  const progress = keypoints.length;

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 56px)' }}>
      {/* Left: Video frame */}
      <div style={{ flex: 2 }}>
        <CalibrationCanvas
          videoFile={videoFile}
          corners={keypoints}
          onCornerClick={handleClick}
          onReset={() => setKeypoints([])}
        />
      </div>

      {/* Right: Controls */}
      <div style={{ flex: 1, minWidth: 300, padding: 16, display: 'flex', flexDirection: 'column', gap: 8, background: '#fafafa', borderLeft: '1px solid #e0e0e0', overflow: 'auto' }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, margin: 0 }}>Court Calibration (12 Keypoints)</h2>

        {/* Templates */}
        {templates.length > 0 && (
          <div>
            <div className="label">Saved Templates</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {templates.map(t => (
                <button key={t.id} onClick={() => loadTemplate(t.id)}
                  style={{ padding: '4px 8px', background: 'white', border: '1px solid #d0d0d0', borderRadius: 4, fontSize: 11, cursor: 'pointer' }}>
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
            if (file) { setVideoFile(file); setKeypoints([]); setSaved(false); }
          }} style={{ fontSize: 12 }} />
        </div>

        {/* Progress bar */}
        <div>
          <div className="label">Keypoints: {progress}/12 {netTopPoints.length > 0 ? `+ ${netTopPoints.length}/2 net tops` : ''}</div>
          <div style={{ height: 6, background: '#e8e8e8', borderRadius: 3, overflow: 'hidden' }}>
            <div style={{
              width: `${((progress + netTopPoints.length) / 14) * 100}%`, height: '100%',
              background: netTopPoints.length === 2 ? '#6c5ce7' : progress >= 12 ? '#00b894' : progress >= 4 ? '#fdcb6e' : '#74b9ff',
              borderRadius: 3, transition: 'width 0.2s',
            }} />
          </div>
        </div>

        {/* Current instruction */}
        <div style={{
          padding: 10, borderRadius: 6, fontSize: 12,
          background: progress >= 12 ? '#f0fff4' : '#f0f4ff',
          border: `1px solid ${progress >= 12 ? '#00b894' : '#74b9ff'}`,
          color: progress >= 12 ? '#00b894' : '#333',
        }}>
          {progress < 12 ? (
            <>Click: <strong>{currentLabel}</strong></>
          ) : netTopPoints.length < 2 ? (
            <>
              <strong style={{ color: '#6c5ce7' }}>
                Now click TOP of {netTopPoints.length === 0 ? 'left' : 'right'} net post (for 3D)
              </strong>
              <br /><span style={{ fontSize: 11, color: '#888' }}>Or save now with 2D-only mode</span>
            </>
          ) : (
            <strong style={{ color: '#6c5ce7' }}>3D calibration ready — net height reference set!</strong>
          )}
        </div>

        {/* Keypoint diagram */}
        <div style={{ fontSize: 10, fontFamily: 'monospace', background: '#1a2332', color: '#8899aa', padding: 8, borderRadius: 6, lineHeight: 1.6 }}>
          <div style={{ textAlign: 'center' }}>
            <span style={{ color: progress > 10 ? '#00b894' : '#555' }}>K11</span>
            ──────────────────
            <span style={{ color: progress > 11 ? '#00b894' : '#555' }}>K12</span>
          </div>
          <div style={{ textAlign: 'center' }}>│ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │</div>
          <div style={{ textAlign: 'center' }}>
            <span style={{ color: progress > 7 ? '#00b894' : '#555' }}>K8</span>
            ────
            <span style={{ color: progress > 8 ? '#00b894' : '#555' }}>K9</span>
            ────
            <span style={{ color: progress > 9 ? '#00b894' : '#555' }}>K10</span>
          </div>
          <div style={{ textAlign: 'center' }}>│ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │</div>
          <div style={{ textAlign: 'center', color: '#fdcb6e' }}>
            <span style={{ color: progress > 5 ? '#00b894' : '#fdcb6e' }}>K6</span>
            ───── NET ─────
            <span style={{ color: progress > 6 ? '#00b894' : '#fdcb6e' }}>K7</span>
          </div>
          <div style={{ textAlign: 'center' }}>│ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │</div>
          <div style={{ textAlign: 'center' }}>
            <span style={{ color: progress > 2 ? '#00b894' : '#555' }}>K3</span>
            ────
            <span style={{ color: progress > 3 ? '#00b894' : '#555' }}>K4</span>
            ────
            <span style={{ color: progress > 4 ? '#00b894' : '#555' }}>K5</span>
          </div>
          <div style={{ textAlign: 'center' }}>│ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │</div>
          <div style={{ textAlign: 'center' }}>
            <span style={{ color: progress > 0 ? '#00b894' : '#555' }}>K1</span>
            ──────────────────
            <span style={{ color: progress > 1 ? '#00b894' : '#555' }}>K2</span>
          </div>
        </div>

        {/* 2D court preview */}
        <div style={{ height: 130 }}><CourtMiniMap players={previewDots} /></div>

        {error && <div style={{ padding: 8, background: '#fff3f3', border: '1px solid #e17055', borderRadius: 6, color: '#e17055', fontSize: 12 }}>{error}</div>}

        {/* Actions */}
        <div style={{ display: 'flex', gap: 6 }}>
          <button className="btn btn-outline" style={{ flex: 1, fontSize: 11 }} onClick={() => {
            if (keypoints.length > 0) setKeypoints(keypoints.slice(0, -1));
          }}>Undo Last</button>
          <button className="btn btn-outline" style={{ flex: 1, fontSize: 11 }} onClick={() => setKeypoints([])}>
            Reset All
          </button>
          <button className="btn btn-success" style={{ flex: 1, fontSize: 11 }} onClick={handleSave}
            disabled={keypoints.length < 4 || saving}>
            {saving ? 'Saving...' : `Save (${progress}pt)`}
          </button>
        </div>

        {/* Minimum notice */}
        {progress >= 4 && progress < 12 && (
          <div style={{ fontSize: 11, color: '#888' }}>
            Min 4 points to save. {12 - progress} more for best accuracy.
          </div>
        )}

        {/* Save as template */}
        {keypoints.length >= 4 && !showSaveTemplate && (
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
            <button className="btn btn-primary" onClick={() => navigate(`/match/${id}/analyze`)}>Analyze Video →</button>
            <button className="btn btn-outline" onClick={() => navigate(`/match/${id}/live`)}>Go Live →</button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Calibration;
