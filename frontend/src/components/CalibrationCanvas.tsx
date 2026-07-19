import React, { useRef, useState, useEffect, useCallback } from 'react';

interface Props {
  videoFile: File | null;
  videoUrl?: string;
  corners: number[][];
  onCornerClick: (x: number, y: number) => void;
  onReset: () => void;
}

const CalibrationCanvas: React.FC<Props> = ({ videoFile, videoUrl, corners, onCornerClick, onReset }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoDims, setVideoDims] = useState({ w: 1280, h: 720 });
  const [videoReady, setVideoReady] = useState(false);

  // Draw the video frame + keypoint overlays
  const drawOverlays = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !videoReady) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw video frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    if (corners.length > 0) {
      // Draw court lines connecting keypoints
      const lines: [number, number][] = [
        [0, 1],   // k1-k2 near baseline
        [10, 11], // k11-k12 far baseline
        [0, 10],  // k1-k11 left sideline
        [1, 11],  // k2-k12 right sideline
        [5, 6],   // k6-k7 net
        [2, 4],   // k3-k5 near service line
        [7, 9],   // k8-k10 far service line
        [3, 8],   // k4-k9 center service line
      ];

      ctx.lineWidth = 2;
      for (const [a, b] of lines) {
        if (a < corners.length && b < corners.length) {
          ctx.beginPath();
          ctx.moveTo(corners[a][0], corners[a][1]);
          ctx.lineTo(corners[b][0], corners[b][1]);
          ctx.strokeStyle = (a === 5 && b === 6) ? '#fdcb6e' : '#00b894';
          ctx.setLineDash([]);
          ctx.stroke();
        }
      }

      const colors = [
        '#74b9ff', '#74b9ff',
        '#55efc4', '#55efc4', '#55efc4',
        '#fdcb6e', '#fdcb6e',
        '#55efc4', '#55efc4', '#55efc4',
        '#74b9ff', '#74b9ff',
      ];

      corners.forEach((c, i) => {
        ctx.beginPath();
        ctx.arc(c[0], c[1], 8, 0, Math.PI * 2);
        ctx.fillStyle = colors[i] || '#74b9ff';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = 'white';
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${i + 1}`, c[0], c[1]);
      });
    }
  }, [corners, videoReady]);

  // Load video and wait for first frame
  useEffect(() => {
    if ((!videoFile && !videoUrl) || !videoRef.current) return;
    setVideoReady(false);
    const video = videoRef.current;
    video.crossOrigin = 'anonymous';
    const source = videoFile ? URL.createObjectURL(videoFile) : videoUrl!;
    video.src = source;
    video.onloadedmetadata = () => {
      setVideoDims({ w: video.videoWidth, h: video.videoHeight });
      video.currentTime = 0.1;
    };
    video.onseeked = () => {
      setVideoReady(true);
    };
    return () => { if (videoFile) URL.revokeObjectURL(source); };
  }, [videoFile, videoUrl]);

  // Redraw when video is ready OR corners change
  useEffect(() => {
    drawOverlays();
  }, [drawOverlays]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = videoDims.w / rect.width;
    const scaleY = videoDims.h / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    onCornerClick(Math.round(x), Math.round(y));
  };

  return (
    <div style={{ position: 'relative', background: '#111', display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
      <video ref={videoRef} style={{ display: 'none' }} muted />
      {videoFile || videoUrl ? (
        <canvas ref={canvasRef} onClick={handleCanvasClick}
          style={{ maxWidth: '100%', maxHeight: '100%', cursor: 'crosshair' }} />
      ) : (
        <div style={{ color: '#888', textAlign: 'center', padding: 40 }}>
          <div style={{ fontSize: 18, marginBottom: 8 }}>Upload a video to calibrate</div>
          <div style={{ fontSize: 13 }}>First frame will be shown for corner selection</div>
        </div>
      )}
      {(videoFile || videoUrl) && corners.length === 0 && (
        <div style={{ position: 'absolute', top: 12, left: 12, padding: '6px 12px', background: 'rgba(0,0,0,0.7)', borderRadius: 6, color: 'white', fontSize: 12 }}>
          Click "Auto-Detect" or manually click 12 court keypoints
        </div>
      )}
      {corners.length > 0 && (
        <div style={{ position: 'absolute', top: 12, left: 12, padding: '6px 12px', background: 'rgba(0,184,148,0.8)', borderRadius: 6, color: 'white', fontSize: 12 }}>
          {corners.length >= 12 ? '12 keypoints detected ✓' : `${corners.length}/12 keypoints`}
        </div>
      )}
      <div style={{ position: 'absolute', bottom: 12, left: 12, padding: '4px 10px', background: 'rgba(116,185,255,0.2)', border: '1px solid #74b9ff', borderRadius: 6, color: '#74b9ff', fontSize: 11 }}>
        {corners.length} points set
      </div>
    </div>
  );
};

export default CalibrationCanvas;
