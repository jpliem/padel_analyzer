import React, { useRef, useState, useEffect } from 'react';

interface Props {
  videoFile: File | null;
  corners: number[][];
  onCornerClick: (x: number, y: number) => void;
  onReset: () => void;
}

const CalibrationCanvas: React.FC<Props> = ({ videoFile, corners, onCornerClick, onReset }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoDims, setVideoDims] = useState({ w: 1280, h: 720 });

  useEffect(() => {
    if (!videoFile || !videoRef.current) return;
    const video = videoRef.current;
    video.src = URL.createObjectURL(videoFile);
    video.onloadedmetadata = () => {
      setVideoDims({ w: video.videoWidth, h: video.videoHeight });
      video.currentTime = 0.1;
    };
    video.onseeked = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.drawImage(video, 0, 0);
    };
  }, [videoFile]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !videoFile) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (videoRef.current) ctx.drawImage(videoRef.current, 0, 0);

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

      // Color coding: baselines=blue, service=cyan, net=yellow
      const colors = [
        '#74b9ff', '#74b9ff',  // k1-k2 baseline
        '#55efc4', '#55efc4', '#55efc4',  // k3-k5 service
        '#fdcb6e', '#fdcb6e',  // k6-k7 net
        '#55efc4', '#55efc4', '#55efc4',  // k8-k10 service
        '#74b9ff', '#74b9ff',  // k11-k12 baseline
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
  }, [corners, videoFile]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Max points handled by parent component
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
      {videoFile ? (
        <canvas ref={canvasRef} onClick={handleCanvasClick}
          style={{ maxWidth: '100%', maxHeight: '100%', cursor: 'crosshair' }} />
      ) : (
        <div style={{ color: '#888', textAlign: 'center', padding: 40 }}>
          <div style={{ fontSize: 18, marginBottom: 8 }}>Upload a video to calibrate</div>
          <div style={{ fontSize: 13 }}>First frame will be shown for corner selection</div>
        </div>
      )}
      {videoFile && (
        <div style={{ position: 'absolute', top: 12, left: 12, padding: '6px 12px', background: 'rgba(0,0,0,0.7)', borderRadius: 6, color: 'white', fontSize: 12 }}>
          Click 4 court corners: near-left → near-right → far-right → far-left
        </div>
      )}
      <div style={{ position: 'absolute', bottom: 12, left: 12, padding: '4px 10px', background: 'rgba(116,185,255,0.2)', border: '1px solid #74b9ff', borderRadius: 6, color: '#74b9ff', fontSize: 11 }}>
        {Math.min(corners.length, 4)}/4 corners{corners.length > 4 ? ` + ${corners.length - 4} ref` : ''}
      </div>
    </div>
  );
};

export default CalibrationCanvas;
