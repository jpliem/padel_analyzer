import React, { useRef, useEffect } from 'react';

interface Props {
  frameBase64: string | null;
  children?: React.ReactNode;
}

const CameraFeed: React.FC<Props> = ({ frameBase64, children }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!frameBase64 || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${frameBase64}`;
  }, [frameBase64]);

  return (
    <div style={{ position: 'relative', height: '100%', background: '#111', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      {frameBase64 ? (
        <canvas ref={canvasRef} style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
      ) : (
        <div style={{ color: '#555', fontSize: 14 }}>Waiting for camera feed...</div>
      )}
      {children}
    </div>
  );
};

export default CameraFeed;
