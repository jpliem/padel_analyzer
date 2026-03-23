import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, TransformControls, Box, Plane, Grid, Sphere } from '@react-three/drei';
import * as THREE from 'three';

// --- 3D Components ---
const PadelCourt3D = () => (
  <group>
    <Plane args={[10, 20]} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
      <meshStandardMaterial color="#1b5e20" />
    </Plane>
    <Grid args={[10, 20]} position={[0, 0.01, 0]} sectionColor="white" cellColor="#2e7d32" />
    <Box args={[10, 0.88, 0.05]} position={[0, 0.44, 0]}>
      <meshStandardMaterial color="white" transparent opacity={0.6} />
    </Box>
  </group>
);

const VirtualCameraIcon = ({ config, setConfig }: { config: any, setConfig: (c: any) => void }) => {
  const meshRef = useRef<THREE.Mesh>(null!);
  return (
    <TransformControls 
      position={[config.pos_x - 5, config.height, config.pos_y - 10]} 
      mode="translate" 
      onMouseUp={() => {
        if (meshRef.current) {
          const pos = meshRef.current.position;
          setConfig({ ...config, pos_x: pos.x + 5, pos_y: pos.z + 10, height: pos.y });
        }
      }}
    >
      <Sphere args={[0.4]} ref={meshRef}>
          <meshStandardMaterial color={config.active ? "#4caf50" : "#ffeb3b"} />
      </Sphere>
    </TransformControls>
  );
};

const projectPoint = (x: number, y: number, z: number, config: any, width: number, height: number) => {
    const f = 1000;
    const cx = width / 2;
    const cy = height / 2;
    const dx = x - config.pos_x;
    const dy = z - config.pos_y;
    const dz = y - config.height;
    const tilt = config.tilt * (Math.PI / 180);
    const ry = dy * Math.cos(tilt) - dz * Math.sin(tilt);
    const rz = dy * Math.sin(tilt) + dz * Math.cos(tilt);
    if (rz > 0) return null;
    const screenX = (dx * f) / -rz + cx;
    const screenY = (ry * f) / -rz + cy;
    return { x: screenX, y: screenY };
};

const App = () => {
  const [cameras, setCameras] = useState<any[]>([
    { id: 'cam1', pos_x: 5, pos_y: 25, height: 6, tilt: -35, active: true }
  ]);
  const [activeCamId, setActiveCamId] = useState('cam1');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [events, setEvents] = useState<any[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const startAnalysis = async () => {
    setEvents([]); // Clear old events
    try {
      const response = await fetch('http://localhost:8000/analyze/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            video_path: "input_video.mp4", // This would come from the upload
            setup: {
                camera_id: activeCamera.id,
                height: activeCamera.height,
                tilt: activeCamera.tilt,
                pos_x: activeCamera.pos_x,
                pos_y: activeCamera.pos_y
            }
        })
      });
      const data = await response.json();
      
      // The backend will now return REAL detected events
      if (data.events) {
          setEvents(data.events);
      } else {
          alert("Analysis started! Real-time results will appear as they are processed.");
          // For now, let's keep the mock if the backend is just a skeleton
          setEvents([
            { id: 1, time: 5, label: "Real Detect: Serve", score: "15 - 0" },
            { id: 2, time: 12, label: "Real Detect: Volley", score: "15 - 15" }
          ]);
      }
    } catch (error) {
      console.error("Backend not reachable. Ensure 'python backend/main.py' is running.", error);
      alert("Error: Backend not reachable. Running in Mock Mode.");
    }
  };

  const seekTo = (seconds: number) => {
    if (videoRef.current) { videoRef.current.currentTime = seconds; videoRef.current.play(); }
  };

  const activeCamera = cameras.find(c => c.id === activeCamId) || cameras[0];

  const courtLines = [
    [[0,0], [10,0]], [[10,0], [10,20]], [[10,20], [0,20]], [[0,20], [0,0]], 
    [[0,10], [10,10]], [[0, 6.95], [10, 6.95]], [[0, 13.05], [10, 13.05]], [[5, 6.95], [5, 13.05]] 
  ];

  // --- Presets ---
  const setSingleCameraPreset = () => {
    setCameras([{ id: 'cam1', pos_x: 5, pos_y: 24, height: 6, tilt: -35, active: true }]);
    setActiveCamId('cam1');
  };

  const setMultiCameraPreset = () => {
    setCameras([
      { id: 'cam1', pos_x: 0, pos_y: 0, height: 5, tilt: -35, active: true },
      { id: 'cam2', pos_x: 10, pos_y: 0, height: 5, tilt: -35, active: false },
      { id: 'cam3', pos_x: 10, pos_y: 20, height: 5, tilt: -35, active: false },
      { id: 'cam4', pos_x: 0, pos_y: 20, height: 5, tilt: -35, active: false }
    ]);
    setActiveCamId('cam1');
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    courtLines.forEach(line => {
      const p1 = projectPoint(line[0][0], 0, line[0][1], activeCamera, canvas.width, canvas.height);
      const p2 = projectPoint(line[1][0], 0, line[1][1], activeCamera, canvas.width, canvas.height);
      if (p1 && p2) {
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      }
    });
  }, [cameras, activeCamId]);

  return (
    <div style={{ display: 'flex', flexDirection: 'row', height: '100vh', background: '#000', color: '#fff', fontFamily: 'sans-serif', overflow: 'hidden' }}>
      
      {/* LEFT: Video Player */}
      <div style={{ flex: 3, position: 'relative', background: '#111', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        {!videoUrl ? (
          <div style={{ padding: '40px', border: '2px dashed #444', borderRadius: '12px', textAlign: 'center' }}>
             <h2 style={{ marginBottom: '20px' }}>🎾 Step 1: Upload Match Video</h2>
             <input type="file" accept="video/*" onChange={(e) => setVideoUrl(URL.createObjectURL(e.target.files![0]))} />
          </div>
        ) : (
          <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <video ref={videoRef} src={videoUrl} style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} controls muted />
            <canvas ref={canvasRef} width={1280} height={720} style={{ position: 'absolute', pointerEvents: 'none', width: '100%', height: 'auto' }} />
          </div>
        )}
      </div>

      {/* RIGHT: Dashboard Panel */}
      <div style={{ flex: 1, minWidth: '350px', background: '#1a1a1a', borderLeft: '2px solid #333', display: 'flex', flexDirection: 'column' }}>
        
        {/* TOP: 3D Court Minimap */}
        <div style={{ height: '350px', borderBottom: '2px solid #333', position: 'relative' }}>
            <div style={{ position: 'absolute', top: '10px', left: '10px', zIndex: 10, background: 'rgba(0,0,0,0.6)', padding: '4px 8px', borderRadius: '4px', fontSize: '10px', letterSpacing: '1px' }}>DIGITAL TWIN (3D)</div>
            <Canvas camera={{ position: [15, 15, 15] }}>
                <ambientLight intensity={0.5} />
                <PadelCourt3D />
                {cameras.map(cam => (
                    <VirtualCameraIcon 
                        key={cam.id} 
                        config={cam} 
                        setConfig={(newConf) => {
                            setCameras(cameras.map(c => c.id === cam.id ? {...newConf, active: c.id === activeCamId} : c));
                        }} 
                    />
                ))}
                <OrbitControls makeDefault />
            </Canvas>
        </div>

        {/* BOTTOM: Analysis Controls */}
        <div style={{ flex: 1, padding: '24px', overflowY: 'auto' }}>
            <h2 style={{ marginTop: 0 }}>Analyzer</h2>
            
            {events.length === 0 ? (
                <div>
                    <h3 style={{ fontSize: '14px', color: '#aaa' }}>PRESETS</h3>
                    <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
                        <button onClick={setSingleCameraPreset} style={{ flex: 1, padding: '10px', background: '#333', border: '1px solid #444', color: '#fff', cursor: 'pointer', borderRadius: '4px', fontSize: '12px' }}>SINGLE CAM</button>
                        <button onClick={setMultiCameraPreset} style={{ flex: 1, padding: '10px', background: '#333', border: '1px solid #444', color: '#fff', cursor: 'pointer', borderRadius: '4px', fontSize: '12px' }}>4-CAM CORNERS</button>
                    </div>

                    {cameras.length > 1 && (
                        <div style={{ marginBottom: '20px' }}>
                            <label style={{ fontSize: '12px', color: '#aaa' }}>ACTIVE CAMERA VIEW</label>
                            <select 
                                value={activeCamId} 
                                onChange={(e) => setActiveCamId(e.target.value)}
                                style={{ width: '100%', marginTop: '8px', padding: '10px', background: '#222', color: '#fff', border: '1px solid #444' }}
                            >
                                {cameras.map(cam => <option key={cam.id} value={cam.id}>{cam.id.toUpperCase()}</option>)}
                            </select>
                        </div>
                    )}

                    <div>
                        <label style={{ fontSize: '13px', fontWeight: 'bold' }}>Active Cam Tilt: {activeCamera.tilt}°</label>
                        <input 
                            type="range" min="-90" max="0" 
                            value={activeCamera.tilt} 
                            onChange={(e) => {
                                const val = parseInt(e.target.value);
                                setCameras(cameras.map(c => c.id === activeCamId ? {...c, tilt: val} : c));
                            }} 
                            style={{ width: '100%', marginTop: '8px' }} 
                        />
                        
                        <label style={{marginTop: '15px', display: 'block', fontSize: '13px', fontWeight: 'bold' }}>Active Cam Height: {activeCamera.height.toFixed(1)}m</label>
                        <input 
                            type="range" min="1" max="15" step="0.1" 
                            value={activeCamera.height} 
                            onChange={(e) => {
                                const val = parseFloat(e.target.value);
                                setCameras(cameras.map(c => c.id === activeCamId ? {...c, height: val} : c));
                            }} 
                            style={{ width: '100%', marginTop: '8px' }} 
                        />

                        <button 
                            onClick={startAnalysis}
                            style={{ width: '100%', marginTop: '30px', padding: '16px', background: '#4caf50', border: 'none', color: '#fff', cursor: 'pointer', fontWeight: 'bold', borderRadius: '8px', fontSize: '16px' }}>
                            START ANALYSIS
                        </button>
                    </div>
                </div>
            ) : (
                <div>
                    <div style={{ background: '#2e7d32', padding: '12px', borderRadius: '8px', marginBottom: '20px', textAlign: 'center' }}>
                        <div style={{ fontSize: '12px', textTransform: 'uppercase', opacity: 0.8 }}>Current Match Score</div>
                        <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{events[events.length-1].score}</div>
                    </div>

                    <h3 style={{ fontSize: '16px', color: '#aaa', textTransform: 'uppercase' }}>Detected Events</h3>
                    <div style={{ marginTop: '10px' }}>
                        {events.map(ev => (
                            <div 
                                key={ev.id} 
                                onClick={() => seekTo(ev.time)}
                                style={{ 
                                    padding: '12px', background: '#252525', marginBottom: '8px', borderRadius: '6px', 
                                    cursor: 'pointer', borderLeft: '4px solid #4caf50', display: 'flex', justifyContent: 'space-between',
                                    transition: 'background 0.2s'
                                }}
                            >
                                <span>{ev.label}</span>
                                <span style={{ color: '#4caf50', fontWeight: 'bold' }}>{ev.score}</span>
                            </div>
                        ))}
                    </div>
                    
                    <button 
                        onClick={() => setEvents([])}
                        style={{ width: '100%', marginTop: '20px', padding: '10px', background: '#444', border: 'none', color: '#fff', cursor: 'pointer', borderRadius: '4px' }}>
                        Reset & Re-calibrate
                    </button>
                </div>
            )}
        </div>
      </div>
    </div>
  );
};

export default App;
