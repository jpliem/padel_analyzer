import React, { useState, useEffect, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, TransformControls, Box, Plane, Grid, Sphere } from '@react-three/drei';
import * as THREE from 'three';

// 1. Digital Twin Court (20m x 10m)
const PadelCourt = () => {
  return (
    <group>
      {/* Turf Area (Green) */}
      <Plane args={[10, 20]} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <meshStandardMaterial color="#2e7d32" />
      </Plane>
      {/* Lines (White) */}
      <Grid args={[10, 20]} position={[0, 0.01, 0]} rotation={[0, 0, 0]} sectionColor="white" cellColor="white" />
      {/* Net */}
      <Box args={[10, 0.88, 0.1]} position={[0, 0.44, 0]}>
        <meshStandardMaterial color="#eeeeee" transparent opacity={0.5} />
      </Box>
    </group>
  );
};

// 2. Interactive Virtual Camera Component
const VirtualCamera = ({ config, setConfig }: { config: any, setConfig: (c: any) => void }) => {
  const meshRef = useRef<THREE.Mesh>(null!);

  const onDrag = () => {
    if (meshRef.current) {
      const pos = meshRef.current.position;
      setConfig({
        ...config,
        pos_x: pos.x,
        pos_y: pos.z,
        height: pos.y
      });
    }
  };

  return (
    <TransformControls 
      position={[config.pos_x, config.height, config.pos_y]} 
      mode="translate" 
      onMouseUp={onDrag}
    >
      <Sphere args={[0.3]} ref={meshRef}>
        <meshStandardMaterial color="yellow" />
      </Sphere>
    </TransformControls>
  );
};

const SetupDashboard = () => {
  const [camera, setCamera] = useState({
    id: 'cam1',
    pos_x: 5,
    pos_y: 22, // Behind the baseline
    height: 5,
    tilt: -30,
    pan: 0
  });

  return (
    <div style={{ display: 'flex', height: '100vh', background: '#1a1a1a' }}>
      {/* Left: 3D Minimap */}
      <div style={{ flex: 1, borderRight: '1px solid #444' }}>
        <Canvas camera={{ position: [20, 20, 20] }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />
          <PadelCourt />
          <VirtualCamera config={camera} setConfig={setCamera} />
          <OrbitControls makeDefault />
        </Canvas>
      </div>

      {/* Right: Controls & Preview */}
      <div style={{ width: '400px', padding: '20px', color: 'white', overflowY: 'auto' }}>
        <h2>Padel Setup</h2>
        <p>Drag the <b>Yellow Sphere</b> to position your camera. Use the sliders to tilt and pan.</p>
        
        <div style={{ marginTop: '20px' }}>
          <label>Height (Meters): {camera.height.toFixed(1)}</label>
          <input 
            type="range" min="0" max="10" step="0.1" 
            value={camera.height} 
            onChange={(e) => setCamera({...camera, height: parseFloat(e.target.value)})} 
            style={{ width: '100%' }}
          />
        </div>

        <div style={{ marginTop: '20px' }}>
          <label>Tilt Angle: {camera.tilt}°</label>
          <input 
            type="range" min="-90" max="0" 
            value={camera.tilt} 
            onChange={(e) => setCamera({...camera, tilt: parseInt(e.target.value)})} 
            style={{ width: '100%' }}
          />
        </div>

        <div style={{ marginTop: '30px', padding: '15px', background: '#333', borderRadius: '8px' }}>
            <h3>Projection Config (JSON)</h3>
            <pre style={{ fontSize: '12px' }}>{JSON.stringify(camera, null, 2)}</pre>
        </div>

        <button 
          onClick={() => alert("Calibration Saved! The CV models will now use this math.")}
          style={{ width: '100%', marginTop: '20px', padding: '10px', background: '#4caf50', color: 'white', border: 'none', cursor: 'pointer' }}>
          ACTIVATE ANALYZER
        </button>
      </div>
    </div>
  );
};

export default SetupDashboard;
