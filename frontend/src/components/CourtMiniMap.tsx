import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Plane, Grid, Box } from '@react-three/drei';

interface PlayerDot {
  id: string;
  x: number;
  y: number;
  team: 'A' | 'B';
}

interface Props {
  players?: PlayerDot[];
  ballPosition?: { x: number; y: number } | null;
  ballTrail?: { x: number; y: number }[];
  height?: number;
}

const Court: React.FC = () => (
  <group>
    <Plane args={[10, 20]} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
      <meshStandardMaterial color="#1b5e20" />
    </Plane>
    <Grid args={[10, 20]} position={[0, 0.01, 0]} sectionColor="white" cellColor="#2e7d32" />
    <Box args={[10, 0.88, 0.05]} position={[0, 0.44, 0]}>
      <meshStandardMaterial color="white" transparent opacity={0.6} />
    </Box>
    <Box args={[10, 0.02, 0.02]} position={[0, 0.01, -3.05]}>
      <meshStandardMaterial color="white" />
    </Box>
    <Box args={[10, 0.02, 0.02]} position={[0, 0.01, 3.05]}>
      <meshStandardMaterial color="white" />
    </Box>
    <Box args={[0.02, 0.02, 6.1]} position={[0, 0.01, 0]}>
      <meshStandardMaterial color="white" />
    </Box>
  </group>
);

const CourtMiniMap: React.FC<Props> = ({ players = [], ballPosition, ballTrail = [], height = 200 }) => (
  <div style={{ height, background: '#1a1a2e', borderRadius: 8, overflow: 'hidden' }}>
    <Canvas camera={{ position: [0, 18, 12], fov: 50 }}>
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 10, 5]} intensity={0.4} />
      <Court />
      {players.map(p => (
        <mesh key={p.id} position={[p.x - 5, 0.2, p.y - 10]}>
          <sphereGeometry args={[0.2]} />
          <meshStandardMaterial color={p.team === 'A' ? '#74b9ff' : '#e17055'} />
        </mesh>
      ))}
      {ballPosition && (
        <mesh position={[ballPosition.x - 5, 0.3, ballPosition.y - 10]}>
          <sphereGeometry args={[0.12]} />
          <meshStandardMaterial color="#fdcb6e" emissive="#fdcb6e" emissiveIntensity={0.5} />
        </mesh>
      )}
      {ballTrail.map((p, i) => (
        <mesh key={i} position={[p.x - 5, 0.15, p.y - 10]}>
          <sphereGeometry args={[0.08]} />
          <meshStandardMaterial color="#fdcb6e" transparent opacity={0.2 + (i / ballTrail.length) * 0.6} />
        </mesh>
      ))}
      <OrbitControls enablePan={false} enableZoom={false} />
    </Canvas>
  </div>
);

export default CourtMiniMap;
