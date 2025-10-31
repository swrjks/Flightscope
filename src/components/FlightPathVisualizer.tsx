import { useRef, useEffect, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Line } from "@react-three/drei";
import * as THREE from "three";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, Pause, RotateCcw, Plane, Sun, Moon } from "lucide-react";

interface FlightSegment {
  start: number;
  end: number;
  duration: number;
  regime: string;
}

interface FlightPathVisualizerProps {
  segments: FlightSegment[];
}

// Aircraft component
function Aircraft({
  position,
  rotation,
}: {
  position: [number, number, number];
  rotation: [number, number, number];
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      // Subtle hovering animation
      meshRef.current.position.y =
        position[1] + Math.sin(state.clock.elapsedTime * 2) * 0.1;
    }
  });

  return (
    <mesh ref={meshRef} position={position} rotation={rotation}>
      <coneGeometry args={[0.3, 1.2, 8]} />
      <meshStandardMaterial color="#3b82f6" />
      {/* Wings */}
      <mesh position={[0, 0, 0.2]} rotation={[0, 0, 0]}>
        <boxGeometry args={[1.5, 0.1, 0.3]} />
        <meshStandardMaterial color="#1e40af" />
      </mesh>
      {/* Tail */}
      <mesh position={[0, 0.2, -0.4]} rotation={[0, 0, 0]}>
        <boxGeometry args={[0.6, 0.4, 0.1]} />
        <meshStandardMaterial color="#1e40af" />
      </mesh>
    </mesh>
  );
}

// Runway component
function Runway() {
  const runwayLength = 20;
  const runwayWidth = 2;

  const stripes = [];
  for (let i = -runwayLength / 2; i < runwayLength / 2; i += 2) {
    stripes.push(
      <mesh key={i} position={[i, 0.01, 0]}>
        <boxGeometry args={[1, 0.02, 0.2]} />
        <meshStandardMaterial color="#fbbf24" />
      </mesh>
    );
  }

  return (
    <group>
      <mesh position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[runwayLength, runwayWidth]} />
        <meshStandardMaterial color="#374151" />
      </mesh>
      {stripes}
    </group>
  );
}

// Flight path animation component
function AnimatedFlightPath({
  isPlaying,
  currentStep,
  onStepUpdate,
  resetTrigger,
  segments,
}: {
  isPlaying: boolean;
  currentStep: number;
  onStepUpdate: (step: number) => void;
  resetTrigger: number;
  segments: FlightSegment[];
}) {
  const [progress, setProgress] = useState(0);

  // Reset when triggered
  useEffect(() => {
    setProgress(0);
  }, [resetTrigger]);

  // Map regimes to 3D positions (simple layout)
  const regimePositions: Record<string, [number, number, number]> = {
    Taxi: [-8, 0.2, 0],
    Takeoff: [-6, 1, 0],
    Climb: [-3, 5, 0],
    Cruise: [0, 10, 5],
    Descent: [3, 5, 10],
    Landing: [6, 0.2, 12],
  };

  const flightPath = segments.map((seg, i) => ({
    position: regimePositions[seg.regime] || [i * 2, i, 0],
    rotation: [0, 0, 0] as [number, number, number],
    phase: seg.regime,
  }));

  useFrame((state, delta) => {
    if (isPlaying && progress < flightPath.length - 1) {
      const newProgress = progress + delta * 0.8;
      setProgress(Math.min(newProgress, flightPath.length - 1));

      const newStep = Math.floor(newProgress);
      if (newStep !== currentStep) {
        onStepUpdate(newStep);
      }
    }
  });

  const getInterpolatedPosition = (progress: number) => {
    const currentIndex = Math.floor(progress);
    const nextIndex = Math.min(currentIndex + 1, flightPath.length - 1);
    const t = progress - currentIndex;

    const current = flightPath[currentIndex];
    const next = flightPath[nextIndex];

    return {
      position: [
        THREE.MathUtils.lerp(current.position[0], next.position[0], t),
        THREE.MathUtils.lerp(current.position[1], next.position[1], t),
        THREE.MathUtils.lerp(current.position[2], next.position[2], t),
      ] as [number, number, number],
      rotation: [0, 0, 0] as [number, number, number],
      phase: current.phase,
    };
  };

  const interpolated = getInterpolatedPosition(progress);

  const currentPathPoints: THREE.Vector3[] = [];
  for (let i = 0; i <= progress; i += 0.1) {
    const point = getInterpolatedPosition(
      Math.min(i, flightPath.length - 1)
    );
    currentPathPoints.push(new THREE.Vector3(...point.position));
  }

  return (
    <group>
      <Aircraft
        position={interpolated.position}
        rotation={interpolated.rotation}
      />
      {currentPathPoints.length > 1 && (
        <Line points={currentPathPoints} color="#3b82f6" lineWidth={3} />
      )}
      <Text
        position={[
          interpolated.position[0],
          interpolated.position[1] + 2,
          interpolated.position[2],
        ]}
        fontSize={0.5}
        color="#3b82f6"
        anchorX="center"
        anchorY="middle"
      >
        {interpolated.phase}
      </Text>
    </group>
  );
}

function FlightScene({
  isPlaying,
  currentStep,
  onStepUpdate,
  isDayMode,
  resetTrigger,
  segments,
}: {
  isPlaying: boolean;
  currentStep: number;
  onStepUpdate: (step: number) => void;
  isDayMode: boolean;
  resetTrigger: number;
  segments: FlightSegment[];
}) {
  return (
    <>
      <ambientLight intensity={isDayMode ? 0.8 : 0.3} />
      <directionalLight
        position={[10, 10, 5]}
        intensity={isDayMode ? 1.5 : 0.6}
      />
      <mesh position={[0, -0.1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[50, 50]} />
        <meshStandardMaterial color={isDayMode ? "#4ade80" : "#1f2937"} />
      </mesh>
      <Runway />
      <AnimatedFlightPath
        isPlaying={isPlaying}
        currentStep={currentStep}
        onStepUpdate={onStepUpdate}
        resetTrigger={resetTrigger}
        segments={segments}
      />
      <OrbitControls target={[0, 5, 0]} />
    </>
  );
}

export function FlightPathVisualizer({ segments }: FlightPathVisualizerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [isDayMode, setIsDayMode] = useState(true);
  const [resetTrigger, setResetTrigger] = useState(0);

  const handlePlayPause = () => setIsPlaying(!isPlaying);
  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
    setResetTrigger((prev) => prev + 1);
  };
  const toggleDayNight = () => setIsDayMode(!isDayMode);

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card className="aviation-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Plane className="h-5 w-5 text-primary" />
            Flight Path Simulation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button onClick={handlePlayPause} className="flight-button flex items-center gap-2">
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {isPlaying ? "Pause" : "Play"}
              </Button>
              <Button onClick={handleReset} variant="outline" className="flex items-center gap-2">
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
              <Button onClick={toggleDayNight} variant="outline" className="flex items-center gap-2">
                {isDayMode ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                {isDayMode ? "Night" : "Day"}
              </Button>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline">
                Phase {currentStep + 1}/{segments.length}
              </Badge>
              <Badge className="bg-primary text-primary-foreground">
                {segments[currentStep]?.regime}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 3D Visualization */}
      <Card className="aviation-card">
        <CardContent className="p-0">
          <div className="h-[600px] w-full rounded-lg overflow-hidden">
            <Canvas
              camera={{ position: [15, 10, 15], fov: 60 }}
              shadows
              style={{
                background: isDayMode
                  ? "linear-gradient(to bottom, #87ceeb, #e0f6ff)"
                  : "linear-gradient(to bottom, #1e293b, #0f172a)",
              }}
            >
              <FlightScene
                isPlaying={isPlaying}
                currentStep={currentStep}
                onStepUpdate={setCurrentStep}
                isDayMode={isDayMode}
                resetTrigger={resetTrigger}
                segments={segments}
              />
            </Canvas>
          </div>
        </CardContent>
      </Card>

      {/* Flight phases list */}
      <Card className="aviation-card">
        <CardHeader>
          <CardTitle>Flight Phases</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {segments.map((seg, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg border ${
                  index === currentStep
                    ? "border-primary bg-primary/10"
                    : index < currentStep
                    ? "border-green-500 bg-green-500/10"
                    : "border-border"
                }`}
              >
                <div className="text-sm font-medium">{seg.regime}</div>
                <div className="text-xs text-muted-foreground mt-1">
                  Step {index + 1}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
