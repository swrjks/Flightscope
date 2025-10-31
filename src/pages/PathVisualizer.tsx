import { useState } from "react";
import { CSVUploader } from "@/components/CSVUploader";
import { FlightPathVisualizer } from "@/components/FlightPathVisualizer";
import { analyzeFlightRegimes, FlightSegment } from "@/lib/regimeAnalyzer";

export default function PathVisualizer() {
  const [segments, setSegments] = useState<FlightSegment[]>([]);
  const [flightData, setFlightData] = useState<any[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);

  const handleDataLoaded = (data: any[], csvHeaders: string[]) => {
    setFlightData(data);
    setHeaders(csvHeaders);

    // ✅ Run regime analysis
    const segs = analyzeFlightRegimes(data, "label");
    setSegments(segs);
  };

  return (
    <div className="page-background">
      <div className="space-y-6 p-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            Flight Path Visualizer
          </h1>
          <p className="text-muted-foreground">
            Watch your flight data come to life in a simulation showing takeoff, cruise, descent, and landing.
          </p>
        </div>

        {/* Upload CSV */}
        <CSVUploader onDataLoaded={handleDataLoaded} />

        {/* Show visualization only after upload */}
        {segments.length > 0 && (
          <FlightPathVisualizer segments={segments} />
        )}
      </div>
    </div>
  );
}
