import { useState } from "react";
import { CSVUploader } from "@/components/CSVUploader";
import { analyzeFlightRegimes, FlightSegment } from "@/lib/regimeAnalyzer";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { DataAnalysis } from "@/components/DataAnalysis";

const Index = () => {
  const [flightData, setFlightData] = useState<any[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [segments, setSegments] = useState<FlightSegment[]>([]);

  const handleDataLoaded = (data: any[], csvHeaders: string[]) => {
    setFlightData(data);
    setHeaders(csvHeaders);

    // ✅ run regime analysis when data loads
    const segs = analyzeFlightRegimes(data, "label");
    setSegments(segs);
  };

  return (
    <div className="page-background">
      <div className="space-y-6 p-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            Flight Data Analysis
          </h1>
          <p className="text-muted-foreground">
            Upload your CSV flight data to get comprehensive analytics and insights.
          </p>
        </div>

        {/* CSV Upload */}
        <CSVUploader onDataLoaded={handleDataLoaded} />

        {/* Metrics Cards + Graphs */}
        {flightData.length > 0 && (
          <DataAnalysis data={flightData} headers={headers} />
        )}

        {/* Regime Summary Table */}
        {segments.length > 0 && (
          <Card className="aviation-card">
            <CardHeader>
              <CardTitle>Flight Regime Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="min-w-full border border-border">
                  <thead className="bg-muted">
                    <tr>
                      <th className="px-3 py-2 border">Time Range</th>
                      <th className="px-3 py-2 border">Duration</th>
                      <th className="px-3 py-2 border">Regime</th>
                    </tr>
                  </thead>
                  <tbody>
                    {segments.map((seg, idx) => (
                      <tr key={idx} className="text-center hover:bg-muted/50">
                        <td className="border px-3 py-2">{seg.timeRange}</td>
                        <td className="border px-3 py-2">{seg.durationFormatted}</td>
                        <td className="border px-3 py-2 font-medium">{seg.regime}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default Index;
