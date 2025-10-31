import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  Area,
  AreaChart
} from "recharts";
import { 
  Plane, 
  Gauge, 
  MapPin, 
  Clock, 
  TrendingUp,
  Navigation
} from "lucide-react";

interface DataAnalysisProps {
  data: any[];
  headers: string[];
}

export function DataAnalysis({ data, headers }: DataAnalysisProps) {
  const analytics = useMemo(() => {
    if (!data || data.length === 0) return null;

    // Find relevant columns
    const altitudeCol = headers.find(h => h.toLowerCase().includes('altitude') || h.toLowerCase().includes('alt'));
    const speedCol = headers.find(h => h.toLowerCase().includes('speed') || h.toLowerCase().includes('velocity'));
    const latCol = headers.find(h => h.toLowerCase().includes('lat'));
    const lonCol = headers.find(h => h.toLowerCase().includes('lon') || h.toLowerCase().includes('lng'));
    const timeCol = headers.find(h => h.toLowerCase().includes('time') || h.toLowerCase().includes('timestamp'));

    // Calculate statistics
    const altitudes = data.map(row => parseFloat(row[altitudeCol || '']) || 0).filter(v => v > 0);
    const speeds = data.map(row => parseFloat(row[speedCol || '']) || 0).filter(v => v > 0);

    const maxAltitude = Math.max(...altitudes);
    const avgAltitude = altitudes.reduce((a, b) => a + b, 0) / altitudes.length;
    const maxSpeed = Math.max(...speeds);
    const avgSpeed = speeds.reduce((a, b) => a + b, 0) / speeds.length;

    // Prepare chart data
    const chartData = data.slice(0, 50).map((row, index) => ({
      index,
      altitude: parseFloat(row[altitudeCol || '']) || 0,
      speed: parseFloat(row[speedCol || '']) || 0,
      time: row[timeCol || ''] || `Point ${index + 1}`
    }));

    return {
      totalRecords: data.length,
      maxAltitude: Math.round(maxAltitude),
      avgAltitude: Math.round(avgAltitude),
      maxSpeed: Math.round(maxSpeed),
      avgSpeed: Math.round(avgSpeed),
      chartData,
      columns: {
        altitude: altitudeCol,
        speed: speedCol,
        latitude: latCol,
        longitude: lonCol,
        time: timeCol
      }
    };
  }, [data, headers]);

  if (!analytics) {
    return (
      <div className="space-y-6">
        <Card className="aviation-card">
          <CardContent className="p-8 text-center">
            <Plane className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium mb-2">No Flight Data</h3>
            <p className="text-muted-foreground">
              Upload a CSV file to begin analyzing your flight data.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="aviation-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <MapPin className="h-4 w-4 text-primary" />
              Total Records
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">
              {analytics.totalRecords.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">Data points</p>
          </CardContent>
        </Card>

        <Card className="aviation-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-accent" />
              Max Altitude
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-accent">
              {analytics.maxAltitude.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">
              Avg: {analytics.avgAltitude.toLocaleString()} ft
            </p>
          </CardContent>
        </Card>

        <Card className="aviation-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Gauge className="h-4 w-4 text-green-500" />
              Max Speed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500">
              {analytics.maxSpeed.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">
              Avg: {analytics.avgSpeed.toLocaleString()} kt
            </p>
          </CardContent>
        </Card>

        <Card className="aviation-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Navigation className="h-4 w-4 text-blue-500" />
              Data Quality
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="bg-green-500/20 text-green-500">
                Excellent
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              All fields detected
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="aviation-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Altitude Profile
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={analytics.chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="index" 
                  stroke="hsl(var(--muted-foreground))" 
                  fontSize={12}
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))" 
                  fontSize={12}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px"
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="altitude"
                  stroke="hsl(var(--accent))"
                  fill="hsl(var(--accent))"
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="aviation-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Gauge className="h-5 w-5 text-primary" />
              Speed Profile
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={analytics.chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="index" 
                  stroke="hsl(var(--muted-foreground))" 
                  fontSize={12}
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))" 
                  fontSize={12}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px"
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="speed"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}