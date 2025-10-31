export interface FlightSegment {
  start: number;       // start time in seconds
  end: number;         // end time in seconds
  duration: number;    // duration in seconds
  regime: string;      // predicted/actual regime
  timeRange: string;   // formatted time range (HH:MM:SS – HH:MM:SS)
  durationFormatted: string; // formatted duration (e.g., "10s", "2.5 min")
}

// helper to format seconds → HH:MM:SS
function formatTime(seconds: number): string {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return [hrs, mins, secs]
    .map((v) => String(v).padStart(2, "0"))
    .join(":");
}

// helper to format duration nicely
function formatDuration(seconds: number): string {
  if (seconds >= 60) {
    return `${(seconds / 60).toFixed(1)} min`;
  }
  return `${seconds.toFixed(1)} s`;
}

export function analyzeFlightRegimes(
  rows: any[],
  labelKey = "label"
): FlightSegment[] {
  if (!rows || rows.length === 0) return [];

  const segments: FlightSegment[] = [];
  let startIdx = 0;

  for (let i = 1; i < rows.length; i++) {
    if (rows[i][labelKey] !== rows[i - 1][labelKey]) {
      const start = parseFloat(rows[startIdx].time_s);
      const end = parseFloat(rows[i - 1].time_s);
      const duration = end - start;

      segments.push({
        start,
        end,
        duration,
        regime: rows[startIdx][labelKey],
        timeRange: `${formatTime(start)} – ${formatTime(end)}`,
        durationFormatted: formatDuration(duration),
      });

      startIdx = i;
    }
  }

  // last segment
  const start = parseFloat(rows[startIdx].time_s);
  const end = parseFloat(rows[rows.length - 1].time_s);
  const duration = end - start;

  segments.push({
    start,
    end,
    duration,
    regime: rows[startIdx][labelKey],
    timeRange: `${formatTime(start)} – ${formatTime(end)}`,
    durationFormatted: formatDuration(duration),
  });

  return segments;
}
