import { useState, useCallback } from "react";
import { Upload, FileText, CheckCircle, AlertCircle } from "lucide-react";
import Papa from "papaparse";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";

interface CSVUploaderProps {
  onDataLoaded: (data: any[], headers: string[]) => void;
}

export function CSVUploader({ onDataLoaded }: CSVUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [fileName, setFileName] = useState<string>("");
  const { toast } = useToast();

  const processFile = useCallback((file: File) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      toast({
        title: "Invalid file type",
        description: "Please upload a CSV file.",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setFileName(file.name);

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.errors.length > 0) {
          console.error("CSV parsing errors:", results.errors);
          setUploadStatus('error');
          toast({
            title: "Parsing error",
            description: "There was an error reading your CSV file.",
            variant: "destructive",
          });
        } else {
          setUploadStatus('success');
          const headers = results.meta.fields || [];
          onDataLoaded(results.data, headers);
          toast({
            title: "Upload successful",
            description: `Loaded ${results.data.length} flight data records.`,
          });
        }
        setIsProcessing(false);
      },
      error: (error) => {
        console.error("CSV parsing error:", error);
        setUploadStatus('error');
        setIsProcessing(false);
        toast({
          title: "Upload failed",
          description: "Failed to process the CSV file.",
          variant: "destructive",
        });
      }
    });
  }, [onDataLoaded, toast]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      processFile(files[0]);
    }
  }, [processFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      processFile(files[0]);
    }
  }, [processFile]);

  const getStatusIcon = () => {
    if (isProcessing) return <Upload className="h-8 w-8 animate-pulse text-primary" />;
    if (uploadStatus === 'success') return <CheckCircle className="h-8 w-8 text-green-500" />;
    if (uploadStatus === 'error') return <AlertCircle className="h-8 w-8 text-destructive" />;
    return <FileText className="h-8 w-8 text-muted-foreground" />;
  };

  return (
    <Card className="aviation-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5 text-primary" />
          Flight Data Upload
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            isDragging
              ? "border-primary bg-primary/10"
              : uploadStatus === 'success'
              ? "border-green-500 bg-green-500/10"
              : uploadStatus === 'error'
              ? "border-destructive bg-destructive/10"
              : "border-border hover:border-primary/50"
          }`}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onDragEnter={() => setIsDragging(true)}
          onDragLeave={() => setIsDragging(false)}
        >
          <div className="flex flex-col items-center gap-4">
            {getStatusIcon()}
            
            <div>
              <h3 className="font-medium mb-2">
                {isProcessing
                  ? "Processing flight data..."
                  : uploadStatus === 'success'
                  ? `Successfully loaded: ${fileName}`
                  : uploadStatus === 'error'
                  ? "Upload failed"
                  : "Drop your CSV file here"}
              </h3>
              
              <p className="text-sm text-muted-foreground mb-4">
                {isProcessing
                  ? "Please wait while we parse your flight data"
                  : uploadStatus === 'success'
                  ? "Flight data is ready for analysis"
                  : "Upload CSV files containing flight data (altitude, speed, coordinates, etc.)"}
              </p>
            </div>

            {uploadStatus !== 'success' && !isProcessing && (
              <div className="space-y-2">
                <Button className="flight-button" asChild>
                  <label className="cursor-pointer">
                    Select CSV File
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </label>
                </Button>
                
                <p className="text-xs text-muted-foreground">
                  Expected columns: timestamp, latitude, longitude, altitude, speed, heading
                </p>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}