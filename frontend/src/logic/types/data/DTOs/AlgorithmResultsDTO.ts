import { RobotDetectionDTO } from "./RobotDetectionDTO";
import { TopDownDetectionDTO } from "./TopDownDetectionDTO";

export type CalibrationTelemetryDTO = {
  useEpipolarGeometry: boolean;
  status: string;
  error: string | null;
  hasCalibrations: boolean;
};

export type AlgorithmResultsDTO = {
  fps: number;
  topDownDetections: TopDownDetectionDTO[];
  detections: RobotDetectionDTO[];
  /** True when the main algorithm loop failed; displayed data is stale */
  dataStale?: boolean;
  calibration?: CalibrationTelemetryDTO;
};
