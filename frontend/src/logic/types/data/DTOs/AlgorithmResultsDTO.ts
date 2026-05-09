import { RobotDetectionDTO } from "./RobotDetectionDTO";
import { TopDownDetectionDTO } from "./TopDownDetectionDTO";

export type AlgorithmResultsDTO = {
  fps: number;
  topDownDetections: TopDownDetectionDTO[];
  detections: RobotDetectionDTO[];
  /** True when the main algorithm loop failed; displayed data is stale */
  dataStale?: boolean;
};
