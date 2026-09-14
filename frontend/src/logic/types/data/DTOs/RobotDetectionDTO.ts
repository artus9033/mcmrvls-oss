import { RobotConfigDTO } from "./RobotConfigDTO";

export type RobotDetectionDTO = {
  /** The robot */
  robot: RobotConfigDTO;
  /** Relative X coordinate */
  x: number;
  /** Relative Y coordinate */
  y: number;
  /** Azimuth in degrees, w.r.t. North (0deg), counted clockwise */
  azimuth: number;
  /**
   * Whether the pose comes from a live AprilTag detection or from the Kalman filter
   * during the grace period after the tag was lost.
   */
  poseSource: "detection" | "estimation";
  /** Unfiltered tag measurement (same frame); omitted when pose is estimation-only */
  raw_x?: number;
  raw_y?: number;
  raw_azimuth?: number;
  /**
   * Measurement-noise diagnostics for the same frame: the R diagonal the filter used
   * (sigmas in normalized map units / degrees) and the normalized tag geometry it was
   * derived from. Present only on live detections; consumed by the offline Kalman
   * calibration tooling, not by the UI.
   */
  meas_sigma_x?: number;
  meas_sigma_y?: number;
  meas_sigma_psi_deg?: number;
  meas_area_norm?: number;
  meas_baseline_norm?: number;
};
