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
};
