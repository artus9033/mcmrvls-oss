export type CameraPreviewsDTO = {
  /** stores JPEG bytes or nulls for each of the cameras */
  [camera: string]: ArrayBuffer | null;
};
