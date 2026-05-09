export type StitchingPreviewsDTO = {
  /** stores JPEG bytes or nulls for each of the stitching stages (names of cameras in stitching pairs) */
  commonFeaturesJPEGsMap: {
    [stitchingStageDescription: string]: ArrayBuffer | null;
  };
};
