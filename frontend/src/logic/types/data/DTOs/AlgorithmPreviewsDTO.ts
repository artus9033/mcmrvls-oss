export type AlgorithmPreviewsDTO = Record<
  "topDownImg" | "topDownImgAnnotated" | "stitchedImg" | "stitchedImgAnnotated",
  ArrayBuffer | null
>;
