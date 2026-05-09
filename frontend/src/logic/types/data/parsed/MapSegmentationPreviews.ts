import { MapSegmentationPreviewsDTO } from "../DTOs/MapSegmentationPreviewsDTO";

export type MapSegmentationPreviews = Record<
  `${keyof MapSegmentationPreviewsDTO}StrJpeg`,
  string | null
>;
