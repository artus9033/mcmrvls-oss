import {AlgorithmResultsDTO} from '../types/data/DTOs/AlgorithmResultsDTO';
import {AlgorithmResults} from '../types/data/parsed/AlgorithmResults';
import {arrayBufferToImageSrcB64} from '../utils';
import {AlgorithmPreviewsDTO} from '../types/data/DTOs/AlgorithmPreviewsDTO';
import {StitchingPreviewsDTO} from '../types/data/DTOs/StitchingPreviewsDTO';
import {AlgorithmPreviews} from '../types/data/parsed/AlgorithmPreviews';
import {StitchingPreviews} from '../types/data/parsed/StitchingPreviews';
import {StoreSliceCreator} from '.';
import {MapSegmentationPreviewsDTO} from '../types/data/DTOs/MapSegmentationPreviewsDTO';
import {MapSegmentationPreviews} from '../types/data/parsed/MapSegmentationPreviews';
import {CameraPreviewsDTO} from '../types/data/DTOs/CameraPreviewsDTO';
import {CameraPreviews} from '../types/data/parsed/CameraPreviews';

export interface ResultsSlice {
  // parsed results
  algorithmResults: AlgorithmResults;
  algorithmPreviews: AlgorithmPreviews;
  stitchingPreviews: StitchingPreviews;
  cameraPreviews: CameraPreviews;
  mapSegmentationPreviews: MapSegmentationPreviews;
  lastDataTimestamp: number | null;

  // results mutators
  detectionsReceived: (data: AlgorithmResultsDTO) => void;
  algorithmPreviewsReceived: (data: AlgorithmPreviewsDTO) => void;
  stitchingPreviewsReceived: (data: StitchingPreviewsDTO) => void;
  cameraPreviewsReceived: (data: CameraPreviewsDTO) => void;
  mapSegmentationPreviewsReceived: (data: MapSegmentationPreviewsDTO) => void;
}

// @ts-ignore-next-line
export const createResultsSlice: StoreSliceCreator<ResultsSlice> = (
  set,
  get,
) => ({
  // parsed results
  algorithmResults: {
    fps: NaN,
    detections: [],
    topDownDetections: [],
    dataStale: true,
    calibration: undefined,
  },
  algorithmPreviews: {
    topDownImgStrJpeg: null,
    topDownImgAnnotatedStrJpeg: null,
    stitchedImgStrJpeg: null,
    stitchedImgAnnotatedStrJpeg: null,
  },
  stitchingPreviews: {
    commonFeaturesJPEGsMap: {},
  },
  cameraPreviews: {
    camerasJPEGsMap: {},
  },
  mapSegmentationPreviews: {
    redStopLinesStrJpeg: null,
    yellowLinesStrJpeg: null,
    roadComponentsStrJpeg: null,
  },
  lastDataTimestamp: null,

  // results mutators
  detectionsReceived(data: AlgorithmResultsDTO) {
    return set((state) => {
      state.results.algorithmResults.fps = data.fps;
      state.results.algorithmResults.detections = data.detections;
      state.results.algorithmResults.topDownDetections = data.topDownDetections;
      state.results.algorithmResults.dataStale = data.dataStale ?? true;
      if (data.calibration !== undefined) {
        state.results.algorithmResults.calibration = data.calibration;
      }

      state.results.lastDataTimestamp = Date.now();

      return state;
    });
  },
  algorithmPreviewsReceived(data: AlgorithmPreviewsDTO) {
    set((state) => {
      state.results.algorithmPreviews.topDownImgStrJpeg =
        arrayBufferToImageSrcB64(data.topDownImg, 'jpeg');
      state.results.algorithmPreviews.topDownImgAnnotatedStrJpeg =
        arrayBufferToImageSrcB64(data.topDownImgAnnotated, 'jpeg');
      state.results.algorithmPreviews.stitchedImgStrJpeg =
        arrayBufferToImageSrcB64(data.stitchedImg, 'jpeg');
      state.results.algorithmPreviews.stitchedImgAnnotatedStrJpeg =
        arrayBufferToImageSrcB64(data.stitchedImgAnnotated, 'jpeg');

      state.results.lastDataTimestamp = Date.now();

      return state;
    });
  },
  stitchingPreviewsReceived(data) {
    set((state) => {
      state.results.stitchingPreviews.commonFeaturesJPEGsMap = {};
      for (const [stitchingStage, jpegBytes] of Object.entries(
        data.commonFeaturesJPEGsMap,
      )) {
        state.results.stitchingPreviews.commonFeaturesJPEGsMap[stitchingStage] =
          arrayBufferToImageSrcB64(jpegBytes, 'jpeg');
      }

      state.results.lastDataTimestamp = Date.now();

      return state;
    });
  },
  cameraPreviewsReceived(data) {
    set((state) => {
      state.results.cameraPreviews.camerasJPEGsMap = {};
      for (const [camera, jpegBytes] of Object.entries(data)) {
        state.results.cameraPreviews.camerasJPEGsMap[camera] =
          arrayBufferToImageSrcB64(jpegBytes, 'jpeg');
      }

      state.results.lastDataTimestamp = Date.now();

      return state;
    });
  },
  mapSegmentationPreviewsReceived(data) {
    set((state) => {
      state.results.mapSegmentationPreviews.redStopLinesStrJpeg =
        arrayBufferToImageSrcB64(data.redStopLines, 'jpeg');
      state.results.mapSegmentationPreviews.yellowLinesStrJpeg =
        arrayBufferToImageSrcB64(data.yellowLines, 'jpeg');
      state.results.mapSegmentationPreviews.roadComponentsStrJpeg =
        arrayBufferToImageSrcB64(data.roadComponents, 'jpeg');
    });
  },
});
