import SocketIOClient, {Socket} from 'socket.io-client';

import {DEV} from '../../constants';
import {AlgorithmResultsDTO} from '../types/data/DTOs/AlgorithmResultsDTO';
import {arrayBufferToImageSrcB64} from '../utils';
import {AlgorithmPreviewsDTO} from '../types/data/DTOs/AlgorithmPreviewsDTO';
import {StitchingPreviewsDTO} from '../types/data/DTOs/StitchingPreviewsDTO';
import {StoreSliceCreator} from '.';
import {MapSegmentationPreviewsDTO} from '../types/data/DTOs/MapSegmentationPreviewsDTO';
import {CameraPreviewsDTO} from '../types/data/DTOs/CameraPreviewsDTO';

export interface SocketioSlice {
  // socket state variables
  sioClient: Socket;
  isConnected: boolean;
  isConnecting: boolean;
  connectionError: Error | null;

  // socket connection state mutators
  connected: () => void;
  disconnected: (reason: Socket.DisconnectReason) => void;
  connectFailed: (error: Error) => void;
  reconnecting: () => void;
}

export const socketEndpoint = `http://${
  DEV ? `${window.location.hostname}:6001` : window.location.host
}`;

// @ts-ignore-next-line
export const createSocketioSlice: StoreSliceCreator<SocketioSlice> = (
  set,
  get,
) => ({
  // socket state variables
  sioClient: (() => {
    const client = SocketIOClient(socketEndpoint, {
      transports: ['websocket'],
      // fail a hanging handshake fast instead of waiting the default 20s, so
      // that a restarting backend is picked up on the next (quick) retry
      timeout: 4000,
      reconnectionDelay: 500,
      reconnectionDelayMax: 2000,
      randomizationFactor: 0.3,
    });

    client.on('disconnect', (reason: Socket.DisconnectReason) => {
      get().socketio.disconnected(reason);
    });

    client.on('connect', () => {
      get().socketio.connected();
    });

    client.on('connect_error', (error: Error) => {
      get().socketio.connectFailed(error);
    });

    client.on('reconnect_attempt', () => {
      get().socketio.reconnecting();
    });

    client.on('all_detections', (data: AlgorithmResultsDTO) => {
      get().results.detectionsReceived(data);
    });

    client.on(
      'system_status',
      (data: {calibration?: AlgorithmResultsDTO['calibration']}) => {
        if (!data.calibration) {
          return;
        }
        set((state) => {
          state.results.algorithmResults.calibration = data.calibration;
        });
      },
    );

    client.on('algorithm_previews', (data: AlgorithmPreviewsDTO) => {
      get().results.algorithmPreviewsReceived(data);
    });

    client.on('stitching_previews', (data: StitchingPreviewsDTO) => {
      get().results.stitchingPreviewsReceived(data);
    });

    client.on('camera_previews', (data: CameraPreviewsDTO) => {
      get().results.cameraPreviewsReceived(data);
    });

    client.on(
      'map_segmentation_previews',
      (data: MapSegmentationPreviewsDTO) => {
        get().results.mapSegmentationPreviewsReceived(data);
      },
    );

    return client;
  })(),
  isConnected: false,
  isConnecting: false,
  connectionError: null,

  // subscription state variables
  shouldSubscribeAlgorithmPreviews: false,
  shouldSubscribeStitchingPreviews: false,

  // parsed results
  algorithmResults: {
    fps: NaN,
    detections: [],
    topDownDetections: [],
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
  lastDataTimestamp: null,

  // socket connection state mutators
  connected() {
    set((state) => {
      // subscribe to all detections room
      state.socketio.sioClient.emit('subscribe_all_detections');

      // subscribe to stitching image previews room (if enabled)
      if (state.settings.shouldSubscribeAlgorithmPreviews) {
        // only execute if true, there's no need to send an unsubscribe
        // event since on start it's already unsubscribed
        state.settings.updateAlgorithmPreviewsSubscription(true);
      }

      // subscribe to stitching image previews room (if enabled)
      if (state.settings.shouldSubscribeStitchingPreviews) {
        // only execute if true, there's no need to send an unsubscribe
        // event since on start it's already unsubscribed
        state.settings.updateStitchingPreviewsSubscription(true);
      }

      // subscribe to camera image previews room (if enabled)
      if (state.settings.shouldSubscribeCameraPreviews) {
        // only execute if true, there's no need to send an unsubscribe
        // event since on start it's already unsubscribed
        state.settings.updateCameraPreviewsSubscription(true);
      }

      // subscribe to map segmentation image previews room (if enabled)
      if (state.settings.shouldSubscribeMapSegmentationPreviews) {
        // only execute if true, there's no need to send an unsubscribe
        // event since on start it's already unsubscribed
        state.settings.updateMapSegmentationPreviewsSubscription(true);
      }

      state.socketio.isConnected = true;
      state.socketio.connectionError = null;
      state.socketio.isConnecting = false;
    });
  },
  disconnected(reason) {
    set((state) => {
      state.socketio.isConnected = false;
      // the client retries on its own for every reason except an explicit
      // disconnect on either end, so go straight to 'connecting' instead of
      // showing 'disconnected' until the first reconnect_attempt lands
      state.socketio.isConnecting =
        reason !== 'io client disconnect' && reason !== 'io server disconnect';
    });
  },
  connectFailed(error) {
    set((state) => {
      state.socketio.connectionError = error;
      state.socketio.isConnected = false;
      // a failed attempt is followed by another one as long as reconnection is
      // still active; keeping isConnecting set avoids the status chip flipping
      // to 'Disconnected' and back between attempts
      state.socketio.isConnecting = state.socketio.sioClient.active;
    });
  },
  reconnecting() {
    set((state) => {
      state.socketio.isConnecting = true;
    });
  },

  // results mutators
  detectionsReceived(data: AlgorithmResultsDTO) {
    set((state) => {
      state.results.algorithmResults.fps = data.fps;
      state.results.algorithmResults.detections = data.detections;
      state.results.algorithmResults.topDownDetections = data.topDownDetections;
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
    });
  },
  stitchingPreviewsReceived(data: StitchingPreviewsDTO) {
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
});
