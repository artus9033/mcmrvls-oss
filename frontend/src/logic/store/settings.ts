import {StoreSliceCreator} from '.';

export interface SettingsSlice {
  // subscription state variables
  shouldSubscribeAlgorithmPreviews: boolean;
  shouldSubscribeCameraPreviews: boolean;
  shouldSubscribeStitchingPreviews: boolean;
  shouldSubscribeMapSegmentationPreviews: boolean;

  // user preference state variables
  darkMode: boolean;

  // subscription mutators
  updateAlgorithmPreviewsSubscription: (shouldBeSubscribed: boolean) => void;
  updateStitchingPreviewsSubscription: (shouldBeSubscribed: boolean) => void;
  updateCameraPreviewsSubscription: (shouldBeSubscribed: boolean) => void;
  updateMapSegmentationPreviewsSubscription: (
    shouldBeSubscribed: boolean,
  ) => void;

  // user preference mutators
  toggleDarkMode: () => void;
}

// @ts-ignore-next-line
export const createSettingsSlice: StoreSliceCreator<SettingsSlice> = (
  set,
  get,
) => ({
  // subscription state variables
  shouldSubscribeAlgorithmPreviews: false,
  shouldSubscribeStitchingPreviews: false,
  shouldSubscribeMapSegmentationPreviews: false,
  shouldSubscribeCameraPreviews: false,

  // user preference state variables
  darkMode: true,

  // subscription mutators
  updateAlgorithmPreviewsSubscription(shouldBeSubscribed) {
    set((state) => {
      state.socketio.sioClient.emit(
        `${shouldBeSubscribed ? '' : 'un'}subscribe_algorithm_previews`,
      );
      state.settings.shouldSubscribeAlgorithmPreviews = shouldBeSubscribed;
    });
  },
  updateStitchingPreviewsSubscription(shouldBeSubscribed) {
    set((state) => {
      state.socketio.sioClient.emit(
        `${shouldBeSubscribed ? '' : 'un'}subscribe_stitching_previews`,
      );
      state.settings.shouldSubscribeStitchingPreviews = shouldBeSubscribed;
    });
  },
  updateCameraPreviewsSubscription(shouldBeSubscribed) {
    set((state) => {
      state.socketio.sioClient.emit(
        `${shouldBeSubscribed ? '' : 'un'}subscribe_camera_previews`,
      );
      state.settings.shouldSubscribeCameraPreviews = shouldBeSubscribed;
    });
  },
  updateMapSegmentationPreviewsSubscription(shouldBeSubscribed) {
    set((state) => {
      state.socketio.sioClient.emit(
        `${shouldBeSubscribed ? '' : 'un'}subscribe_map_segmentation_previews`,
      );
      state.settings.shouldSubscribeMapSegmentationPreviews =
        shouldBeSubscribed;
    });
  },

  // user preference mutators
  toggleDarkMode() {
    set((state) => {
      state.settings.darkMode = !state.settings.darkMode;
    });
  },
});
