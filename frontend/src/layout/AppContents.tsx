import {Divider, Grid2, Stack, Switch} from '@mui/material';
import {contentPaddings} from '../styles/common';
import {useAppState} from '../logic/store';
import {
  StreamImageDisplay,
  StreamImageDisplayProps,
} from '../components/StreamImageDisplay';
import {useCallback, useEffect, useMemo, useState} from 'react';
import {SectionHeader} from '../components/SectionHeader';
import {Size2D} from '../logic/Size2D';
import {makeStyles} from 'tss-react/mui';

enum AlgorithmPreviewStream {
  TOP_DOWN = 'Top down view',
  TOP_DOWN_ANNOTATED = 'Top down view with annotations',
  STITCHED = 'Stitched view',
  STITCHED_ANNOTATED = 'Stitched view with annotations',
}

enum MapSegmentationPreviewStream {
  MAP_SEGMENTATION_YELLOW_LINES = 'Lane lines (yellow)',
  MAP_SEGMENTATION_RED_STOP_LINES = 'Stop lines (red)',
  MAP_SEGMENTATION_ROAD_COMPONENTS = 'Roads',
}

type StitchingPreviewStream = `Stitching stage ${string}`;

type CameraPreviewStream = `Camera ${string}`;

type Stream =
  | CameraPreviewStream
  | AlgorithmPreviewStream
  | StitchingPreviewStream
  | MapSegmentationPreviewStream;

type AppContentsProps = {};

const useStyles = makeStyles()((theme) => ({
  previewsGrid: {
    justifyContent: 'space-evenly',
  },
}));

export const AppContents: React.FC<AppContentsProps> = () => {
  const {classes} = useStyles();

  const [activeStream, setActiveStream] = useState<Stream | null>(null);
  const [stitchedViewSize, setStitchedViewSize] = useState<Size2D | undefined>(
    undefined,
  );

  const getStreamProps = useCallback<
    (
      stream: Stream,
    ) => Pick<StreamImageDisplayProps, 'onClick' | 'active' | 'title'>
  >(
    (stream) => {
      const active = activeStream === stream;

      return {
        onClick: () => {
          setActiveStream(active ? null : stream);
        },
        active,
        title: stream,
      };
    },
    [activeStream],
  );

  // previews
  const algorithmPreviews = useAppState(
    (state) => state.results.algorithmPreviews,
  );
  const _stitchingPreviews = useAppState(
    (state) => state.results.stitchingPreviews,
  );
  const cameraPreviews = useAppState((state) => state.results.cameraPreviews);
  const mapSegmentationPreviews = useAppState(
    (state) => state.results.mapSegmentationPreviews,
  );

  // subscription state variables
  const algorithmPreviewsSubscribed = useAppState(
    (state) => state.settings.shouldSubscribeAlgorithmPreviews,
  );
  const stitchingPreviewsSubscribed = useAppState(
    (state) => state.settings.shouldSubscribeStitchingPreviews,
  );
  const cameraPreviewsSubscribed = useAppState(
    (state) => state.settings.shouldSubscribeCameraPreviews,
  );
  const mapSegmentationPreviewsSubscribed = useAppState(
    (state) => state.settings.shouldSubscribeMapSegmentationPreviews,
  );

  // subscription mutators
  const updateAlgorithmPreviewsSubscription = useAppState(
    (state) => state.settings.updateAlgorithmPreviewsSubscription,
  );
  const updateStitchingPreviewsSubscription = useAppState(
    (state) => state.settings.updateStitchingPreviewsSubscription,
  );
  const updateCameraPreviewsSubscription = useAppState(
    (state) => state.settings.updateCameraPreviewsSubscription,
  );
  const updateMapSegmentationPreviewsSubscription = useAppState(
    (state) => state.settings.updateMapSegmentationPreviewsSubscription,
  );

  const stitchingStages = useMemo(() => {
    const stages = Object.entries(_stitchingPreviews.commonFeaturesJPEGsMap);

    if (stages.length === 0) {
      return [
        ['(0+1)', ''],
        ['((0+1)+2)', ''],
      ];
    }

    return stages;
  }, [_stitchingPreviews.commonFeaturesJPEGsMap]);

  // collapse active stream if its parent channel is not subscribed effect
  useEffect(() => {
    if (activeStream) {
      const isAlgorithmPreviewStreamActive = Object.values(
        AlgorithmPreviewStream,
      ).includes(activeStream as AlgorithmPreviewStream);

      console.log(
        Object.values(AlgorithmPreviewStream),
        activeStream,
        isAlgorithmPreviewStreamActive,
        algorithmPreviewsSubscribed,
      );
      const isStitchingPreviewStreamActive =
        activeStream.startsWith('Stitching stage');

      if (
        (!algorithmPreviewsSubscribed && isAlgorithmPreviewStreamActive) ||
        (!stitchingPreviewsSubscribed && isStitchingPreviewStreamActive)
      ) {
        setActiveStream(null);
      }
    }
  }, [activeStream, algorithmPreviewsSubscribed, stitchingPreviewsSubscribed]);

  return (
    <Stack>
      <SectionHeader disabled={!algorithmPreviewsSubscribed}>
        🔢 Previews of algorithm processing stages{' '}
        <Switch
          checked={algorithmPreviewsSubscribed}
          onChange={(event) => {
            updateAlgorithmPreviewsSubscription(event.target.checked);
          }}
        />
      </SectionHeader>

      <Grid2 container sx={contentPaddings} className={classes.previewsGrid}>
        <StreamImageDisplay
          {...getStreamProps(AlgorithmPreviewStream.TOP_DOWN)}
          imageString={algorithmPreviews.topDownImgStrJpeg}
          disabled={!algorithmPreviewsSubscribed}
        />

        <StreamImageDisplay
          {...getStreamProps(AlgorithmPreviewStream.TOP_DOWN_ANNOTATED)}
          imageString={algorithmPreviews.topDownImgAnnotatedStrJpeg}
          onMeasureImg={(size) => {
            if (size === null) {
              setStitchedViewSize(undefined);
            } else {
              setStitchedViewSize(size);
            }
          }}
          disabled={!algorithmPreviewsSubscribed}
        />

        <StreamImageDisplay
          {...getStreamProps(AlgorithmPreviewStream.STITCHED)}
          imageString={algorithmPreviews.stitchedImgStrJpeg}
          forceImageSize={stitchedViewSize}
          disabled={!algorithmPreviewsSubscribed}
        />

        <StreamImageDisplay
          {...getStreamProps(AlgorithmPreviewStream.STITCHED_ANNOTATED)}
          imageString={algorithmPreviews.stitchedImgAnnotatedStrJpeg}
          forceImageSize={stitchedViewSize}
          disabled={!algorithmPreviewsSubscribed}
        />
      </Grid2>

      <Divider sx={{mt: 6}} variant="middle" />

      <SectionHeader disabled={!mapSegmentationPreviewsSubscribed}>
        📋 Map segmentation previews{' '}
        <Switch
          checked={mapSegmentationPreviewsSubscribed}
          onChange={(event) => {
            updateMapSegmentationPreviewsSubscription(event.target.checked);
          }}
        />
      </SectionHeader>

      <Grid2 container sx={contentPaddings} className={classes.previewsGrid}>
        <StreamImageDisplay
          {...getStreamProps(
            MapSegmentationPreviewStream.MAP_SEGMENTATION_YELLOW_LINES,
          )}
          imageString={mapSegmentationPreviews.yellowLinesStrJpeg}
          forceImageSize={stitchedViewSize}
          disabled={!mapSegmentationPreviewsSubscribed}
        />

        <StreamImageDisplay
          {...getStreamProps(
            MapSegmentationPreviewStream.MAP_SEGMENTATION_RED_STOP_LINES,
          )}
          imageString={mapSegmentationPreviews.redStopLinesStrJpeg}
          forceImageSize={stitchedViewSize}
          disabled={!mapSegmentationPreviewsSubscribed}
        />

        <StreamImageDisplay
          {...getStreamProps(
            MapSegmentationPreviewStream.MAP_SEGMENTATION_ROAD_COMPONENTS,
          )}
          imageString={mapSegmentationPreviews.roadComponentsStrJpeg}
          forceImageSize={stitchedViewSize}
          disabled={!mapSegmentationPreviewsSubscribed}
        />
      </Grid2>

      <Divider sx={{mt: 6}} variant="middle" />

      <SectionHeader disabled={!cameraPreviewsSubscribed}>
        📎 Previews of cameras{' '}
        <Switch
          checked={cameraPreviewsSubscribed}
          onChange={(event) => {
            updateCameraPreviewsSubscription(event.target.checked);
          }}
        />
      </SectionHeader>

      <Grid2 container sx={contentPaddings} className={classes.previewsGrid}>
        {Object.entries(cameraPreviews.camerasJPEGsMap).map(
          ([camera, commonFeaturesImgStrJpeg], index) => (
            <StreamImageDisplay
              key={index}
              {...getStreamProps(`Camera ${camera}`)}
              imageString={commonFeaturesImgStrJpeg}
              wide
              disabled={!stitchingPreviewsSubscribed}
            />
          ),
        )}
      </Grid2>

      <Divider sx={{mt: 6}} variant="middle" />

      <SectionHeader disabled={!stitchingPreviewsSubscribed}>
        📎 Previews of stitching stages{' '}
        <Switch
          checked={stitchingPreviewsSubscribed}
          onChange={(event) => {
            updateStitchingPreviewsSubscription(event.target.checked);
          }}
        />
      </SectionHeader>

      <Grid2 container sx={contentPaddings} className={classes.previewsGrid}>
        {stitchingStages.map(
          ([stitchingStage, commonFeaturesImgStrJpeg], index) => (
            <StreamImageDisplay
              key={index}
              {...getStreamProps(`Stitching stage ${stitchingStage}`)}
              imageString={commonFeaturesImgStrJpeg}
              wide
              disabled={!stitchingPreviewsSubscribed}
            />
          ),
        )}
      </Grid2>
    </Stack>
  );
};
