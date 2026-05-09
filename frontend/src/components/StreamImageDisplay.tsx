import { makeStyles } from "tss-react/mui";

import {
  NoPhotography as NoPhotographyIcon,
  WifiOff,
} from "@mui/icons-material";
import { Avatar, Box, Grid2, Typography } from "@mui/material";

import { useAppState } from "../logic/store";
import { green } from "@mui/material/colors";
import { Size2D } from "../logic/Size2D";
import { useMeasure } from "@uidotdev/usehooks";
import { useEffect, useRef } from "react";
import { useCommonStyles } from "../styles/common";

const useStyles = makeStyles()((theme) => ({
  title: {
    marginTop: theme.spacing(2),
  },
  streamImageDisplay: {
    border: "1px dashed",
    borderColor: theme.palette.text.primary,
    justifyContent: "center",
    alignItems: "center",
    display: "flex",
    flexDirection: "column",
    padding: theme.spacing(1),
    cursor: "pointer",
  },
  streamImageDisplayDisabled: {
    cursor: "unset",
    borderColor: theme.palette.text.disabled,
    filter: "grayscale(80%)",
  },
  streamImageDisplayActive: {
    border: `1px dashed ${green.A700}`,
  },
  caption: {
    marginTop: theme.spacing(1.5),
  },
  noPhotoAvatar: {
    padding: 40,
  },
  noStreamPlaceholderBox: {
    margin: 40,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },
  streamContainer: {
    position: "relative",
    justifyContent: "center",
    alignContent: "center",
    alignItems: "center",
    display: "flex",
    maxWidth: "100%", // do not exceed the parent's width
  },
  stream: {
    marginTop: 20,
    objectFit: "contain",
    width: "100%",
    maxHeight: "calc(min(80vh, 80vw))",
    maxWidth: "calc(min(80vh, 80vw))",
    minHeight: 100,
    minWidth: 100,
  },
  streamBlurred: {
    filter: "blur(0.125rem)",
  },
  noSignalAvatarOverride: {
    position: "absolute",
    backgroundColor: theme.palette.error.main,
    color: theme.palette.error.contrastText,
  },
}));

export type StreamImageDisplayProps = {
  title: string;
  imageString: string | null;
  active: boolean;
  onClick: () => void;
  forceImageSize?: Size2D;
  onMeasureImg?: (size: Size2D | null) => void;
  wide?: boolean;
  disabled?: boolean;
};

export const StreamImageDisplay: React.FC<StreamImageDisplayProps> = ({
  title,
  imageString,
  active,
  onClick,
  forceImageSize,
  onMeasureImg,
  wide = false,
  disabled = false,
}) => {
  const { classes, cx } = useStyles();
  const { classes: commonClasses } = useCommonStyles();
  const isConnected = useAppState((state) => state.socketio.isConnected);

  const [imgMeasureRef, imgSize] = useMeasure();

  const imgRef = useRef<HTMLImageElement | null>(null);
  const previousActiveRef = useRef<boolean>(active);
  const lastImgSize = useRef<{
    width: number | null;
    height: number | null;
  } | null>(null);
  const lastImgValidAndEnabledRef = useRef<boolean>(false);

  // measure image size upon change effect
  useEffect(() => {
    const isImgValidAndEnabled = !!imageString && !disabled;

    if (
      lastImgValidAndEnabledRef.current !== isImgValidAndEnabled ||
      lastImgSize.current?.width !== imgSize.width ||
      lastImgSize.current?.height !== imgSize.height
    ) {
      if (imageString && imgSize.width !== null && imgSize.height !== null) {
        onMeasureImg?.(imgSize as Size2D);
      } else {
        onMeasureImg?.(null);
      }

      lastImgSize.current = imgSize;
      lastImgValidAndEnabledRef.current = isImgValidAndEnabled;
    }
  }, [imgSize, onMeasureImg, imageString, disabled]);

  // scroll image into view upon activation effect
  useEffect(() => {
    if (active !== previousActiveRef.current) {
      imgRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "center",
        inline: "center",
      });
    }

    previousActiveRef.current = active;
  }, [active]);

  return (
    <Grid2
      size={
        active
          ? { sm: 12 }
          : {
              sm: 12,
              md: 6,
              lg: 6,
              xl: wide ? 6 : 3,
            }
      }
      className={cx(
        classes.streamImageDisplay,
        disabled && classes.streamImageDisplayDisabled,
        active && classes.streamImageDisplayActive
      )}
      onClick={disabled ? undefined : onClick}
    >
      <Typography
        variant="subtitle1"
        className={cx(classes.title, disabled && commonClasses.disabledText)}
      >
        {title}
      </Typography>

      {imageString ? (
        <>
          <div className={classes.streamContainer}>
            <img
              ref={(ref) => {
                imgRef.current = ref;

                imgMeasureRef(ref);
              }}
              src={imageString}
              alt={title}
              className={cx(
                classes.stream,
                !isConnected && classes.streamBlurred
              )}
              style={
                forceImageSize
                  ? {
                      width: forceImageSize.width,
                      height: forceImageSize.height,
                    }
                  : undefined
              }
            />

            {!isConnected && (
              <Avatar
                classes={{ root: classes.noSignalAvatarOverride }}
                className={classes.noPhotoAvatar}
              >
                <WifiOff fontSize="large" />
              </Avatar>
            )}
          </div>
        </>
      ) : (
        <Box className={classes.noStreamPlaceholderBox}>
          <Avatar className={classes.noPhotoAvatar}>
            <NoPhotographyIcon fontSize="large" />
          </Avatar>

          <Typography
            textAlign="center"
            variant="caption"
            className={cx(classes.caption, commonClasses.disabledText)}
          >
            Stream unavailable
          </Typography>
        </Box>
      )}
    </Grid2>
  );
};
