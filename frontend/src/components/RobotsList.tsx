import {
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  useTheme,
} from "@mui/material";
import React from "react";

import { Navigation as NavigationIcon } from "@mui/icons-material";
import { useAppState } from "../logic/store";

import { makeStyles } from "tss-react/mui";

const useStyles = makeStyles()((theme) => ({
  robotIcon: {
    color: theme.palette.secondary.light,
  },
}));

type RobotsListProps = {};

export const RobotsList: React.FC<RobotsListProps> = () => {
  const { classes } = useStyles();
  const {
    palette: {
      text: { primary: primaryText },
    },
  } = useTheme();

  const detections = useAppState(
    (state) => state.results.algorithmResults.detections
  );

  return (
    <List>
      {detections?.map(({ robot, x, y, azimuth, poseSource }) => (
        <ListItem key={robot.id}>
          <ListItemText
            primary={
              <u>
                {robot.id} ({robot.host})
                {poseSource === "estimation" ? " · coasted" : ""}
              </u>
            }
            secondary={
              <>
                <b>X</b>: {Math.round(x * 100)}%{"\n"}
                <b>Y</b>: {Math.round(y * 100)}%{"\n"}
                <b>θ</b>: {azimuth.toFixed()}°{"\n"}
                <b>pose</b>: {poseSource}
              </>
            }
            primaryTypographyProps={{ fontSize: 20 }}
            secondaryTypographyProps={{
              color: primaryText,
              fontSize: 18,
              whiteSpace: "pre-wrap",
            }}
          />
          <ListItemAvatar>
            <Avatar sx={{ padding: 3 }}>
              <NavigationIcon
                className={classes.robotIcon}
                style={{
                  rotate: `${azimuth.toFixed(3)}deg`,
                }}
                fontSize="large"
              />
            </Avatar>
          </ListItemAvatar>
        </ListItem>
      ))}
    </List>
  );
};
