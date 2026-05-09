import { Typography } from "@mui/material";
import { useAppState } from "../logic/store";
import moment from "moment";

export const ResultsLastUpdated: React.FC = () => {
  const lastDataTime = useAppState((state) =>
    moment(state.results.lastDataTimestamp)
  );

  return (
    <Typography sx={{ flex: 1 }}>
      Last updated at {lastDataTime?.format("YYYY-MM-DD HH:mm:ss") ?? "never"}
    </Typography>
  );
};
