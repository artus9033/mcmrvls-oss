import { Footer as MuiFooter } from "@mui-treasury/layout";
import { Box, Divider, Link, Typography } from "@mui/material";

import { contentPaddings, useCommonStyles } from "../styles/common";
import { ResultsLastUpdated } from "../components/ResultsLastUpdated";

export function Footer() {
  const { classes } = useCommonStyles();

  return (
    <MuiFooter className={classes.footerWrapper}>
      <Divider sx={{ mt: 4, mb: 4 }} variant="middle" />

      <Box sx={contentPaddings}>
        <Box className={classes.rowSpaceBetween}>
          <ResultsLastUpdated />

          <Typography variant="h6" textAlign="center" sx={{ flex: 2 }}>
            Multi-Camera Multi-Robot Visual Localization System
          </Typography>

          <Typography sx={{ flex: 1 }} textAlign="end">
            © {new Date().getFullYear()}
          </Typography>
        </Box>

        <br />

        <Typography style={{ whiteSpace: "pre-wrap" }} textAlign="center">
          Authors:{" "}
          <Link
            target="_blank"
            rel="noreferrer"
            href="https://www.researchgate.net/profile/Artur-Morys-Magiera"
          >
            Artur Morys - Magiera
          </Link>
          ,{" "}
          <Link
            target="_blank"
            rel="noreferrer"
            href="https://www.researchgate.net/profile/Marek-Dlugosz"
          >
            Marek Długosz
          </Link>
          ,{" "}
          <Link
            target="_blank"
            rel="noreferrer"
            href="https://www.researchgate.net/profile/Pawel-Skruch"
          >
            Paweł Skruch
          </Link>
          {"\n"}
          AGH University of Krakow Faculty of Electrical Engineering,
          Automatics, Computer Science and Biomedical Engineering
          {"\n"}
          Department of Automatics and Robotics
        </Typography>
      </Box>
    </MuiFooter>
  );
}
