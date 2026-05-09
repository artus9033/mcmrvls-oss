import { createTheme as createMuiTheme } from "@mui/material";
import { amber, blueGrey, grey } from "@mui/material/colors";

export function createTheme(darkMode: boolean) {
  return createMuiTheme({
    palette: {
      primary: blueGrey,
      secondary: amber,
      mode: darkMode ? "dark" : "light",
      background: { default: grey[800], paper: grey[900] },
    },
  });
}
