import {
  Content,
  EdgeSidebar,
  Header,
  Root,
  getStandardScheme,
} from '@mui-treasury/layout';
import {
  Breakpoint,
  Chip,
  CssBaseline,
  IconButton,
  Paper,
  Tooltip,
  Typography,
} from '@mui/material';
import {useMemo} from 'react';

import pkg from '../../package.json';
import logo from '../assets/logo.svg';
import {useAppState} from '../logic/store';
import {RobotsList} from '../components/RobotsList';
import {useCommonStyles} from '../styles/common';
import {AppContents} from './AppContents';
import {socketEndpoint} from '../logic/store/socketio';
import {ThemeProvider} from '@emotion/react';
import {createTheme} from '../logic/theme';
import {DarkMode, LightMode} from '@mui/icons-material';
import {HeaderConfig} from '@mui-treasury/layout/Header/HeaderBuilder';

const scheme = getStandardScheme();

for (const key in scheme.header.config as Record<Breakpoint, HeaderConfig>) {
  const c = scheme.header.config[key as Breakpoint];

  if (c) c.position = 'sticky';
}

function AppLayout() {
  const {classes, cx} = useCommonStyles();

  const isConnected = useAppState((state) => state.socketio.isConnected);
  const isConnecting = useAppState((state) => state.socketio.isConnecting);
  const connectionError = useAppState(
    (state) => state.socketio.connectionError,
  );
  const detectionsFPS = useAppState(
    (state) => state.results.algorithmResults.fps,
  );
  const _dataStale = useAppState(
    (state) => state.results.algorithmResults.dataStale ?? true,
  );
  const dataStale = _dataStale || !isConnected;

  const darkMode = useAppState((state) => state.settings.darkMode);
  const toggleDarkMode = useAppState((state) => state.settings.toggleDarkMode);

  const statusChip = useMemo(
    () =>
      isConnected
        ? {label: 'Connected', color: 'success' as const}
        : isConnecting
        ? {label: 'Connecting', color: 'default' as const}
        : {
            label: 'Disconnected',
            color: 'error' as const,
            tooltip: `Endpoint: ${socketEndpoint}`,
          },
    [isConnected, isConnecting],
  );

  const theme = useMemo(() => createTheme(darkMode), [darkMode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      <Root scheme={scheme}>
        <Header className={classes.header}>
          {/* left block */}
          <div className={cx(classes.fullHeight, classes.rowCentered)}>
            <img src={logo} className={classes.logo} alt="logo" />

            <div>
              <Typography variant="h6">M-C M-R VLS</Typography>
              <Typography variant="subtitle1">v{pkg.version}</Typography>
            </div>
          </div>

          {/* central block */}
          <div className={cx(classes.fullHeight, classes.rowCentered)}>
            <Tooltip
              title={statusChip.tooltip}
              arrow
              disableHoverListener={!statusChip.tooltip}>
              <Chip
                label={statusChip.label}
                size="small"
                color={statusChip.color}
                sx={{fontWeight: 600}}
              />
            </Tooltip>
          </div>

          {/* semi-right block */}
          <div className={cx(classes.fullHeight, classes.rowCentered)}>
            <Tooltip
              title={
                dataStale
                  ? 'Currently the algorithm is unable to perform stitching, displayed data may be outdated'
                  : undefined
              }
              arrow>
              <Chip
                label={dataStale ? 'Stale data' : 'Live data'}
                size="small"
                color={dataStale ? 'warning' : 'success'}
                sx={{mr: 1}}
              />
            </Tooltip>
            <Tooltip
              title={connectionError?.message}
              arrow
              disableHoverListener={!connectionError?.message}>
              <Chip
                label={
                  connectionError?.message
                    ? `Error: ${connectionError.message}`
                    : `${(Math.round(detectionsFPS * 100) / 100).toFixed(
                        2,
                      )} FPS`
                }
                size="small"
                color={connectionError ? 'error' : 'default'}
                sx={{maxWidth: 240}}
              />
            </Tooltip>
          </div>

          {/* right block */}
          <div className={cx(classes.fullHeight, classes.rowCentered)}>
            <Tooltip
              title={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
              arrow>
              <IconButton onClick={toggleDarkMode}>
                {darkMode ? <LightMode /> : <DarkMode />}
              </IconButton>
            </Tooltip>
          </div>
        </Header>

        <EdgeSidebar anchor="left">
          <Paper className={classes.edgeSidebarPaper}>
            <RobotsList />
          </Paper>
        </EdgeSidebar>

        <Content>
          <AppContents />
        </Content>
      </Root>
    </ThemeProvider>
  );
}

export default AppLayout;
