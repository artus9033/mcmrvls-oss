import {SxProps} from '@mui/material';
import {Theme} from '@mui/system';
import {makeStyles} from 'tss-react/mui';

export const useCommonStyles = makeStyles()((theme) => ({
  header: {
    justifyContent: 'space-between',
    flexDirection: 'row',
    paddingLeft: '1rem',
    paddingRight: '1rem',
  },
  fullHeight: {
    height: '100%',
  },
  rowCentered: {
    display: 'flex',
    flexDirection: 'row',
    alignItems: 'center',
  },
  rowSpaceBetween: {
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  logo: {
    height: '100%',
    width: 'auto',
    aspectRatio: 1,
    marginRight: '0.5rem',
  },
  footerWrapper: {
    marginBottom: theme.spacing(1),
    marginTop: theme.spacing(2),
    padding: theme.spacing(2, 4),
  },
  edgeSidebarPaper: {
    height: '100%',
  },
  disabledText: {
    color: theme.palette.text.disabled,
  },
}));

export const contentPaddings = {
  paddingLeft: 1.5,
  paddingRight: 1.5,
  paddingTop: 0.5,
  paddingBottom: 0.5,
} satisfies Partial<SxProps<Theme>>;
