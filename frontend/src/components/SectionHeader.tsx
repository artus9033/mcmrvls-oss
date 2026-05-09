import { Typography } from "@mui/material";
import { PropsWithChildren } from "react";
import { makeStyles } from "tss-react/mui";
import { useCommonStyles } from "../styles/common";

export type SectionHeaderProps = PropsWithChildren<{
  disabled?: boolean;
}>;

const useStyles = makeStyles()((theme) => ({
  sectionHeader: {
    marginTop: theme.spacing(4),
    marginBottom: theme.spacing(4),
  },
}));

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  children,
  disabled = false,
}) => {
  const { classes, cx } = useStyles();
  const { classes: commonClasses } = useCommonStyles();

  return (
    <Typography
      variant="h5"
      textAlign="center"
      className={cx(
        classes.sectionHeader,
        disabled && commonClasses.disabledText
      )}
    >
      {children}
    </Typography>
  );
};
