import _ from "lodash";
import { create, StateCreator } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

import { createResultsSlice, ResultsSlice } from "./results";
import { createSocketioSlice, SocketioSlice } from "./socketio";
import { createSettingsSlice, SettingsSlice } from "./settings";

export type MergedStoreState = {
  socketio: SocketioSlice;
  results: ResultsSlice;
  settings: SettingsSlice;
};

export type StoreSliceCreator<T> = StateCreator<
  MergedStoreState,
  [["zustand/immer", never], never],
  [],
  T
>;

export const useAppState = create<MergedStoreState>()(
  persist(
    immer(
      devtools(
        (...args) => ({
          socketio: createSocketioSlice(...args),
          results: createResultsSlice(...args),
          settings: createSettingsSlice(...args),
        }),
        {
          stateSanitizer: (state: any) => {
            return {
              ...state,
              socketio: {
                ...state.socketio,
                sioClient: "<<HIDDEN>>", // replace the socketio instance so that it is not shown in the devtools, as it is a large structure
              },
            };
          },
        }
      )
    ),
    {
      name: "mc-mr-vls-store",
      partialize: (state) => _.pick(state, ["settings"]),
      // allow for deep merges since using slices: https://github.com/pmndrs/zustand/discussions/985
      merge: (persistedState, currentState) =>
        _.merge(currentState, persistedState),
    }
  )
);
