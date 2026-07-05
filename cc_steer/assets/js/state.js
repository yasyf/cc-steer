// Shared mutable client state. ES-module live bindings make this single object
// the one source of truth every module reads and mutates.
export const state = {
  view: "pairs",
  rows: [],
  cards: [],
  picks: {},
  flags: {},
  q: "",
};
