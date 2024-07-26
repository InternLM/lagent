import { EN } from "./en";
import { ZH } from "./zh";

export default {
  "zh-CN": ZH,
  "en-US": EN,
} as {
  [propName: string]: any;
};

export const DEFAULT_LANG = "zh-CN";
