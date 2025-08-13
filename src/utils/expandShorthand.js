// src/utils/expandShorthand.js
export function expandShorthand(input) {
  if (!input || !/^s:\s*/i.test(input.trim())) return input; // pass-through
  const lines = input.split(/\r?\n/);
  let S = "", G = "", Q = "";
  for (const ln of lines) {
    const t = ln.trim();
    if (/^s:/i.test(t)) S = t.slice(2).trim();
    else if (/^g:/i.test(t)) G = t.slice(2).trim();
    else if (/^q:/i.test(t)) Q = t.slice(2).trim();
  }
  if (!S || !G || !Q) return input; // not valid shorthand
  return `STATE:\n${S}\n\nGOAL:\n${G}\n\nQUESTION:\n${Q}`;
}
