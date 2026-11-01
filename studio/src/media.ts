import type { Project } from "./adapter";
import { audible, fade } from "./adapter";
import impulse from "../../contracts/fixtures/v1/impulse.wav?url";
import sine from "../../contracts/fixtures/v1/sine.wav?url";
import long from "../../contracts/fixtures/v1/long.wav?url";
import short from "../../contracts/fixtures/v1/short.wav?url";
export const urls: Record<string, string> = { impulse, sine, long, short };
const cache = new Map<string, Promise<Float32Array>>();
// Bounded PCM16 fixture reader; no browser resampling, remote media or ML.
export function samples(id: string): Promise<Float32Array> {
  if (!urls[id]) return Promise.reject(new Error("missing_media"));
  if (!cache.has(id))
    cache.set(
      id,
      fetch(urls[id])
        .then((r) => {
          if (!r.ok) throw Error("missing_media");
          return r.arrayBuffer();
        })
        .then((b) => {
          const v = new DataView(b);
          let data = 0,
            size = 0,
            rate = 0,
            channels = 0,
            bits = 0,
            format = 0;
          for (let i = 12; i + 8 <= b.byteLength; ) {
            const name = String.fromCharCode(...new Uint8Array(b, i, 4)),
              n = v.getUint32(i + 4, true);
            if (name === "fmt ") {
              format = v.getUint16(i + 8, true);
              channels = v.getUint16(i + 10, true);
              rate = v.getUint32(i + 12, true);
              bits = v.getUint16(i + 22, true);
            }
            if (name === "data") {
              data = i + 8;
              size = n;
              break;
            }
            i += 8 + n + (n % 2);
          }
          if (
            format !== 1 ||
            rate !== 48000 ||
            channels !== 1 ||
            bits !== 16 ||
            !data ||
            data + size > b.byteLength
          )
            throw Error("unsupported_fixture_media");
          return Float32Array.from(
            { length: size / 2 },
            (_, i) => v.getInt16(data + i * 2, true) / 32768,
          );
        }),
    );
  return cache.get(id)!;
}
export async function mix(p: Project) {
  const length = Math.max(
    ...p.clips.map((c) => c.start + c.out_tick - c.in_tick),
  );
  if (length > 48000 * 60) throw Error("fixture_preview_limit");
  const output = new Float32Array(length);
  const unique = new Map(
    await Promise.all(
      [...new Set(p.clips.map((c) => c.asset_id))].map(
        async (id) => [id, await samples(id)] as const,
      ),
    ),
  );
  for (const c of p.clips)
    if (audible(p, c)) {
      const source = unique.get(c.asset_id)!;
      if (c.out_tick > source.length) throw Error("missing_media");
      for (let i = 0; i < c.out_tick - c.in_tick; i++)
        output[c.start + i] += source[c.in_tick + i] * c.gain * fade(c, i);
    }
  let peak = 0;
  for (const x of output) peak = Math.max(peak, Math.abs(x));
  if (peak > 1) throw Error("clipping: reduce gain before preview");
  return output;
}
