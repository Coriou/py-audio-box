import WaveformPlaylist from "waveform-playlist";
import fixture from "../../contracts/fixtures/v1/production-30.json";
const assets = import.meta.glob("../../contracts/fixtures/v1/*.wav", {
  query: "?url",
  import: "default",
  eager: true,
});
document.querySelector("#load").onclick = async () => {
  try {
    const playlist = WaveformPlaylist.init({
      container: document.querySelector("#playlist"),
      samplesPerPixel: 128,
      zoomLevels: [128, 256, 512],
      sampleRate: 48000,
      state: "shift",
      controls: { show: true, width: 120 },
    });
    const clips = fixture.clips;
    await playlist.load(
      clips.map((c) => ({
        src: assets[`../../contracts/fixtures/v1/${c.asset_id}.wav`],
        name: c.id,
        start: c.start / 48000,
        cuein: c.in_tick / 48000,
        cueout: c.out_tick / 48000,
        gain: c.gain,
        muted: c.mute,
      })),
    );
    window.playlist = playlist;
    document.querySelector("#result").textContent = JSON.stringify(
      {
        loaded: playlist.tracks.length,
        canonicalTracks: fixture.tracks.length,
        clips: clips.length,
        first: playlist.getInfo()[0],
        sampleRate: playlist.ac.sampleRate,
      },
      null,
      2,
    );
  } catch (e) {
    document.querySelector("#result").textContent = String(e);
  }
};
// Console-callable probe; mutates only the third-party component, never fixture.
window.probeZero = async () => {
  const c = fixture.clips[0];
  await window.playlist.load([
    {
      src: assets[`../../contracts/fixtures/v1/${c.asset_id}.wav`],
      name: "zero-probe",
      gain: 0,
    },
  ]);
  return window.playlist.tracks.at(-1).gain;
};
