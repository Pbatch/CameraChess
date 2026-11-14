// Replays saved model outputs through ChessCam's move tracking, @chesscam/recorder's BeamTracker (pinned in
// package.json), and scores each game against its true moves. Called by run.py, which owns the settings.
//
// Reads JSON on stdin: { games: [{ dir, probs }], fps, leadIn, settings: [{ temperature, evidenceWeight, ... }] }
// and writes, for each setting, each game's score: plies, correctPrefix, editDistance, afterEnd.
//
// Scoring is ChessCam's dev/replay.mjs and src/dev/replay.ts: replay starts LEAD_IN s before the first true move and
// stops once the true last move is shown; editDistance is the fewest moves to add, remove or change to turn the
// tracker's game into the true one, with moves after the true last move free.
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { makeSan, parseUci, Position, replay } from "@chesscam/recorder";
// Not exported from the package, so imported from its files; package.json pins the version
import { BeamTracker } from "./node_modules/@chesscam/recorder/dist/vision/beamTracker.js";

const toSans = (ucis) => {
  const pos = Position.default();
  return ucis.map((uci) => {
    const move = parseUci(uci);
    if (move === undefined || !pos.isLegal(move)) throw new Error(`Bad move ${uci}`);
    const san = makeSan(pos, move);
    pos.play(move);
    return san;
  });
};

const score = (predicted, truth, truthTimes) => {
  const end = truthTimes[truthTimes.length - 1];
  let firstAfterEnd = predicted.length;
  while (firstAfterEnd > 0 && predicted[firstAfterEnd - 1].time > end) firstAfterEnd -= 1;
  let correctPrefix = 0;
  while (correctPrefix < Math.min(predicted.length, truth.length) && predicted[correctPrefix].san === truth[correctPrefix]) {
    correctPrefix += 1;
  }
  let previous = Array.from({ length: truth.length + 1 }, (_, j) => j);
  let editDistance = (firstAfterEnd === 0) ? previous[truth.length] : Number.POSITIVE_INFINITY;
  for (let i = 1; i <= predicted.length; i++) {
    const row = [i];
    for (let j = 1; j <= truth.length; j++) {
      const change = previous[j - 1] + ((predicted[i - 1].san === truth[j - 1]) ? 0 : 1);
      row.push(Math.min(change, previous[j] + 1, row[j - 1] + 1));
    }
    previous = row;
    if (i >= firstAfterEnd) editDistance = Math.min(editDistance, row[truth.length]);
  }
  return { plies: truth.length, correctPrefix, editDistance, afterEnd: predicted.length - firstAfterEnd };
};

const job = JSON.parse(fs.readFileSync(0, "utf8"));
const games = job.games.map(({ dir, probs }) => {
  const meta = JSON.parse(fs.readFileSync(path.join(dir, `${probs}.json`), "utf8"));
  if (meta.fps !== job.fps) throw new Error(`${dir}: saved at ${meta.fps} fps, not ${job.fps}`);
  const buffer = zlib.gunzipSync(fs.readFileSync(path.join(dir, `${probs}.f32.gz`)));
  const [game] = JSON.parse(fs.readFileSync(path.join(dir, "overlay_moves.json"), "utf8"));
  return {
    name: path.basename(dir), model: meta.model, game,
    probs: new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4), truth: toSans(game.moves)
  };
});
const results = job.settings.map((setting) => games.map(({ name, model, game, probs, truth }) => {
  const start = Math.max(0, game.times[0] - job.leadIn);
  const until = (shown) => shown.length >= truth.length && shown[truth.length - 1].san === truth[truth.length - 1];
  const moves = replay(probs, job.fps, start, new BeamTracker(setting), until);
  return { game: name, model, ...score(moves, truth, game.times) };
}));
process.stdout.write(JSON.stringify(results));
