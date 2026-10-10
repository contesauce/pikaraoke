function getScoreData(scoreValue) {
  function randomPhrase(phrases) {
    return phrases[Math.floor(Math.random() * phrases.length)];
  }

  if (scoreValue < 30) {
    return { applause: "applause-l.mp3", review: randomPhrase(scoreReviews.low) };
  } else if (scoreValue < 60) {
    return { applause: "applause-m.mp3", review: randomPhrase(scoreReviews.mid) };
  } else {
    return { applause: "applause-h.mp3", review: randomPhrase(scoreReviews.high) };
  }
}

function getScoreValue() {
  const random = Math.random();
  const bias = 2; // adjust this value to control the bias
  const scoreValue = Math.pow(random, 1 / bias) * 99;
  return Math.floor(scoreValue);
}

async function showFinalScoreWithAudio(
  scoreTextElement,
  scoreValue,
  scoreReviewElement,
  scoreData,
  applauseElement
) {
  scoreTextElement.text(String(scoreValue).padStart(2, "0"));
  scoreReviewElement.text(scoreData.review);
  launchFireworkShow(scoreValue);
  applauseElement.play();
  return new Promise((resolve) => {
    applauseElement.onended = resolve;
  });
}

async function rotateScore(scoreTextElement, duration) {
  const interval = 100;
  const startTime = performance.now();

  while (true) {
    const elapsed = performance.now() - startTime;

    if (elapsed >= duration) break;

    const randomScore = String(Math.floor(Math.random() * 99) + 1).padStart(
      2,
      "0"
    );
    scoreTextElement.text(randomScore);

    const nextUpdate = interval - (performance.now() - (startTime + elapsed));
    await new Promise((resolve) =>
      setTimeout(resolve, Math.max(0, nextUpdate))
    );
  }
}

// How much longer than the drum roll to wait for a real score before giving
// up on it. The server starts scoring the moment the performance ends, same
// as the roll below starts, but the two races independently -- scoring has
// to close and finalize the recording's WAV file and run its own analysis
// over the whole thing, so it can lose that race on a longer song or a slow
// disk even though it eventually succeeds. Extending the wait here (rather
// than just extending the roll itself) keeps the common fast case exactly as
// snappy as before and only costs time when the real score is actually running
// late.
const realScoreGraceMs = 4000;

function waitForRealScore(maxWaitMs) {
  return new Promise((resolve) => {
    if (pendingRealScore !== null) {
      resolve();
      return;
    }
    const pollInterval = 100;
    let waited = 0;
    const poll = setInterval(() => {
      waited += pollInterval;
      if (pendingRealScore !== null || waited >= maxWaitMs) {
        clearInterval(poll);
        resolve();
      }
    }, pollInterval);
  });
}

async function startScore(staticPath) {
  try {
    const r = await fetch(PikaraokeConfig.scorePhrasesUrl);
    scoreReviews = await r.json();
  } catch (_e) {
    // Network failure: keep the last successfully fetched phrases
  }

  const scoreElement = $("#score");
  const scoreTextElement = $("#score-number-text");
  const scoreReviewElement = $("#score-review-text");

  const drums = new Audio(staticPath + "sounds/score-drums.mp3");
  // Pre-create applause audio NOW (empty src) to capture the user activation
  // window -- Mobile Safari only allows audio.play() within a brief window
  // after user events, and that permission attaches to the element, not to
  // whichever file it ends up playing. The real src is set below once the
  // score is known.
  const applause = new Audio();

  scoreElement.show();
  drums.volume = 0.3;
  drums.play();
  const drumDuration = 4100;

  await rotateScore(scoreTextElement, drumDuration);

  // pendingRealScore (splash.js) arrives, if at all, from a server-side
  // scorer that starts work the moment the performance ends, racing
  // independently against the roll above. Most of the time it has already
  // arrived by now; when it hasn't, give it a bounded grace period rather
  // than immediately assuming none is coming. Falls back to the random
  // score only if it still hasn't shown up after that -- e.g. no scoring
  // device configured for this install.
  if (pendingRealScore === null) {
    await waitForRealScore(realScoreGraceMs);
  }
  const scoreValue = pendingRealScore !== null ? pendingRealScore : getScoreValue();
  pendingRealScore = null;
  const scoreData = getScoreData(scoreValue);
  applause.src = staticPath + "sounds/" + scoreData.applause;

  await showFinalScoreWithAudio(
    scoreTextElement,
    scoreValue,
    scoreReviewElement,
    scoreData,
    applause
  );
  scoreReviewElement.text("");
  scoreElement.hide();
}
