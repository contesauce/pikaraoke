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
  // scorer that starts work the moment the performance ends -- well before
  // the roll above finishes. Falls back to the random score when nothing
  // arrived in time, e.g. no scoring device configured for this install.
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
