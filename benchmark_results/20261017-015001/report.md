# VCM benchmark - 202521354 - 20261017-015001

Wake word: **Alexa** - trials: 202 with the wake word + 16 without - shuffle seed: 24992 - connection: ssh - holdout: huggingface

Pi log file: ~/vcm_benchmark/benchmark_20261017_011334.log

Mic check (Pi input plughw:3,0): signal-to-noise 23.1 dB, laptop speech -19.9 dBFS, room noise -42.9 dBFS - ok

## At a glance

|                                   | overall      | real voice   | synthetic voice |
|-----------------------------------|--------------|--------------|-----------------|
| intent accuracy (19)              | 73.3%        | 65.6%        | 80.2%           |
| command accuracy (93)             | 72.8%        | 64.6%        | 80.2%           |
| false accept (out of scope fired) | 12.5% (2/16) | 10.0% (1/10) | 16.7% (1/6)     |
| false reject (command ignored)    | 25.3%        | 33.7%        | 18.0%           |
| false wake (no wake word, fired)  | 31.2% (5/16) | 0.0% (0/7)   | 55.6% (5/9)     |
| slot exact                        | 98.8%        | 96.8%        | 100.0%          |
| latency p95                       | 1.19 s       | 1.01 s       | 1.81 s          |

**Pi:** real-time factor 0.007 (p95 0.012), inference 12 ms, CPU temp max 57.9 C, runtime CPU 90% mean, runtime RAM 155 MB peak

**Check:**

- the Pi answered only 82.7% of commands: check volume, distance, wake word and the log path

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject)  | 93 commands (+reject) |
|---------------------------------------------------|-----------------------|-----------------------|
| accuracy                                          | 73.3%                 | 72.8%                 |
| balanced accuracy                                 | 71.6%                 | 71.7%                 |
| precision (macro)                                 | 91.3%                 | 91.6%                 |
| recall (macro)                                    | 71.6%                 | 71.7%                 |
| F1 (macro)                                        | 76.8%                 | 77.6%                 |
| F2 (macro)                                        | 72.6%                 | 73.4%                 |
| false accept rate (OOS fired)                     | 12.5%                 | 12.5%                 |
| false reject rate (in-scope silent/rejected)      | 25.3%                 | 25.3%                 |
| misfire rate (wrong command fired)                | 2.7%                  | 3.2%                  |
| accuracy 95% CI                                   | [67-79%]              | [66-78%]              |
| false accept 95% CI                               | [3-36%] (2/16)        | [3-36%]               |
| false wake rate (command without wake word fired) | 31.2% [14-56%] (5/16) | 31.2% [14-56%] (5/16) |

Responses: 82.7% of trials fired a command; no response: 35; extra fires: 0; wake detect rate: 82.7%

**False wakes (no wake word, Pi fired):** Brightness 60 percent -> OUT_OF_SCOPE; Message -> OUT_OF_SCOPE; Tell me the time -> OUT_OF_SCOPE; Timer 10 seconds -> OUT_OF_SCOPE; Turn on the lights -> OUT_OF_SCOPE

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 73.3% [67-79%] | 65.6% [56-74%] | 80.2% [72-87%]  |
| balanced accuracy              | 71.6%          | 62.3%          | 80.0%           |
| F1 (macro)                     | 76.8%          | 67.4%          | 82.5%           |
| F2 (macro)                     | 72.6%          | 63.2%          | 80.1%           |
| false accept rate              | 12.5% (2/16)   | 10.0% (1/10)   | 16.7% (1/6)     |
| false reject rate              | 25.3%          | 33.7%          | 18.0%           |
| misfire rate                   | 2.7%           | 3.5%           | 2.0%            |
| **93 commands** accuracy       | 72.8%          | 64.6%          | 80.2%           |
| balanced accuracy              | 71.7%          | 62.0%          | 79.6%           |
| F1 (macro)                     | 77.6%          | 60.4%          | 79.1%           |
| F2 (macro)                     | 73.4%          | 61.1%          | 79.2%           |
| misfire rate                   | 3.2%           | 4.7%           | 2.0%            |
| slot exact (intent right)      | 98.8% (n=80)   | 96.8% (n=31)   | 100.0% (n=49)   |
| latency p50 / p95              | 0.62 / 1.19 s  | 0.56 / 1.01 s  | 0.64 / 1.81 s   |
| false wake rate (no wake word) | 31.2% (5/16)   | 0.0% (0/7)     | 55.6% (5/9)     |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 12 | 100.0% | 0.0 min        | 0.000          | 0.000         | 0.000     |
| BRIGHTNESS      | 11 | 100.0% | 0.0 %          | 0.000          | 0.000         | 0.000     |
| COLOR           | 15 | 100.0% | -              | -              | 0.000         | 0.000     |
| CREATE_REMINDER | 10 | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 16 | 93.8%  | 0.1 deg        | 0.016          | 0.016         | 0.014     |
| TIMER           | 16 | 100.0% | 0.0 s          | 0.000          | 0.175         | 0.231     |
| ALL             | 80 | 98.8%  | -              | 0.005          | 0.038         | 0.049     |

## Raspberry Pi

- **Raspberry Pi 4 Model B Rev 1.5**, 4 cores  up to 1800.0 MHz, RAM 3789.8 MB, Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-v8, Python 3.13.5
- packages: onnxruntime 1.21, numpy 2.2.4, sounddevice 0.5.6

| metric                                      | mean / p95 / max         |
|---------------------------------------------|--------------------------|
| response latency (command end -> Pi output) | 0.528 / 1.193 / 3.317 s  |
| latency p50 / p99                           | 0.617 / 2.849 s          |
| inference time (Pi-reported)                | 12.3 / 16.0 / 28.9 ms    |
| real-time factor (infer / audio window)     | 0.007 / 0.012 / 0.029    |
| CPU temperature                             | 51.0 / 54.0 / 57.9 C     |
| CPU use, whole Pi                           | 23.2 / 67.6 / 81.2 %     |
| CPU use, your runtime process               | 89.6 / 267.0 / 324.6 %   |
| RAM (RSS), your runtime process             | 154.9 / 155.1 / 155.1 MB |
| RAM used, whole Pi                          | 573.2 / 574.0 / 574.7 MB |
| CPU clock                                   | 1597 / 1800 / 1800 MHz   |
| load average (1 min)                        | 1.00 / 1.44 / 1.84       |
| runtime CPU-seconds per second of speech    | 9.035                    |
| runtime CPU share of wall time              | 89.4%                    |
| throttling flags seen                       | none                     |
| test wall time                              | 60.7 min                 |

## Most frequent confusions

**intent level:** BRIGHTNESS -> REJECT (7); ALARM -> REJECT (6); CREATE_REMINDER -> REJECT (6); STOP -> REJECT (4); COLOR -> REJECT (3); MESSAGE -> REJECT (3); TIME -> REJECT (3); CREATE_REMINDER -> LIST_REMINDERS (2); LIGHT_OFF -> REJECT (2); TEMPERATURE -> REJECT (2)

**command level:** Send my message -> REJECT (2); Stop playing -> REJECT (2); Time -> REJECT (2); Brightness 20 percent -> REJECT (2); Wake me up at 8:00 AM -> REJECT (1); Remind me to Exercise -> REJECT (1); Pause song -> Next song (1); Reminder Drink water -> Reminders (1); Set color to Green -> REJECT (1); Switch color to Green -> REJECT (1)

## Per-intent scores

| class           | n  | precision | recall | F1    | F2    |
|-----------------|----|-----------|--------|-------|-------|
| ALARM           | 18 | 100.0%    | 66.7%  | 80.0% | 71.4% |
| BRIGHTNESS      | 18 | 100.0%    | 61.1%  | 75.9% | 66.3% |
| CALL            | 6  | 100.0%    | 83.3%  | 90.9% | 86.2% |
| COLOR           | 18 | 100.0%    | 83.3%  | 90.9% | 86.2% |
| CREATE_REMINDER | 18 | 100.0%    | 55.6%  | 71.4% | 61.0% |
| LIGHT_OFF       | 6  | 100.0%    | 50.0%  | 66.7% | 55.6% |
| LIGHT_ON        | 6  | 85.7%     | 100.0% | 92.3% | 96.8% |
| LIST_REMINDERS  | 6  | 71.4%     | 83.3%  | 76.9% | 80.6% |
| MESSAGE         | 6  | 100.0%    | 50.0%  | 66.7% | 55.6% |
| NEXT            | 6  | 62.5%     | 83.3%  | 71.4% | 78.1% |
| PAUSE           | 6  | 100.0%    | 66.7%  | 80.0% | 71.4% |
| PLAY_MUSIC      | 6  | 100.0%    | 50.0%  | 66.7% | 55.6% |
| REJECT          | 16 | 23.0%     | 87.5%  | 36.4% | 56.0% |
| STOP            | 6  | 100.0%    | 33.3%  | 50.0% | 38.5% |
| TEMPERATURE     | 18 | 100.0%    | 88.9%  | 94.1% | 90.9% |
| TIME            | 6  | 100.0%    | 50.0%  | 66.7% | 55.6% |
| TIMER           | 18 | 100.0%    | 88.9%  | 94.1% | 90.9% |
| VOLUME_DOWN     | 6  | 100.0%    | 83.3%  | 90.9% | 86.2% |
| VOLUME_UP       | 6  | 83.3%     | 83.3%  | 83.3% | 83.3% |
| WEATHER         | 6  | 100.0%    | 83.3%  | 90.9% | 86.2% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
