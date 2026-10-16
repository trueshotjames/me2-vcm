# voice_pipeline/intent_parser.py
#
# The command grammar, the text parser and the VCM beam search.
#
# Each command is written once as a pattern (PATTERNS below). At import, all patterns are
# compiled into one deterministic word graph. The same graph is used by
#   - parse_text(): typed or decoded text -> (action, value)
#   - BeamDecoder / decode_command(): CTC beam search over the VCM scores, limited to
#     word sequences the graph allows.
# Slot values (numbers, colours, ...) are read from the words at the end (extract_value),
# so the graph does not need a separate branch for every value.

import math

import numpy as np

# --- Words that may be dropped from, or added inside, any command ("turn off lights") ---
FUNCTION_WORDS = {"the", "a", "an", "my", "me", "to", "for", "at", "is", "it", "some", "are"}

# --- Numbers ---
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n):
    """0-100 -> 'twenty two' style words."""
    if n == 100:
        return "one hundred"
    if n < 20:
        return _ONES[n]
    return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])


def words_to_number(words, i=0):
    """Reads a number written in words starting at words[i] -> (value, next index), or (None, i)."""
    if i < len(words) and words[i] == "hundred":
        return 100, i + 1
    if i + 1 < len(words) and words[i] == "one" and words[i + 1] == "hundred":
        return 100, i + 2
    if i < len(words) and words[i] in _ONES:
        return _ONES.index(words[i]), i + 1
    if i < len(words) and words[i] in _TENS[2:]:
        value = _TENS.index(words[i]) * 10
        if i + 1 < len(words) and words[i + 1] in _ONES[1:10]:
            return value + _ONES.index(words[i + 1]), i + 2
        return value, i + 1
    return None, i


TEMP_RANGE = range(16, 31)         # degrees Celsius
BRIGHTNESS_RANGE = range(0, 101)   # percent
TIMER_RANGE = range(1, 60)         # seconds or minutes
HOUR_RANGE = range(1, 13)          # alarm hour (with am/pm)
MINUTE_RANGE = range(1, 60)        # alarm minutes, e.g. "six thirty am"

# --- Slots: each is a list of word sequences ---
SLOTS = {
    "TEMP": [number_to_words(n) for n in TEMP_RANGE],
    "PCT": [number_to_words(n) for n in BRIGHTNESS_RANGE] + ["hundred"],
    "N59": [number_to_words(n) for n in TIMER_RANGE],
    "HOUR": [number_to_words(n) for n in HOUR_RANGE],
    "MIN": [number_to_words(n) for n in MINUTE_RANGE],
    "UNIT": ["second", "seconds", "minute", "minutes"],
    "MUNIT": ["minute", "minutes"],
    "SUNIT": ["second", "seconds"],
    "COLOR": ["red", "blue", "green"],
    "TASK": ["drink water", "study", "exercise", "pay bills"],
    "PERSON": ["mom", "dad"],
    "LIGHT": ["light", "lights"],
    "TURN": ["turn", "switch", "power"],
    "TURNOFF": ["turn", "switch", "power", "shut"],
    "WHAT": ["what's", "what is"],
    "AMPM": ["am", "pm"],
}

# --- Command patterns: (action, pattern) ---
# Pattern words are separated by spaces. a|b = either word, [a|b] = optional, {SLOT} = slot.
# Words in FUNCTION_WORDS are always optional, and any of them may also appear between words.
PATTERNS = [
    ("music_play", "play|start [some] music|song|playback"),
    ("music_play", "start playing [music|song]"),
    ("music_pause", "pause [music|audio|song|playback]"),
    ("music_stop", "stop [music|audio|song|playing|playback]"),
    ("music_stop", "end [music|song|playback]"),
    ("music_skip", "[play] next [song]"),
    ("music_skip", "skip [song]"),
    ("music_vol_up", "volume up|louder"),
    ("music_vol_up", "turn|switch [the] volume up"),
    ("music_vol_up", "turn up [the] volume"),
    ("music_vol_up", "increase [the] volume"),
    ("music_vol_up", "louder"),
    ("music_vol_down", "volume down|quieter|lower"),
    ("music_vol_down", "turn|switch [the] volume down"),
    ("music_vol_down", "turn down [the] volume"),
    ("music_vol_down", "lower [the] volume"),
    ("music_vol_down", "quieter"),
    ("weather", "[{WHAT}] [the] weather"),
    ("weather", "tell [me] [the] weather"),
    ("time", "time"),
    ("time", "{WHAT} [the] time"),
    ("time", "what time [is] [it]"),
    ("time", "tell [me] [the] time"),
    ("toggle", "{TURN} on [the] {LIGHT}"),
    ("toggle", "{TURN} [the] {LIGHT} on"),
    ("toggle", "{LIGHT} on"),
    ("toggle", "{TURNOFF} off [the] {LIGHT}"),
    ("toggle", "{TURNOFF} [the] {LIGHT} off"),
    ("toggle", "kill [the] {LIGHT}"),
    ("toggle", "{LIGHT} off|out"),
    ("dim", "[set|adjust|change] [the] brightness [level] [to] {PCT} percent"),
    ("dim", "dim [the] {LIGHT} [to] {PCT} percent"),
    ("temp", "[set|change] [the] temperature [to] {TEMP} degrees"),
    ("color", "[change|switch|set] [the] color [to] {COLOR}"),
    ("color", "change|switch|set|turn [the] {LIGHT} [color] [to] {COLOR}"),
    ("timer", "[set|start] [a] timer [for] {N59} {UNIT}"),
    ("timer", "[set|start] [a] timer [for] {N59} {MUNIT} {N59} {SUNIT}"),
    ("timer", "countdown [for] {N59} {UNIT}"),
    ("timer", "countdown [for] {N59} {MUNIT} {N59} {SUNIT}"),
    ("alarm", "[set] [an] alarm [for] [at] {HOUR} [o'clock] {AMPM}"),
    ("alarm", "[set] [an] alarm [for] [at] {HOUR} {MIN} {AMPM}"),
    ("alarm", "wake [me] up [at] {HOUR} [o'clock] {AMPM}"),
    ("alarm", "wake [me] up [at] {HOUR} {MIN} {AMPM}"),
    ("reminder", "remind [me] [to] {TASK}"),
    ("reminder", "[create|set] [a] reminder [to] {TASK}"),
    ("get_reminders", "[show|list|tell] [me] [my] reminders"),
    ("get_reminders", "what [are] [my] reminders"),
    ("call_mom", "[make] [a] [phone] call [{PERSON}]"),
    ("call_mom", "call|phone {PERSON}"),
    ("message", "[send] [a|my] message"),
]

# What the UI suggests ("What can I say?"), grouped by command (exactly the phrase list in variations.csv)
GUIDE = {
    "PLAY_MUSIC": ["Play music", "Start music", "Play some music"],
    "WEATHER": ["Weather", "What's the weather?", "Tell me the weather"],
    "TIME": ["Time", "What time is it?", "Tell me the time"],
    "LIGHT_ON": ["Lights on", "Power on the lights", "Turn on the lights"],
    "LIGHT_OFF": ["Lights out", "Kill the lights", "Shut off the lights"],
    "PAUSE": ["Pause", "Pause audio", "Pause song"],
    "STOP": ["Stop", "Stop playing", "End playback"],
    "NEXT": ["Next song", "Skip song", "Play next song"],
    "VOLUME_UP": ["Volume up", "Increase the volume", "Turn the volume up"],
    "VOLUME_DOWN": ["Volume down", "Lower the volume", "Turn the volume down"],
    "CALL": ["Call", "Make a call", "Make a phone call"],
    "MESSAGE": ["Message", "Send a message", "Send my message"],
    "LIST_REMINDERS": ["Reminders", "Show my reminders", "List my reminders"],
    "TIMER": ["Timer 10 seconds", "Timer 30 seconds", "Timer 1 minute", "Countdown for 10 seconds", "Countdown for 30 seconds", "Countdown for 1 minute", "Start a timer for 10 seconds", "Start a timer for 30 seconds", "Start a timer for 1 minute"],
    "ALARM": ["Alarm 6:00 AM", "Alarm 8:00 AM", "Alarm 9:00 PM", "Wake me up at 6:00 AM", "Wake me up at 8:00 AM", "Wake me up at 9:00 PM", "Set an alarm for 6:00 AM", "Set an alarm for 8:00 AM", "Set an alarm for 9:00 PM"],
    "TEMPERATURE": ["Temperature 18 degrees", "Temperature 22 degrees", "Temperature 26 degrees", "Change the temperature to 18 degrees", "Change the temperature to 22 degrees", "Change the temperature to 26 degrees", "Set the temperature to 18 degrees", "Set the temperature to 22 degrees", "Set the temperature to 26 degrees"],
    "BRIGHTNESS": ["Brightness 20 percent", "Brightness 60 percent", "Brightness 100 percent", "Adjust brightness to 20 percent", "Adjust brightness to 60 percent", "Adjust brightness to 100 percent", "Brightness level 20 percent", "Brightness level 60 percent", "Brightness level 100 percent"],
    "COLOR": ["Change color to Red", "Change color to Blue", "Change color to Green", "Switch color to Red", "Switch color to Blue", "Switch color to Green", "Set color to Red", "Set color to Blue", "Set color to Green"],
    "CREATE_REMINDER": ["Reminder Drink water", "Reminder Study", "Reminder Exercise", "Remind me to Drink water", "Remind me to Study", "Remind me to Exercise", "Create a reminder to Drink water", "Create a reminder to Study", "Create a reminder to Exercise"],
}


def command_guide():
    """Phrases the UI can suggest, grouped by command."""
    return {action: list(phrases) for action, phrases in GUIDE.items()}


# --- Pattern compiler: patterns -> one deterministic word graph ---

def _compile(patterns):
    """Returns (arcs, final, can_continue): arcs[state] = {word: next state}; state 0 is the start;
    final[state] = action of the first pattern that ends there (or None)."""
    nfa = [[]]   # nfa[state] = list of (word or None, next state); None = no word needed
    ends = {}    # nfa end state -> pattern index

    def new():
        nfa.append([])
        return len(nfa) - 1

    def add_seq(start, words):
        s = start
        for w in words:
            t = new()
            nfa[s].append((w, t))
            s = t
        return s

    for pi, (_, pattern) in enumerate(patterns):
        s = 0
        tokens = pattern.split()
        for ti, tok in enumerate(tokens):
            optional = tok.startswith("[")
            tok = tok.strip("[]")
            if tok.startswith("{"):
                choices = [c.split() for c in SLOTS[tok.strip("{}")]]
            else:
                choices = [[w] for w in tok.split("|")]
            if all(len(c) == 1 and c[0] in FUNCTION_WORDS for c in choices):
                optional = True
            t = new()
            for c in choices:
                e = add_seq(s, c[:-1])
                nfa[e].append((c[-1], t))
            if optional:
                nfa[s].append((None, t))
            s = t
            if ti < len(tokens) - 1:   # small words may appear between words, not after the last one
                for fw in FUNCTION_WORDS:
                    nfa[s].append((fw, s))
        ends.setdefault(s, pi)

    def closure(states):
        stack, seen = list(states), set(states)
        while stack:
            for w, t in nfa[stack.pop()]:
                if w is None and t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)

    start = closure([0])
    ids, order, arcs = {start: 0}, [start], []
    while len(arcs) < len(order):
        cur = order[len(arcs)]
        moves = {}
        for s in cur:
            for w, t in nfa[s]:
                if w is not None:
                    moves.setdefault(w, set()).add(t)
        out = {}
        for w, ts in moves.items():
            nxt = closure(ts)
            if nxt not in ids:
                ids[nxt] = len(order)
                order.append(nxt)
            out[w] = ids[nxt]
        arcs.append(out)
    final = []
    for st in order:
        hits = [ends[s] for s in st if s in ends]
        final.append(patterns[min(hits)][0] if hits else None)
    # A command "can continue" if a further (non-loop) word leads somewhere ("pause" -> "pause song")
    can_continue = [any(t != i for t in arcs[i].values()) for i in range(len(arcs))]
    return arcs, final, can_continue


GRAPH_ARCS, GRAPH_FINAL, GRAPH_CAN_CONTINUE = _compile(PATTERNS)


def extract_value(action, words):
    """Reads the slot value from the matched words -> (action, value)."""
    if action == "toggle":
        return action, "off" if any(w in ("off", "out", "kill", "shut") for w in words) else "on"
    if action in ("dim", "temp"):
        for i in range(len(words)):
            n, _ = words_to_number(words, i)
            if n is not None:
                return action, n
    if action == "color":
        return action, next(w for w in words if w in SLOTS["COLOR"])
    if action == "reminder":
        text = " ".join(words)
        return action, next(t for t in SLOTS["TASK"] if t in text)
    if action == "call_mom":
        return ("call_dad" if "dad" in words else "call_mom"), None
    if action == "timer":
        minutes = seconds = 0
        i = 0
        while i < len(words):
            n, j = words_to_number(words, i)
            if n is not None and j < len(words) and words[j] in SLOTS["UNIT"]:
                if words[j].startswith("minute"):
                    minutes += n
                else:
                    seconds += n
                i = j + 1
            else:
                i += 1
        return action, {"minutes": minutes, "seconds": seconds}
    if action == "alarm":
        i = next(k for k, w in enumerate(words) if w in SLOTS["HOUR"])
        hour = _ONES.index(words[i])
        minute, _ = words_to_number(words, i + 1)
        pm = "pm" in words
        return action, "{:02d}:{:02d}:00".format(hour % 12 + (12 if pm else 0), minute or 0)
    return action, None


def match_words(words):
    """Runs words through the grammar -> (action, value), or ("Unrecognized", None)."""
    state = 0
    for w in words:
        nxt = GRAPH_ARCS[state].get(w)
        if nxt is None:
            if w in FUNCTION_WORDS:   # e.g. a leading "the"
                continue
            return "Unrecognized", None
        state = nxt
    action = GRAPH_FINAL[state]
    if action is None:
        return "Unrecognized", None
    return extract_value(action, words)


def parse_text(text):
    # Clean the incoming text
    text = text.replace('[BLANK]', '').replace('[blank]', '').strip().lower()
    text = ' '.join(text.replace('[unk]', ' ').split())   # ignore [UNK] tokens
    # Remove common punctuation so variations match
    text = text.replace('?', '').replace('.', '').replace(',', '').replace('!', '')
    # Typed digits -> words, e.g. "set the temperature to 24 degrees", "alarm 6:30 am"
    words = []
    for w in text.replace(':00', '').replace(':', ' ').split():
        words.extend(number_to_words(int(w)).split() if w.isdigit() and int(w) <= 100 else [w])
    return match_words(words)


# --- VCM grammar beam search (Pi Voice Command Blueprint, Stage 6) ---
# The VCM emits one score vector per 40 ms step (CTC: words + [UNK] + blank).
# The beam search follows the word graph above, with an [UNK] filler loop before, inside and
# after the command ("turn on the [UNK] lights" -> "turn on the lights"). If the best command
# scores worse than "all filler", the audio is rejected and '' is returned (parse_text then
# gives "Unrecognized").
# Speed: each step, only labels the model gives a real chance (top LABEL_TOP_K with log-prob
# above LABEL_MIN_LOGP) are tried as new words, so the work per step does not grow with the grammar.

LABEL_TOP_K = 8
LABEL_MIN_LOGP = -12.0

_PRE = -1          # grammar state before any command word
_NEG_INF = float('-inf')


def _logaddexp(a, b):
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class BeamDecoder:
    """Grammar-constrained CTC beam search that takes the VCM's steps as they arrive.

    push(log_probs) adds [steps, classes] log-softmax rows; status() says where the best
    hypothesis is in the grammar (used to stop recording as soon as a command is finished);
    result() gives the decoded command text over all pushed steps.
    """

    def __init__(self, vocab, blank_id, unk_id, beam_width=16):
        self.words = [w.lower() for w in vocab]
        self.blank_id, self.unk_id, self.beam_width = blank_id, unk_id, beam_width
        self.function_ids = {i for i, w in enumerate(self.words) if w in FUNCTION_WORDS}
        self.steps = 0
        # Hypothesis key: (grammar state, command word ids, last emitted label)
        #   -> [log P(ends in blank), log P(ends in label)]
        # Hypotheses with the same key have the same future, so they are merged.
        self.hyps = {(_PRE, (), None): [0.0, _NEG_INF]}

    def copy(self):
        """A copy that can be pushed further without changing this decoder."""
        other = BeamDecoder.__new__(BeamDecoder)
        other.__dict__.update(self.__dict__)
        return other   # push() replaces hyps instead of changing it, so sharing it is safe

    def push(self, log_probs):
        blank_id, unk_id, beam_width = self.blank_id, self.unk_id, self.beam_width
        words, function_ids = self.words, self.function_ids
        hyps = self.hyps
        rows = np.asarray(log_probs, dtype=np.float64)
        if len(rows) == 0:
            return self
        k = min(LABEL_TOP_K + 1, rows.shape[1])
        tops = np.argpartition(-rows, k - 1, axis=1)[:, :k].tolist()
        for row, top in zip(rows.tolist(), tops):
            lp = row
            # Labels worth trying as a new word this step
            cands = [c for c in top if c != blank_id and lp[c] > LABEL_MIN_LOGP]
            new = {}

            def add(k, idx, val):
                cur = new.get(k)
                if cur is None:
                    cur = new[k] = [_NEG_INF, _NEG_INF]
                cur[idx] = _logaddexp(cur[idx], val)

            for key, (pb, pnb) in hyps.items():
                state, seq, last = key
                ptot = _logaddexp(pb, pnb)
                add(key, 0, ptot + lp[blank_id])                 # blank
                if last is not None:
                    add(key, 1, pnb + lp[last])                  # repeated label, no new word
                for c in cands:
                    if c == unk_id or (state == _PRE and c in function_ids):
                        nkey = (state, seq, c)                   # filler: [UNK], or a small word before the command
                    else:
                        nxt = GRAPH_ARCS[0 if state == _PRE else state].get(words[c])
                        if nxt is None:
                            continue
                        nkey = (nxt, seq + (c,), c)
                    add(nkey, 1, (pb if c == last else ptot) + lp[c])   # same label again needs a blank between
            ranked = sorted(new.items(), key=lambda kv: _logaddexp(*kv[1]), reverse=True)
            # Keep the top hypotheses, and always keep the filler-only ones for the rejection test.
            hyps = dict(ranked[:beam_width])
            hyps.update((k, v) for k, v in ranked[beam_width:] if k[0] == _PRE)
            self.steps += 1
        self.hyps = hyps
        return self

    def _scores(self):
        """Filler score, and complete commands ranked by meaning: [(text, score, (action, value))]."""
        filler_score = _NEG_INF
        by_meaning = {}
        for (state, seq, _), (pb, pnb) in self.hyps.items():
            score = _logaddexp(pb, pnb)
            if state == _PRE:
                filler_score = _logaddexp(filler_score, score)
            elif GRAPH_FINAL[state] is not None:
                ws = [self.words[c] for c in seq]
                meaning = extract_value(GRAPH_FINAL[state], ws)
                key = repr(meaning)
                text, best, total = by_meaning.get(key, (None, _NEG_INF, _NEG_INF))
                if score > best:
                    text, best = " ".join(ws), score
                by_meaning[key] = (text, best, _logaddexp(total, score))
        ranked = sorted(((t, total) for t, _, total in by_meaning.values()), key=lambda kv: kv[1], reverse=True)
        return ranked, filler_score

    def status(self):
        """Where the single best hypothesis is right now.

        state: 'filler'     - no command started (only [UNK]/blank so far)
               'unfinished' - inside a command that still needs more words
               'complete'   - a whole command; can_continue says if a longer one starts with it
        phrase: the best complete command if it beats the filler path, else ''
        """
        state = max(self.hyps.items(), key=lambda kv: _logaddexp(*kv[1]))[0][0]
        ranked, filler_score = self._scores()
        best = ranked[0][0] if ranked and ranked[0][1] > filler_score else ''
        margin = ranked[0][1] - filler_score if ranked and filler_score != _NEG_INF else None
        if state == _PRE:
            kind = 'filler'
        elif GRAPH_FINAL[state] is None:
            kind = 'unfinished'
        else:
            kind = 'complete'
        return {"state": kind, "can_continue": state != _PRE and GRAPH_CAN_CONTINUE[state],
                "phrase": best, "margin": margin}

    def result(self, return_details=False):
        ranked, filler_score = self._scores()
        best = ranked[0][0] if ranked and ranked[0][1] > filler_score else ''
        if not return_details:
            return best
        finite = lambda v: None if v == _NEG_INF else round(v, 2)
        details = {
            "top": [[p, finite(s)] for p, s in ranked[:3]],
            "filler": finite(filler_score),
            "margin": finite(ranked[0][1] - filler_score) if ranked and filler_score != _NEG_INF else None,
        }
        return best, details


def decode_command(log_probs, vocab, blank_id, unk_id, beam_width=16, return_details=False):
    """Grammar-constrained CTC beam search.

    log_probs: [steps, classes] log-softmax scores from the VCM.
    Returns the matched command text (parse_text() turns it into an action), or '' if rejected.
    With return_details=True, returns (text, details): the top 3 commands (by meaning) with
    their log scores and the best one's margin over the filler-only path.
    """
    return BeamDecoder(vocab, blank_id, unk_id, beam_width).push(log_probs).result(return_details)
