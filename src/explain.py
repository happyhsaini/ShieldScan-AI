"""
Explain why a prediction was made by highlighting top influential words.
"""
import re


# Common spam / phishing trigger words (helps build human-readable reason)
SPAM_SIGNALS = [
    "free", "win", "winner", "prize", "claim", "click", "urgent", "limited",
    "offer", "congratulations", "selected", "lucky", "cash", "reward",
    "account", "verify", "password", "bank", "credit", "loan", "investment",
    "guaranteed", "risk free", "act now", "apply now", "dear customer",
    "you have been chosen", "otp", "kyc", "suspicious", "unauthorized",
]

HAM_SIGNALS = [
    "meeting", "schedule", "attached", "please find", "regards", "sincerely",
    "invoice", "receipt", "order", "delivery", "shipped", "confirmed",
    "update", "new features", "release", "digest", "newsletter",
    "unsubscribe", "preferences", "manage",
]


def explain_prediction(message: str, prediction, fake_prob: float, suspicious_prob: float = 0.0) -> dict:
    """
    Return a dict with:
      - label   : "Fake" / "Real" / "Suspicious"
      - reason  : human-readable sentence
      - signals : list of trigger words found
      - tip     : advice for the user
    """
    msg_lower = message.lower()

    def contains_signal(signal):
        if " " in signal:
            return signal in msg_lower
        return bool(re.search(rf"\b{re.escape(signal)}\b", msg_lower))

    found_spam = [word for word in SPAM_SIGNALS if contains_signal(word)]
    found_ham = [word for word in HAM_SIGNALS if contains_signal(word)]

    if prediction == "Fake" or prediction == 1:
        label = "Fake"
        reason = (
            f"The message combines high-risk request patterns with these signals: {', '.join(found_spam[:5])}."
            if found_spam else
            "The message pattern strongly matches the Fake examples in the training data."
        )
        tip = "Do not open links or share codes, passwords, payment details, or personal information. Verify through a known official channel."
        signals = found_spam[:6]
    elif prediction == "Suspicious":
        label = "Suspicious"
        signals = (found_spam + found_ham)[:6]
        reason = (
            "The message has uncertain or mixed signals. Review the sender and verify the request through a channel you already trust."
        )
        if signals:
            reason += f" Signals found: {', '.join(signals[:5])}."
        tip = "Pause before clicking or replying. Check the account or organization directly, and never share a one-time code."
    else:
        label = "Real"
        reason = (
            f"The message contains routine communication signals: {', '.join(found_ham[:5])}."
            if found_ham else
            "The message has no strong fraud pattern and is closest to the Real examples in the training data."
        )
        tip = "No strong scam pattern was detected. Still verify unexpected requests before sharing sensitive information."
        signals = found_ham[:4]

    return {
        "label":   label,
        "reason":  reason,
        "signals": signals,
        "tip":     tip,
    }
