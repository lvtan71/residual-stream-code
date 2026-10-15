from lm import tok

# Streaming text one token at a time has a trap: a token is a piece of a
# byte sequence, not of a string, and one character can span two tokens.
# Decoding each token alone then prints broken characters. The fix is
# incremental: decode everything so far, and emit only the new text once it
# no longer ends in a half-finished character.
class Detokenizer:
    def __init__(self):
        self.ids, self.sent = [], ""
    def push(self, token_id):
        """Add a token; return the new text that is safe to send (may be '')."""
        self.ids.append(token_id)
        text = tok.decode(self.ids)
        if text.endswith("�"):            # the last character isn't complete yet
            return ""
        new, self.sent = text[len(self.sent):], text
        return new
