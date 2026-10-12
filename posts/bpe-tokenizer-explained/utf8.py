"""UTF-8: how characters become bytes, and why 256 symbols cover everything."""

from collections import Counter

for ch in ("s", "é", "à", "牛", "🙃"):
    b = ch.encode("utf-8")
    print(f"{ch}  U+{ord(ch):04X}  {len(b)} byte{'s' * (len(b) > 1)}: "
          f"{' '.join(f'{x:02X}' for x in b):12s} = {list(b)}")
for s in ("héllo", "Xin chào"):
    print(f"{s!r}: {len(s)} characters, {len(s.encode())} bytes {list(s.encode())}")

# Every code point that UTF-8 can encode (surrogates are excluded by the standard).
by_len, low_inside = Counter(), 0
for cp in range(0x110000):
    if 0xD800 <= cp <= 0xDFFF:
        continue
    b = chr(cp).encode("utf-8")
    by_len[len(b)] += 1
    if len(b) > 1:
        low_inside += sum(x < 0x80 for x in b)
print("\ncode points by UTF-8 length:",
      ", ".join(f"{n} byte{'s' * (n > 1)}: {by_len[n]:,}" for n in sorted(by_len)))
print(f"total {sum(by_len.values()):,}; bytes below 0x80 inside a multi-byte character: "
      f"{low_inside}")
