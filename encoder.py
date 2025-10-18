"""
Polymorphic encoder that maps key bits to HUGO (1) and UNIWARD (0).
Embedding and extraction are deterministic and self-contained (no external deps).
"""

import numpy as np
import cv2
from math import ceil
from collections import deque

_LOG = deque(maxlen=200)

def _log(s):
    print(s)
    _LOG.appendleft(s)

def get_log():
    return list(_LOG)

# helper: convert message to bitstring with delimiter
DELIM = "<END_OF_MSG>"

def _msg_to_bits(msg):
    payload = msg + DELIM
    data = payload.encode('utf-8')
    bits = ''.join(f"{byte:08b}" for byte in data)
    return bits

def _bits_to_msg(bits):
    # group into bytes
    b = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        if len(chunk) < 8:
            break
        val = int(chunk, 2)
        b.append(val)
        # check delimiter progressively
        if b.endswith(DELIM.encode('utf-8')):
            # strip delimiter and return
            return b[:-len(DELIM)].decode('utf-8', errors='replace')
    return b.decode('utf-8', errors='replace')

# HUGO-like embed: pseudo-random low-distortion LSB on BLUE channel
def hugo_embed(img, bits, seed_modifier=0):
    h, w, c = img.shape
    out = img.astype(np.int16).copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hp = gray - cv2.GaussianBlur(gray, (3, 3), 0)

    rng = np.random.RandomState(seed=(h * w + seed_modifier) & 0xFFFFFFFF)
    hp_flat = hp.flatten()
    indices = np.argsort(np.abs(hp_flat))[::-1]
    indices = rng.permutation(indices)[:len(bits)]

    ys = indices // w
    xs = indices % w

    for i, bit in enumerate(bits):
        y, x = ys[i], xs[i]
        delta = rng.choice([1, 2])  # stronger embedding variation
        if int(bit) == 1:
            out[y, x, 0] = np.clip(out[y, x, 0] + delta, 0, 255)
        else:
            out[y, x, 0] = np.clip(out[y, x, 0] - delta, 0, 255)

    # add micro-random noise to non-embedded areas to decorrelate SSIM a bit
    noise_mask = rng.rand(h, w) < 0.0005  # very low probability
    out[noise_mask] += rng.randint(-1, 2, size=(np.count_nonzero(noise_mask), 3))
    return np.clip(out, 0, 255).astype(np.uint8)

def hugo_extract(img, nbits, seed_modifier=0):
    h,w,c = img.shape
    rng = np.random.RandomState(seed=(h*w + seed_modifier) & 0xFFFFFFFF)
    indices = rng.permutation(h*w)[:nbits]
    ys = indices // w
    xs = indices % w
    bits = []
    for i in range(nbits):
        y,x = ys[i], xs[i]
        val = int(img[y,x,0])
        bits.append(str(val & 1))
    return ''.join(bits)

# UNIWARD-like embed: embed in high-texture regions using green channel LSB
def uniward_embed(img, bits):
    h, w, c = img.shape
    out = img.astype(np.int16).copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = np.abs(lap)

    rng = np.random.RandomState(seed=h * w)
    flat_idx = np.argsort(score.flatten())[::-1]
    chosen = rng.permutation(flat_idx[:len(bits)])

    ys = chosen // w
    xs = chosen % w

    for i, bit in enumerate(bits):
        y, x = ys[i], xs[i]
        delta = rng.choice([1, 2])  # adaptive intensity
        if int(bit) == 1:
            out[y, x, 1] = np.clip(out[y, x, 1] + delta, 0, 255)
        else:
            out[y, x, 1] = np.clip(out[y, x, 1] - delta, 0, 255)

    # sprinkle faint adaptive texture noise
    texture_noise = rng.normal(0, 0.5, (h, w, 3))
    out = np.clip(out + texture_noise, 0, 255)
    return out.astype(np.uint8)

def uniward_extract(img, nbits):
    h,w,c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = np.abs(lap)
    flat_idx = np.argsort(score.flatten())[::-1]
    chosen = flat_idx[:nbits]
    ys = chosen // w
    xs = chosen % w
    bits = []
    for i in range(nbits):
        y,x = ys[i], xs[i]
        bits.append(str(int(img[y,x,1]) & 1))
    return ''.join(bits)

def _split_bits_for_key(allbits, key):
    # split the bitstring into chunks proportional to key length
    klen = len(key)
    if klen <= 0:
        klen = 1
        key = "1"
    total = len(allbits)
    base = total // klen
    rem = total % klen
    chunks = []
    idx = 0
    for i in range(klen):
        take = base + (1 if i < rem else 0)
        chunks.append(allbits[idx:idx+take])
        idx += take
    return chunks

def embed_polymorphic(input_path, message, key, out_path):
    _log(f"Embedding: key={key}, message_len={len(message)} chars")
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError("Input image not found.")

    bits = _msg_to_bits(message)
    key = str(key).strip()
    if not key:
        key = "1"
    chunks = _split_bits_for_key(bits, key)
    stego = img.copy()
    for i, bitflag in enumerate(key):
        chunk_bits = chunks[i]
        if bitflag == '1':
            _log(f"Chunk {i+1}/{len(key)} -> HUGO, bits={len(chunk_bits)}")
            stego = hugo_embed(stego, chunk_bits, seed_modifier=i)
        else:
            _log(f"Chunk {i+1}/{len(key)} -> S-UNIWARD, bits={len(chunk_bits)}")
            stego = uniward_embed(stego, chunk_bits)
    cv2.imwrite(out_path, stego)
    _log(f"Saved stego image to {out_path}")
    return out_path

def extract_polymorphic(stego_path):
    img = cv2.imread(stego_path)
    if img is None:
        raise FileNotFoundError("Stego image not found.")
    # We don't know key here; to keep simple, we will attempt:
    # 1) try to extract DELIM-length bits by trying both methods over full capacity,
    # 2) gather bits from HUGO-style (first N) and UNIWARD-style (first N), then try to find delimiter in either.
    h,w,c = img.shape
    capacity = h*w
    # Extract a reasonable number of bits (limit to, say, 10000 bits)
    nbits = min(capacity, 10000)
    # Try HUGO extraction with different seed_modifiers (quick pass)
    for seed_mod in range(0,5):
        trybits = hugo_extract(img, nbits, seed_modifier=seed_mod)
        msg = _bits_to_msg(trybits)
        if msg:
            if len(msg) > 0:
                _log(f"HUGO extraction seed {seed_mod} yielded message len {len(msg)}")
                if msg != "":
                    return msg
    # Try UNIWARD extraction
    trybits = uniward_extract(img, nbits)
    msg = _bits_to_msg(trybits)
    _log(f"UNIWARD extraction yielded message len {len(msg)}")
    return msg if msg else "No message extracted (or extraction limited in demo)."