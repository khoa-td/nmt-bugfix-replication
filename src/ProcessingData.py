from pathlib import Path
import random
import os


def load_and_split(buggy_path, fixed_path, seed= 42):
    with open(buggy_path, 'r', encoding= 'utf-8') as bug_file:
        bug_lines = [line.strip() for line in bug_file]
    
    with open(fixed_path, 'r', encoding= 'utf-8') as fix_file:
        fix_lines = [line.strip() for line in fix_file]

    assert len(bug_lines) == len(fix_lines)

    seen = set()
    unique_pairs = []
    for b, f in zip(bug_lines, fix_lines):
        if (b, f) not in seen:
            seen.add((b, f))
            if b != f and len(b) > 0 and len(f) > 0:
                unique_pairs.append((b, f))
    

    random.seed(seed)
    random.shuffle(unique_pairs)

    n = len(unique_pairs)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    
    train_pairs = unique_pairs[: n_train]
    valid_pairs = unique_pairs[n_train: n_train + n_valid]
    test_pairs = unique_pairs[n_train + n_valid : ]

    return train_pairs, valid_pairs, test_pairs

def save_pairs_to_files(pairs, split, out_dir):
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    bug_out = os.path.join(split_dir, "buggy.txt")
    fix_out = os.path.join(split_dir, "fixed.txt")

    with open(bug_out, "w", encoding="utf-8") as fb, \
         open(fix_out, "w", encoding="utf-8") as ff:
        for b, f in pairs:
            fb.write(b + "\n")
            ff.write(f + "\n")



class Vocab:
    def __init__(self):
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.token2idx = {self.pad_token: 0, self.sos_token: 1, self.eos_token: 2, self.unk_token: 3}
        self.idx2token = {0: self.pad_token, 1: self.sos_token, 2: self.eos_token, 3: self.unk_token}

    
    def build_vocab(self, lines): # Build vocab from training dataset
        for line in lines:
            for token in line.split():
                if token not in self.token2idx:
                    idx = len(self.token2idx)
                    self.token2idx[token] = idx
                    self.idx2token[idx] = token
    
    def encode(self, sentence, add_sos_eos= True): # Convert token(abstract ID) to number ID
        tokens = sentence.split()
        if add_sos_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        
        unk_id = self.token2idx[self.unk_token]
        return [self.token2idx.get(t, unk_id) for t in tokens]

    def save_vocab(self, path):
        with open(path, 'w', encoding= 'utf-8') as f:
            for token, idx in self.token2idx.items():
                f.write(f"{token}\t{idx}\n")

    def load_vocab(self, path):
        self.token2idx = {}
        self.idx2token = {}

        with open(path, 'r', encoding= 'utf-8') as f:
            for line in f:
                token, idx = line.strip().split("\t")
                self.token2idx[token] = int(idx)
                self.idx2token[int(idx)] = token
    

    def decode(self, indicies): # Convert number ID to abstract ID
        tokens = []
        for idx in indicies:
            if idx == self.token2idx[self.eos_token]:
                break
            if idx not in (self.token2idx[self.pad_token], self.token2idx[self.sos_token]):
                tokens.append(self.idx2token.get(idx, self.unk_token))
        
        return " ".join(tokens)

ROOT = Path(__file__).resolve().parent.parent

if __name__ == '__main__':
    for name in ["small", "medium"]:
        src = ROOT / "myabs" / "out" / name
        out = ROOT / "processed" / name
        buggy, fixed = src / "buggy_all.txt", src / "fixed_all.txt"
        if not (buggy.exists() and fixed.exists()):
            continue

        train, valid, test = load_and_split(str(buggy), str(fixed))
        for split, pairs in [("train", train), ("eval", valid), ("test", test)]:
            save_pairs_to_files(pairs, split, str(out))

        bv, fv = Vocab(), Vocab()
        bv.build_vocab([b for b, _ in train])
        fv.build_vocab([f for _, f in train])
        
        bv.save_vocab(str(out / "train" / "vocab.buggy.txt"))
        fv.save_vocab(str(out / "train" / "vocab.fixed.txt"))

    print("Done!")