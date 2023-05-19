import stanza
from stanza.models.common.doc import Document, Sentence, Word
from tqdm import tqdm
import csv
from typing import Dict, List
from pathlib import Path
import csv
from settings import TARGET_WORDS


def main():
    target_words = TARGET_WORDS
    nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse")
    target_rows: Dict[str, List] = {target_word: [] for target_word in target_words}
    Sentences_idx: Dict[str,int] = {target_word:0 for target_word in target_words}

    melbert_input_dir = Path("data/cc100-limited/melbert-input")
    melbert_input_dir.mkdir(parents=True, exist_ok=True)

    for target_word in target_words:
        try:
            with (melbert_input_dir / f"{target_word}.tsv").open("r") as f:
                total_lines = sum(1 for line in f)
                Sentences_idx[target_word] = total_lines
        except:
            Sentences_idx[target_word] = 0

    def makeline(sentence: Sentence, idx: int, target_word: str):
        words: List[Word] = sentence.words
        for i,word in enumerate(words):
            if word.deprel == "obj" and words[word.head - 1].lemma == target_word:
                text = " ".join(word.text for word in words)
                row = [f"b1g-fragment02 {idx}", "0", text, "VERB", word.head - 1, i]
                Sentences_idx[target_word] += 1
                return row
        return []

    def chunk_proc(chunk,target_words):
        doc = nlp("\n".join(chunk))
        for sentence in doc.sentences:
            for target_word in target_words:
                if target_word in sentence.text:
                    row = makeline(sentence, Sentences_idx[target_word], target_word)
                    if not row:
                        continue
                    target_rows[target_word].append(row)
                    break

    def save(target_words):
        for target_word in target_words:
            with (melbert_input_dir / f"{target_word}.tsv").open("a") as f:
                writer = csv.writer(f, delimiter="\t")
                for row in target_rows[target_word]:
                    writer.writerow(row)
                target_rows[target_word].clear()

    with open('./data/cc100-limited/melbert-input/now_status.txt','r') as f:
        now_status = int(f.read())
    checkpoint = now_status + 1000000
    count = 0
    chunk = []
    samples_of_chunk = 4096

    with open("/data/data/CC-100/en.txt") as f:
        print(target_words)
        for line in tqdm(f):
            count += 1
            if count < now_status:
                continue
            for target_word in target_words:
                if target_word in line:
                    chunk.append(line)
                    break
            if len(chunk) >= samples_of_chunk:
                chunk_proc(chunk,target_words)
                chunk = []
            if count > checkpoint:
                with open(melbert_input_dir / "now_status-added.txt", "w") as ns:
                    ns.write(f"{checkpoint}\n")
                checkpoint += 1000000
                save(target_words)
            if count >= 1000000000:
                break


if __name__ == "__main__":
    main()
