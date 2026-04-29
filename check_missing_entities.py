# Script to check for missing entity IDs in WN18RR splits

def load_entity_ids_from_definitions(def_path):
    ids = set()
    with open(def_path, encoding='utf-8') as f:
        for line in f:
            fs = line.strip().split('\t')
            if len(fs) == 3:
                ids.add(fs[0])
    return ids

def load_entity_ids_from_split(split_path):
    ids = set()
    with open(split_path, encoding='utf-8') as f:
        for line in f:
            fs = line.strip().split('\t')
            if len(fs) == 3:
                ids.add(fs[0])
                ids.add(fs[2])
    return ids

if __name__ == "__main__":
    def_path = "data/WN18RR/wordnet-mlj12-definitions.txt"
    split_files = ["data/WN18RR/train.txt", "data/WN18RR/valid.txt", "data/WN18RR/test.txt"]
    def_ids = load_entity_ids_from_definitions(def_path)
    for split in split_files:
        split_ids = load_entity_ids_from_split(split)
        missing = split_ids - def_ids
        if missing:
            print(f"Missing in {split}: {missing}")
        else:
            print(f"All entity IDs in {split} are present in definitions.")
