import json

def view_intent_train():
    dataset       = json.load(open("./data/intent/eval.json"))
    max_text_len  = 0
    intent        = {}
    split_intents = {}
    for data in dataset:
        if len(data['text'].split()) > max_text_len: max_text_len = len(data['text'].split())
        if data['intent'] not in intent.keys(): intent[data['intent']] = 1
        else                                  : intent[data['intent']] += 1
        for split_intent in data['intent'].split('_'):
            if split_intent not in split_intents.keys(): split_intents[split_intent] = 1
            else                                       : split_intents[split_intent] += 1
    print(f"max_text_len     : {max_text_len}")        # 28
    print(f"dataset_len      : {len(dataset)}")        # 15000
    print(f"intent_len       : {len(intent)}")         # 150
    print(f"split_intents_len: {len(split_intents)}")  # 191
    return

def view_slot_train():
    dataset       = json.load(open("./data/slot/test.json"))
    max_text_len  = 0
    tags          = []
    for data in dataset:
        if len(data['tokens']) > max_text_len: max_text_len = len(data['tokens'])
        for tag in data['tags']:
            if tag not in tags: tags.append(tag)
    print(f"max_text_len: {max_text_len}")  # 35
    print(f"dataset_len : {len(dataset)}")  # 7244
    print(f"tags_len    : {len(tags)}")     # 9
    return

if __name__ == '__main__':
    # view_intent_train()
    view_slot_train()