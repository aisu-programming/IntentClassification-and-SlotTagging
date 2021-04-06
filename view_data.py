import json

def view_intent_train():
    dataset       = json.load(open("./data/intent/train.json"))
    max_text_len  = 0
    intent        = {}
    split_intents = {}
    for data in dataset:
        if len(data['text']) > max_text_len: max_text_len = len(data['text'])
        if data['intent'] not in intent.keys(): intent[data['intent']] = 1
        else                                  : intent[data['intent']] += 1
        for split_intent in data['intent'].split('_'):
            if split_intent not in split_intents.keys(): split_intents[split_intent] = 1
            else                                       : split_intents[split_intent] += 1
    print(f"max_text_len     : {max_text_len}")        # 15000
    print(f"dataset_len      : {len(dataset)}")        # 15000
    print(f"intent_len       : {len(intent)}")         # 150
    print(f"split_intents_len: {len(split_intents)}")  # 191
    return

if __name__ == '__main__':
    view_intent_train()