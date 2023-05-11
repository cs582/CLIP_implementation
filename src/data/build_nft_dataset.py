import pandas as pd
import json

def create_description(t, accessories):
    if len(accessories) == 0:
        a_desc = f"nothing else"
    elif len(accessories) == 1:
        a_desc = f"{accessories[0]}"
    else:
        a_desc = ", ".join(accessories[:-1]) + " and " + accessories[-1]

    return f"An NFT of a {t} with {a_desc}"

def build():

    # Load data
    nfts = { f"{idx}.png": None for idx in range(0, 10000) }
    with open("data/cryptopunks/txn_history-2021-10-07.jsonl", "r") as f:
        for line in f:
            transact = json.loads(line)
            img, t, des = transact['punk_id'], transact['type'][0], transact['accessories']
            nfts[f"{img}.png"] = create_description(t, des)

    df = pd.DataFrame([ [y, x] for x, y in nfts.items()], columns=["query", "img"])
    print(df.shape)
    print(df.head())
    return