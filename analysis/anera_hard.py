import json
import fire

def analysis(input_file):
    question_type_idct = {}
    cluster = set()
    with open('question.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            input_items = json.loads(line)
            question_type_idct[input_items["question_id"]] = input_items['cluster']
            cluster.add(input_items['cluster'])
    print(len(cluster))
    print(cluster)
    for i in sorted(cluster):
        print(i)
    # return question_type_idct

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            input_items = json.loads(line)
            game = input_items["games"][0]
            # game1
            weight = 0
            if game["score"] == "A=B":
                weight += 0
            elif game["score"] == "A>B":
                weight += -1
            if game["score"] == "A>>B":
                weight += -3
            elif game["score"] == "B>A":
                weight += 1
            elif game["score"] == "B>>A":
                weight += 3
            else:
                weight += 0

            game = input_items["games"][1]
            # game1
            if game["score"] == "A=B":
                weight += 0
            elif game["score"] == "A>B":
                weight += 1
            if game["score"] == "A>>B":
                weight += 3
            elif game["score"] == "B>A":
                weight += -1
            elif game["score"] == "B>>A":
                weight += -3
            else:
                weight += 0
            # print(input_items["question_id"], question_type_idct[input_items["question_id"]].replace(' ', '_'), weight)

if __name__ == '__main__':
    fire.Fire(analysis)