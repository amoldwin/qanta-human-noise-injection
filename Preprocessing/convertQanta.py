import json
import operator


with open('qanta.train.json','r') as f:
        reformatted_data = {}
        for line in f:
                entry = json.loads(line)

                entry = entry["questions"]
                for question in entry:
                        qid = question["proto_id"]
                        answer = question["page"]
                        reformatted_data[qid] = answer
        outfile = open('byIdQanta.json', 'w+')
        json.dump(reformatted_data, outfile)
                


