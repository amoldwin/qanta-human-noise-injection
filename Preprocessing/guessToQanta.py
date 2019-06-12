import json
import operator


with open('protobowl.log','r') as f:
        with open('byIdQantaTest.json','r') as qanta:
                for line in qanta:
                        qantalist = json.loads(line)

                reformatted_data = {"questions": {}}
                i = 1
                for line in f:
                        entry = json.loads(line)
                        qid = entry["object"]["qid"]
                        if qid not in qantalist:
                                continue

                        text = entry["object"]["question_text"]
                        guess = entry["object"]["guess"]
                        ruling = entry["object"]["ruling"]
                        if ruling == True:
                            reformatted_data[guess] = qantalist[qid]

                    
                outfile = open('GuessesToQantaAnswersMappingTest.json', 'w+')
                json.dump(reformatted_data, outfile)
                

