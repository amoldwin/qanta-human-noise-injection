import json

with open('byIdQantaTest.json','r') as f:
    qanta = json.load(f)
with open('GuessesToQantaAnswersMappingTest.json','r') as f:
    answer_map = json.load(f)
    answer_map.pop("questions")
with open('protobowl.log','r') as f:
    reformatted_data = {"questions": {}}
    i = 1
    str = 'guess'
    for line in f:
        
        entry = json.loads(line)
        qid = entry["object"]["qid"]
        text = entry["object"]["question_text"]
        answer = entry["object"]["answer"]
        guess = entry["object"]["guess"]
        if qid in qanta:
            answer = qanta[qid]
            #Create question if not yet in data
            if qid not in reformatted_data["questions"]:
                reformatted_data["questions"][qid] = {}
                reformatted_data["questions"][qid]["text"] = text 
                reformatted_data["questions"][qid]["answer"] = answer 
                #reformatted_data["questions"][qid]["trueGuesses"]={}
                reformatted_data["questions"][qid]["guesses"]={}
                #reformatted_data["questions"][qid]["promptedGuesses"]={}

            # If user guess was correct, increment that guess's count in the trueGuess set
            # if entry["object"]["ruling"] == True:
            #     if guess not in reformatted_data["questions"][qid]["trueGuesses"]:
            #         reformatted_data["questions"][qid]["trueGuesses"][guess] = 0
            #     reformatted_data["questions"][qid]["trueGuesses"][guess] += 1

            # If user guess was incorrect, increment that guess's count in the falseGuess set. 
            # Later, will increment through all falseGuesse sets, identify if a guess is a correct answer for another question,
            # and merge the count of that incorrect guess as that.
            if entry["object"]["ruling"] == True:
                guess =  answer
                if guess not in reformatted_data["questions"][qid]["guesses"]:
                    reformatted_data["questions"][qid]["guesses"][guess] = 0
                reformatted_data["questions"][qid]["guesses"][guess] += 1          

            if entry["object"]["ruling"] == False:
                if guess in answer_map:
                    guess = answer_map[guess]
                    if guess not in reformatted_data["questions"][qid]["guesses"]:
                        reformatted_data["questions"][qid]["guesses"][guess] = 0
                    reformatted_data["questions"][qid]["guesses"][guess] += 1

            #Same idea as the others, but with promptedGuesses
            # elif entry["object"]["ruling"] == "prompt":
            #     if guess not in reformatted_data["questions"][qid]["promptedGuesses"]:
            #         reformatted_data["questions"][qid]["promptedGuesses"][guess] = 0
            #     reformatted_data["questions"][qid]["promptedGuesses"][guess] += 1
        

        
    outfile = open('consolidated4Test.json', 'w+')
    json.dump(reformatted_data, outfile)