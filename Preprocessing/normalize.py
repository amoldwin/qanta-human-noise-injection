import json
import copy

with open('consolidated4.json','r') as f:
    human_data = json.loads(f.read())


filteredData = copy.deepcopy(human_data)
for entry in human_data["questions"]:
    answer = human_data["questions"][entry]["answer"]
    if answer in  human_data["questions"][entry]["guesses"]:
        numcorrect = human_data["questions"][entry]["guesses"][answer]
    else:
        numcorrect = 0
    
    total = 0

    for guess in human_data["questions"][entry]["guesses"]:

        if human_data["questions"][entry]["guesses"][guess] > numcorrect*0.02 :

            total += human_data["questions"][entry]["guesses"][guess]
        else:
            filteredData["questions"][entry]["guesses"].pop(guess)

    for guess in filteredData["questions"][entry]["guesses"]:
        filteredData["questions"][entry]["guesses"][guess] /= total

with open('trainingData.json', 'w') as f:
    f.write(json.dumps(filteredData))
