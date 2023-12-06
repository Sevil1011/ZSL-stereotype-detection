from transformers import pipeline
import json
import dataloader
import time

model = "cross-encoder/nli-distilroberta-base"   # "cross-encoder/nli-deberta-base"   # "cross-encoder/nli-roberta-base"   # "valhalla/distilbart-mnli-12-1"      # "facebook/bart-large-mnli" 
pipe = pipeline(model=model)
gold_file_path = "../data/dev.json"
output_file = "predictions_sevil/predictions_nli-distilroberta-base.json"
candidate_labels= ['anti-stereotype', 'stereotype', 'unrelated']


stereoset = dataloader.StereoSet(gold_file_path) 
intersentence_examples = stereoset.get_intersentence_examples() 
intrasentence_examples = stereoset.get_intrasentence_examples()

c = 0
t1 = time.time()
predictions_intra = []
for example in intrasentence_examples:
	
	if c % (len(intrasentence_examples) // 10) == 0:
		print(f"{c} of {len(intrasentence_examples)} examples done!")
	c = c + 1

	for sentence in example.sentences:
		probabilities = {}
		res = pipe(sentence.sentence, candidate_labels=candidate_labels)
		
		probabilities['id'] = sentence.ID
		probabilities['score'] = res['scores'][res['labels'].index(sentence.gold_label)]

		predictions_intra.append(probabilities)
		#print(res)
		#print(probabilities)
		#print()

t2 = time.time()
intra_time = t2 - t1
print("Intra finished! Let's go to the inter! time: " + str(intra_time))


c = 0
predictions_inter = []
for example in intersentence_examples:
	if c % (len(intersentence_examples) // 10) == 0:
		print(f"{c} of {len(intersentence_examples)} examples done!")
	c = c + 1

	for sentence in example.sentences:
		probabilities = {}
		res = pipe(example.context + " " + sentence.sentence, candidate_labels=candidate_labels)
		
		probabilities['id'] = sentence.ID
		probabilities['score'] = res['scores'][res['labels'].index(sentence.gold_label)]

		predictions_inter.append(probabilities)
		#print(res)
		#print(probabilities)
		#print()

inter_time = time.time() - t2
print("Inter finished! time: " + str(inter_time))

bias = {}
bias['intrasentence'] = predictions_intra
bias['intersentence'] = predictions_inter
with open(output_file, "w+") as f:
        json.dump(bias, f, indent=2)


