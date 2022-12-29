import os 
import json
import argparse
import csv


def add_trans(fname, zh_fname, es_fname, ofname):
	fr = open(fname)
	data = json.load(fr)

	zh_fr = open(zh_fname)
	zh_data = csv.reader(zh_fr)
	zh_data = list(zh_data)[1:]

	es_fr = open(es_fname)
	es_data = csv.reader(es_fr)
	es_data = list(es_data)[1:]

	zh_j = 0
	es_j = 0
	outputs = []
	for d in data:
		context = d['context'].strip().replace('\n\n', '\n')
		if 'annotations' not in d or d['response']=='' :
			continue
		context_parts = context.split('\n')
		
		context_zh_output = zh_data[zh_j][-1]
		context_es_output = es_data[es_j][-2]
		context_pa_output = es_data[es_j][-1]
		zh_j+=1
		es_j+=1
		for p in context_parts[1:]:
			context_zh_output += '\n' + zh_data[zh_j][-1]
			context_es_output += '\n' + es_data[es_j][-2]
			context_pa_output += '\n' + es_data[es_j][-1]
			zh_j+=1
			es_j+=1
			
		response_zh_output = zh_data[zh_j][-1]
		response_es_output = es_data[es_j][-2]
		response_pa_output = es_data[es_j][-1]
		zh_j+=1
		es_j+=1
		d['context_zh'] = context_zh_output
		d['context_es'] = context_es_output
		d['context_pa'] = context_pa_output
		d['response_zh'] = response_zh_output
		d['response_es'] = response_es_output
		d['response_pa'] = response_pa_output
		outputs.append(d)
		
	with open(ofname, 'w') as fw:
		json.dump(outputs, fw, ensure_ascii=False, indent=6)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_name', default='convai2-grade', type=str, required=True, help="name of input data")
	parser.add_argument('--data_path', default='human_evaluation_data/DSTC10/metadata/', type=str, required=False, help="path of input original data")
	parser.add_argument('--zh_data_path', default='human_evaluation_data/DSTC10/en_zh/', type=str, required=False, help="path of Chinese translated data")
	parser.add_argument('--es_data_path', default='human_evaluation_data/DSTC10/en_es_en/', type=str, required=False, help="path of Spanish translated data")

	args=parser.parse_args()
	fname = os.path.join(args.data_path, f'{args.data_name}/{args.data_name}_eval.json')
	zh_fname = os.path.join(args.zh_data_path, f'{args.data_name}/{args.data_name}_eval_main_en_zh.csv')
	es_fname = os.path.join(args.es_data_path, f'{args.data_name}/{args.data_name}_eval_main_en_es_en.csv')
	ofname = fname.replace('_eval.json', '_eval_zh_es_pa.json')
	add_trans(fname, zh_fname, es_fname, ofname)
