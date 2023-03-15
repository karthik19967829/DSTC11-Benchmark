import os 
import json
import argparse
import pandas as pd
import shutil
import ast


def best_paraphrases(pa_data):
	pa_data['PARAPHRASES'] = pa_data['PARAPHRASES'].apply(ast.literal_eval)
	pa_data['COS_SIM_MONO_SP'] = pa_data['COS_SIM_MONO_SP'].apply(ast.literal_eval)

	pa_data_list = []
	for i in range(len(pa_data)):
		pa_list = pa_data['PARAPHRASES'][i]
		cos_sim_list = pa_data['COS_SIM_MONO_SP'][i]

		pa = None
		for i in range(len(pa_list)):
			max_value = max(cos_sim_list)
			max_index = cos_sim_list.index(max_value)
			if max_value == 1:
				del pa_list[max_index]
				del cos_sim_list[max_index]
			else:
				pa = pa_list[max_index]

		if not pa:
			pa = pa_data['BACKTRANSLATION'][i]

		pa_data_list.append(pa)

	pa_data = pd.DataFrame(pa_data_list, columns=['PARAPHRASES'])

	return pa_data


def add_trans(fname, zh_fname, es_fname, pa_fname, ofname):

	fr = open(fname)
	data = json.load(fr)

	usecols=['TRANSLATION']
	zh_data = pd.read_csv(zh_fname, usecols=usecols)

	usecols=['TRANSLATION']
	es_data = pd.read_csv(es_fname, usecols=usecols)

	usecols=['PARAPHRASES', 'COS_SIM_MONO_SP', 'BACKTRANSLATION']
	pa_data = pd.read_csv(pa_fname, usecols=usecols)
	pa_data = best_paraphrases(pa_data)
	

	turn = 0
	outputs = []
	for d in data:
		context = d['context'].strip().replace('\n\n', '\n')

		if 'annotations' not in d or d['response']=='' :
			continue
		context_parts = context.split('\n')
		
		context_zh_output = zh_data['TRANSLATION'][turn]
		context_es_output = es_data['TRANSLATION'][turn]
		context_pa_output = pa_data['PARAPHRASES'][turn]
		turn+=1

		for p in context_parts[1:]:
			context_zh_output += '\n' + zh_data['TRANSLATION'][turn]
			context_es_output += '\n' + es_data['TRANSLATION'][turn]
			context_pa_output += '\n' + pa_data['PARAPHRASES'][turn]
			turn+=1
			
		response_zh_output = zh_data['TRANSLATION'][turn]
		response_es_output = es_data['TRANSLATION'][turn]
		response_pa_output = pa_data['PARAPHRASES'][turn]
		turn+=1

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
	parser.add_argument('--data_name', default=None, type=str, required=True, help="name of input data")
	parser.add_argument('--data_path', default='DSTC_11_Track_4/metadata/dev/en/', type=str, required=False, help="path of input original data")
	parser.add_argument('--zh_data_path', default='DSTC_11_Track_4/task1/dev/en_zh/', type=str, required=False, help="path of Chinese translated data")
	parser.add_argument('--es_data_path', default='DSTC_11_Track_4/task1/dev/en_es/', type=str, required=False, help="path of Spanish translated data")
	parser.add_argument('--pa_data_path', default='DSTC_11_Track_4/task2/dev/', type=str, required=False, help="path of Spanish translated data")

	args=parser.parse_args()
	if not args.data_name:
		  raise Exception("Please provide a valid data_name.")
	fname = os.path.join(args.data_path, f'{args.data_name}/{args.data_name}_eval.json')
	zh_fname = os.path.join(args.zh_data_path, f'{args.data_name}_multilingual_en_zh.csv')
	es_fname = os.path.join(args.es_data_path, f'{args.data_name}_multilingual_en_es.csv')
	pa_fname = os.path.join(args.pa_data_path, f'{args.data_name}_paraphrases.csv')
	ofname = fname.replace('_eval.json', '_eval_zh_es_pa.json')
	add_trans(fname, zh_fname, es_fname, pa_fname, ofname)
	
