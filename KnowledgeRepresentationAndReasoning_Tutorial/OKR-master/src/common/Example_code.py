import okr
import numpy as np

graph1=okr.load_graph_from_file("../../data/baseline/test/car_bomb.xml")

'''
in current example code, each graph1 object contains the following attributes:
name: car_bomb.xml
sentences: 37 sentences in total, all about bomb.
ignored indices: some words in the 37 sentences are ignored. But it is not clear why we need to do this.
tweet_ids: 37 elements in total, but all of them have a value of "None"
entities: 53 entities in total
propositions: 47 propositions in total
'''

# print('name:',graph1.name)  # XML file name
# print('sentences:',graph1.sentences)  # Dictionary of sentence ID (starts from 1) to tokenized sentence
# print('ignored indices:',graph1.ignored_indices)  # set of words to ignore, in format sentence_id[index_id]
# print('tweet_ids:',graph1.tweet_ids)  # Dictionary of sentence ID to tweet ID
# print('entities:',graph1.entities)  # Dictionary of entity ID to Entity object, entity length: 37
# print('propositions:',graph1.propositions)  # Dictionary of proposition id to Proposition object, proposition length: 36
print('proposition keys:', graph1.propositions.keys())
print('entity keys:', graph1.entities.keys())

INDEX_P = 8
INDEX_E = 4

proposition_values = graph1.propositions.values()
entity_values = graph1.entities.values()

# for i in np.arange(len(proposition_values)):
#     print(proposition_values[i])
print('---------------------------------')
print('proposition id:',proposition_values[INDEX_P].id)
print('proposition name:',proposition_values[INDEX_P].name)
print('proposition mentions keys:',proposition_values[INDEX_P].mentions.keys())
proposition_mention_values = proposition_values[INDEX_P].mentions.values()
for j in np.arange(len(proposition_mention_values)):
    print('     proposition mention parents:',proposition_mention_values[j].parent)
print('proposition terms:',proposition_values[INDEX_P].terms)
print('proposition entailment graph:',proposition_values[INDEX_P].entailment_graph)
print('proposition atributor:',proposition_values[INDEX_P].attributor)
print('---------------------------------')
print('entity id:', entity_values[INDEX_E].id)
print('entity name:',entity_values[INDEX_E].name)
print('entity mentions keys:',entity_values[INDEX_E].mentions.keys())
entity_mention_values = entity_values[INDEX_P].mentions.values()
for k in np.arange(len(entity_mention_values)):
    print('     entity mention parents:',entity_mention_values[k].parent)
print('entity terms:',entity_values[INDEX_E].terms)
print('entity entailment graph:',entity_values[INDEX_E].entailment_graph)
print('---------------------------------')

# for id,proposition in graph1.propositions.iteritems():
#      print("proposition: "+str(id))
#      for m_id,mention in proposition.mentions.iteritems():
#            print ("\tmention: "+str(m_id))
#            print "\ttemplate: "+mention.template
#            print "\targuments:"
#            for a_id,arg in mention.argument_mentions.iteritems():
#                                    print "\t\tA"+a_id+": "+arg.desc
# 				   mention_type= "Entity: "if arg.mention_type==0 else "Proposition: "
# 				   print "\t\t\t"+mention_type+str(arg.parent_id)+" mention: "+str(arg.parent_mention_id)
# 				   print "\t\t\t"+arg.parent_name
