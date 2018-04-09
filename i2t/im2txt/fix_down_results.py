import json

lines = open('down_model/word_counts_org.txt','r').read().split('\n')
# print (eval(lines[0].split()[0]))
lines = [eval(line.split()[0]).decode('utf-8') + ' ' + line.split()[1] for line in lines]
open('down_model/word_counts.txt','w').write('\n'.join(lines))

# r = json.load(open('../coco-caption/results/captions_val2014_down_results.json', 'r'))
# # print (r)
# for _ in range(len(r)):
#   c = r[_]['caption']
#   p = c.find(' <S>')
#   if p != -1: r[_]['caption'] = c[:p]
# with open('../coco-caption/results/captions_val2014_down-fix_results.json', 'w') as f:
#   dataset = [ '{\"image_id\": %s, \"caption\": \"%s\"},' % ( _['image_id'], _['caption'][:-1] )
#               for _ in r ]
#   f.write('[' + '\n'.join(dataset)[:-1] + ']')
# # json.dump(r, )
