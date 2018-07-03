import json, cPickle
goals = cPickle.load(open('user_goals_first_turn_template.part.movie.v1.p'))

slots = []
for i in goals:
	for j in i['inform_slots'].keys():
		slots.append(j)
	for j in i['request_slots'].keys():
		slots.append(j)

print slots