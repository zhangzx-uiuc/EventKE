import json
import math

from nltk import sent_tokenize

def sent_tokenize_file(input_str):
    # input: text string
    # output: sents: list: ["...", "..."]
    # output: start_idxs: list: [0, 222, 345]
    sents = sent_tokenize(input_str)
    start_idxs = []
    for sent in sents:
        start_idxs.append(input_str.find(sent))

    return sents, start_idxs

def read_graph(graph):
    # an input graph is a dictionary
    schemas = graph["schemas"][0]

    events = schemas["steps"]
    entity_rels = schemas["entityRelations"]
    entities = schemas["entities"]

    offsets = schemas["provenanceData"]

    # step #1: first to read provenance data into a dictionary
    offsets_dict = {} # {prove_id: [doc_id, start, end]}
    for offset in offsets:
        offsets_dict.update({offset["provenance"]: [offset["childID"], offset["offset"], offset["offset"]+offset["length"]]})

    # print(offsets_dict)
    # print(len(offsets_dict))


    # step #2: read entities and relations
    # first read entities with their types
    entity_id_to_type = {} # {entity_id: [subtype, type]}
    for entity in entities:
        entity_id_to_type.update({entity["@id"]:[entity["name"], entity["entityTypes"]]})
    
    # second, read entity provenance data from relations
    relations = {} # {relation_id: {"type":xx, "start", "end"}}
    entities = {} # same format as ace one {"type":xx, "subtype":xx, "mentions": [prov1, prov2]}
    for rel in entity_rels:
        rel_type = rel["relations"]["relationPredicate"]

        subj_id = rel["relationSubject"]
        obj_id = rel["relations"]["relationObject"]

        subj_offsets = rel["provenance"]
        obj_offsets = rel["relations"]["provenance"]

        # update realtions
        if rel["relations"]["@id"] not in relations:
            relations.update({rel["relations"]["@id"]: {"id": rel["relations"]["@id"], "type": rel_type, "subtype": rel_type, "start_entity": subj_id, "end_entity": obj_id, "mentions": []}})
        
        # update entities
        if subj_id not in entities:
            entities.update({subj_id: {"type": entity_id_to_type[subj_id][1], "subtype": entity_id_to_type[subj_id][0], "mentions": [offsets_dict[ids] for ids in subj_offsets]}})
        
        if obj_id not in entities:
            entities.update({obj_id: {"type": entity_id_to_type[obj_id][1], "subtype": entity_id_to_type[obj_id][0], "mentions": [offsets_dict[ids] for ids in obj_offsets]}})
    
    # print(relations['caci:Schemas/Instantiated/cluster_0/Relations/Relation_87'])
    # print(len(relations))
    # print(entities)
    # print(len(entities))

    # read events
    events_dict = {} # {event_id: {"type":xx, "mentions": [[doc_id, start, end],...], "args": [{"entity_id": xxx, "arg_type": yyy}]}
    for event in events:
        event_id = event["@id"]
        event_type = event["@type"]
        event_prov = [offsets_dict[ids] for ids in event["provenance"]]

        args = []

        roles = event["participants"]
        
        for role in roles:
            if len(role["values"]) > 1:
                print(1234)
            else:
                pass
            new_role = {"entity_id": role["values"][0]["entity"], "arg_type": role["role"]}
            args.append(new_role)
        
        events_dict.update({event_id: {"type": event_type, "mentions": event_prov, "event_args": args}})

    # print(events_dict['caci:Schemas/Instantiated/cluster_0/Steps/EN_Event_0027731'])
    # print(len(events_dict))
    # sumamrize list

    doc_list = []
    for key in offsets_dict:
        if offsets_dict[key][0] not in doc_list:
            doc_list.append(offsets_dict[key][0])
    
    # print(doc_list)

    return entities, relations, events_dict, doc_list

def span_to_mention(doc_dict, span):
    # input: [doc_id, start_idx, end_idx]
    sents = doc_dict[span[0]]["sentences"]
    idxs = doc_dict[span[0]]["idxs"]

    start = span[1]
    end = span[2]

    sent_num = len(sents)
    new_idxs = idxs.copy()
    new_idxs.append(math.inf)

    found = False

    for i in range(sent_num):
        sent_range = [new_idxs[i], new_idxs[i+1]]
        if start >= sent_range[0] and end <= sent_range[1]:
            selected_sent = sents[i]
            new_start = start - idxs[i]
            new_end = end - idxs[i]
            found = True
            break
    
    if found:
        return selected_sent, new_start, new_end, found
    else:
        return "", 0, 0, found

def select_sentences(entities, relations, events, doc_list, base_dir):
    # first we read these docs into dict
    doc_dict = {}
    # {doc_id: {"sentences": [list of sents], "idxs": [list of starting idxs for these sentences]}}

    for doc_id in doc_list:
        # print(doc_id)
        with open(base_dir + doc_id + ".rsd.txt", "r", encoding="utf-8") as f:
            doc_string = f.read()
        
        sents, start_idxs = sent_tokenize_file(doc_string)
        doc_dict.update({doc_id: {"sentences": sents, "idxs": start_idxs}})

    # map all [doc_id, starts, ends] into sent, start, end
    # print(entities['caci:Schemas/Instantiated/cluster_0/Entities/EN_Entity_EDL_ENG_0004255'])
    # print(events['caci:Schemas/Instantiated/cluster_0/Steps/EN_Event_0028899'])
    # print(relations['caci:Schemas/Instantiated/cluster_0/Relations/Relation_90'])

    new_entities, new_relations, new_events = {}, {}, {}

    for key in entities:
        mentions = entities[key]["mentions"]
        entity_id = key
        new_mentions = []

        for i,mention in enumerate(mentions):
            sent, s, e, found = span_to_mention(doc_dict, mention)
            if found:
                new_mention = {"mention_id": key+"_"+str(i), "mention_type": "NAM", "full_span": [s, e], "head_span": [s, e], "full_text": sent[s:e], "head_text": sent[s:e], "sent": sent}
                new_mentions.append(new_mention)
        
        if new_mentions != []:
            new_entities.update({key: {"type": entities[key]["type"], "subtype": entities[key]["subtype"], "mentions": new_mentions, "id": key}})
    
    for key in events:
        mentions = events[key]["mentions"]
        event_id = key
        new_mentions = []

        for i,mention in enumerate(mentions):
            sent, s, e, found = span_to_mention(doc_dict, mention)
            if found:
                new_mention = {"event_mention_id": key+"_"+str(i), "trigger": [s, e], "trigger_text": sent[s:e], "args":[], "sent": sent}
                new_mentions.append(new_mention)

            
        if new_mentions != []:
            new_events.update({key: {"id": key, "type": events[key]["type"], "event_args": events[key]["event_args"], "mentions": new_mentions}})
    
    return new_entities, relations, new_events

def read_whole_dataset(base_dir, data_json_dir, entity_dir, relation_dir, event_dir):
    # output: list of entities, relations and events
    # another function: filter out the useless entities
    entity_list, relation_list, event_list = [], [], []

    with open(data_json_dir, "r", encoding="utf-8") as f:
        done = 0
        num = 0
        while not done:
            line = f.readline()
            print(num)
            if line != "":
                data_dict = json.loads(line)
                init_ents, init_rels, init_evts, doc_list = read_graph(data_dict)
                final_ents, final_rels, final_evts = select_sentences(init_ents, init_rels, init_evts, doc_list, base_dir)
                entity_list.append(final_ents)
                relation_list.append(final_rels)
                event_list.append(final_evts)
            else:
                done = 1
            num += 1
    
    total_ent_dict, total_rel_dict, total_evt_dict = {}, {}, {}
    
    for entities in entity_list:
        for key in entities:
            total_ent_dict.update({key: entities[key]})
    
    for relations in relation_list:
        for key in relations:
            total_rel_dict.update({key: relations[key]})
    
    for events in event_list:
        for key in events:
            total_evt_dict.update({key: events[key]})
    
    # first check relations
    rel_involved_ents = {}

    with open(relation_dir, "w", encoding="utf-8") as f1:
        for rel_id in total_rel_dict:
            start = total_rel_dict[rel_id]["start_entity"]
            end = total_rel_dict[rel_id]["end_entity"]

            if (start in total_ent_dict) and (end in total_ent_dict):
                rel_involved_ents.update({start: 1})
                rel_involved_ents.update({end: 1})

                f1.write(json.dumps(total_rel_dict[rel_id])+'\n')
    print(len(rel_involved_ents))
    # then check entities:
    with open(entity_dir, "w", encoding="utf-8") as f2:
        for ent_id in total_ent_dict:
            f2.write(json.dumps(total_ent_dict[ent_id])+'\n')
    
    # finally check events
    with open(event_dir, "w", encoding="utf-8") as f3:
        for evt_id in total_evt_dict:
            f3.write(json.dumps(total_evt_dict[evt_id])+'\n')

    
if __name__ == "__main__":
    with open("data.json", "r", encoding="utf-8") as f:
        data_dict = json.loads(f.readline())
    
    base_dir = "/shared/nas/data/m1/zixuan11/kairos_all_files/"
    data_dir = "/shared/nas/data/m1/zixuan11/data.json"

    entity_dir = "/shared/nas/data/m1/zixuan11/kairos_kg/entities.json"
    relation_dir = "/shared/nas/data/m1/zixuan11/kairos_kg/relations.json"
    event_dir = "/shared/nas/data/m1/zixuan11/kairos_kg/events.json"

    read_whole_dataset(base_dir, data_dir, entity_dir, relation_dir, event_dir)
    # ents, rels, evts, docs = read_graph(data_dict)
    # # print(1)
    # entities, relations, events = select_sentences(ents, rels, evts, docs, base_dir)
    # print(entities)
    # print(relations)
    # print(events)
        