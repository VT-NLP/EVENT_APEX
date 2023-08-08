# coding=utf-8
import json
import os


class Metadata(object):
    def __init__(self, metadata_path=os.getenv('PROJECT_ROOT')+'/utils/metadata.json'):

        with open(metadata_path, 'r') as j:
            meta = json.loads(j.read())
        self.ace = DatasetFactContainer(meta['ace'])
        self.ere = DatasetFactContainer(meta['ere'])
        self.maven = DatasetFactContainer(meta['maven'])

        self.ace.novel_types = [x for x in self.ace.trigger_set if
                                x.split(':')[0].lower() in {'life', 'personnel', 'transaction'} and x not in {
                                    'Justice:Acquit', 'Justice:Pardon', 'Justice:Extradite', 'Personnel:Nominate'}]
        self.ace.novel_ids = sorted([self.ace.triggers_to_ids[x] for x in self.ace.novel_types])
        self.ace.base_types = [x for x in self.ace.trigger_set if
                               x.split(':')[0].lower() not in {'life', 'personnel', 'transaction'} and x not in {
                                   'Justice:Acquit', 'Justice:Pardon', 'Justice:Extradite', 'Personnel:Nominate'}]
        self.ace.base_ids = sorted([self.ace.triggers_to_ids[x] for x in self.ace.base_types])

        ere_filtered_class = {'business:declarebankruptcy', 'personnel:nominate', 'transaction:transaction'}
        self.ere.novel_types = [x for x in self.ere.trigger_set if
                                x.split(':')[0].lower() in {'life', 'personnel',
                                                            'transaction'} and x not in ere_filtered_class]
        self.ere.base_types = [x for x in self.ere.trigger_set if
                               x.split(':')[0].lower() not in {'life', 'personnel',
                                                               'transaction'} and x not in ere_filtered_class]
        self.ere.novel_ids = sorted([self.ere.triggers_to_ids[x] for x in self.ere.novel_types])
        self.ere.base_ids = sorted([self.ere.triggers_to_ids[x] for x in self.ere.base_types])

        self.maven.novel_types = ['Adducing', 'Award', 'Bearing_arms', 'Carry_goods', 'Commerce_buy', 'Commerce_pay',
                                  'Containing', 'Cost', 'Create_artwork', 'Cure', 'Exchange', 'Extradition', 'Filling',
                                  'Forming_relationships', 'Having_or_lacking_access', 'Hiding_objects',
                                  'Imposing_obligation', 'Incident', 'Ingestion', 'Institutionalization', 'Justifying',
                                  'Kidnapping', 'Labeling', 'Lighting', 'Limiting', 'Patrolling', 'Practice', 'Prison',
                                  'Ratification', 'Renting', 'Research', 'Reveal_secret', 'Revenge',
                                  'Rewards_and_punishments', 'Risk', 'Rite', 'Robbery', 'Scouring',
                                  'Submitting_documents', 'Surrounding', 'Suspicion', 'Theft', 'Vocalizations',
                                  'Wearing', 'Writing']
        self.maven.base_types = ['Achieve', 'Action', 'Agree_or_refuse_to_act', 'Aiming', 'Arranging', 'Arrest',
                                 'Arriving', 'Assistance', 'Attack', 'Becoming', 'Becoming_a_member',
                                 'Being_in_operation', 'Besieging', 'Bodily_harm', 'Body_movement', 'Bringing',
                                 'Building', 'Catastrophe', 'Causation', 'Cause_change_of_position_on_a_scale',
                                 'Cause_change_of_strength', 'Cause_to_amalgamate', 'Cause_to_be_included',
                                 'Cause_to_make_progress', 'Change', 'Change_event_time', 'Change_of_leadership',
                                 'Change_sentiment', 'Check', 'Choosing', 'Collaboration', 'Come_together',
                                 'Coming_to_be', 'Coming_to_believe', 'Commerce_sell', 'Commitment', 'Committing_crime',
                                 'Communication', 'Competition', 'Confronting_problem', 'Connect', 'Conquering',
                                 'Control', 'Convincing', 'Creating', 'Criminal_investigation', 'Damaging', 'Death',
                                 'Deciding', 'Defending', 'Departing', 'Destroying', 'Dispersal', 'Earnings_and_losses',
                                 'Education_teaching', 'Employment', 'Emptying', 'Escaping', 'Expansion',
                                 'Expend_resource', 'Expressing_publicly', 'GetReady', 'Getting', 'GiveUp', 'Giving',
                                 'Hindering', 'Hold', 'Hostile_encounter', 'Influence', 'Judgment_communication',
                                 'Killing', 'Know', 'Legal_rulings', 'Legality', 'Manufacturing', 'Military_operation',
                                 'Motion', 'Motion_directional', 'Name_conferral', 'Openness', 'Participation',
                                 'Perception_active', 'Placing', 'Presence', 'Preserving', 'Preventing_or_letting',
                                 'Process_end', 'Process_start', 'Protest', 'Publishing', 'Quarreling', 'Receiving',
                                 'Recording', 'Recovering', 'Reforming_a_system', 'Releasing', 'Removing', 'Reporting',
                                 'Request', 'Rescuing', 'Resolve_problem', 'Response', 'Scrutiny', 'Self_motion',
                                 'Sending', 'Sign_agreement', 'Social_event', 'Statement', 'Supply', 'Supporting',
                                 'Surrendering', 'Telling', 'Temporary_stay', 'Terrorism', 'Testing', 'Traveling',
                                 'Use_firearm', 'Using', 'Violence', 'Warning']
        self.maven.novel_ids = sorted([self.maven.triggers_to_ids[x] for x in self.maven.novel_types])
        self.maven.base_ids = sorted([self.maven.triggers_to_ids[x] for x in self.maven.base_types])

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


class DatasetFactContainer(object):
    def __init__(self, trigger_arg_dic, arg_entity_dic=None):
        self.trigger_arg_dic = trigger_arg_dic
        self.arg_entity_dic = arg_entity_dic
        self.trigger_set = {}
        self.arg_set = {}
        self.entity_set = {}
        self.args_to_ids, self.ids_to_args = {}, {}
        self.triggers_to_ids, self.ids_to_triggers = {}, {}
        self.pos_set = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                        'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        self.pos2id = dict((v, i) for v, i in zip(sorted(self.pos_set), range(len(self.pos_set))))
        self.entity_to_ids = {'FAC': 0, 'GPE': 1, 'LOC': 2, 'ORG': 3, 'PER': 4, 'VEH': 5, 'WEA': 6, 'O': 7, '[PAD]': 8}
        self.arg_count = None
        self.event_count = None
        self.entity_count = None
        self.pos_count = None
        self.init_setup()


    def init_setup(self):
        self.trigger_set = set(self.trigger_arg_dic.keys())
        self.triggers_to_ids, self.ids_to_triggers = self.set_to_ids(self.trigger_set)

        self.arg_set = set(sum(list(self.trigger_arg_dic.values()), []))
        self.args_to_ids, self.ids_to_args = self.set_to_ids(self.arg_set)
        self.arg_count = len(self.arg_set)
        self.event_count = len(self.trigger_set)
        self.entity_count = len(self.entity_to_ids)-1
        self.pos_count = len(self.pos_set)

    @staticmethod
    def set_to_ids(input_set):
        items_to_ids = dict((v, i) for v, i in zip(sorted(list(input_set)), range(len(input_set))))
        items_to_ids['O'] = len(items_to_ids)
        items_to_ids['[PAD]'] = len(items_to_ids)
        ids_to_items = dict((v, k) for k, v in items_to_ids.items())
        return items_to_ids, ids_to_items



