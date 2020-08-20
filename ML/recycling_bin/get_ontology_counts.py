        if get_ontology_counts:

            # capture list of most commonly occurring topics
            ontology_counts_dict = data.get_ontology_counts_dict()

            # store result
            with open(f'data/ontology_counts_dict_row_lim_{row_lim}.pickle', 'wb') as handle:
                pickle.dump(ontology_counts_dict, handle, protocol=3)

