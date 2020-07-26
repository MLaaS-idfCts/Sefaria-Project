class MultiStageClassifier

    def __init__(self, raw_df, supertopics):

        self.raw_df = raw_df
        self.supertopics = supertopics

    def super_classify():
        
        df = all data with topics and exp topics

        supertopics = [top 4 or 5]

    limited_df = without extraneous super topics

    # stage 1 result
    pred_super_df  = df with pred supertopics

    limiter = Limiter()

    # now stage 2
    for supertopic in supertopics:

        # eliminate all topics that are not under this supertopic
        df = limiter.limit_subtopics(df)

        train

        predict

        


    result = df with pred topics

    return result