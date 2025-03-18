import numpy

class DataPreparing:
    @staticmethod
    def normalize(feature):
        new_feature = []

        for item in feature:
            new_feature.append(item / numpy.max(feature))

        return numpy.array(new_feature)

    @staticmethod
    def z_score(feature):
        return (feature - numpy.mean(feature)) / numpy.std(feature)

    @staticmethod
    def z_score_predict(values, feature):
        return (values - numpy.mean(feature)) / numpy.std(feature)