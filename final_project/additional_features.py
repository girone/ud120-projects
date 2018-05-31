from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np


def get_without_NaN(data, person, key):
    return float(data[person][key]) if data[person].has_key(key) else 0.


class NewFeature:
    __metaclass__ = ABCMeta

    @abstractmethod
    def extend(self, data_set):
        """Extends a data set with new features.

        Returns the data_set.
        """

    @abstractproperty
    def new_feature_names(self):
        """Returns a list of the names this class adds."""


class EmailShares(NewFeature):
    FEATURE_1 = "emails_to_poi_share"
    FEATURE_2 = "emails_from_poi_share"

    def extend(self, data_dict):
        """Extends the data set `data_dict` with the share of emails from/to POIs."""

        for person, data in data_dict.iteritems():
            to_poi = get_without_NaN(data_dict, person,
                                     "from_this_person_to_poi")
            from_total = get_without_NaN(data_dict, person, "from_messages")
            data_dict[person][
                self.
                FEATURE_1] = to_poi / from_total if from_total and not np.isnan(
                    from_total) else 0.
            from_poi = get_without_NaN(data_dict, person,
                                       "from_poi_to_this_person")
            to_total = get_without_NaN(data_dict, person, "to_messages")
            data_dict[person][
                self.
                FEATURE_2] = from_poi / to_total if to_total and not np.isnan(
                    to_total) else 0.
        return data_dict

    def new_feature_names(self):
        return [self.FEATURE_1, self.FEATURE_2]


class PaymentsStockRatio(NewFeature):
    FEATURE_PAYMENT_STOCK_RATIO = "total_payments_to_stock_value_ratio"

    def extend(self, data_dict):
        """Extends the data_dict with the share of 'total_payments' to 'total_stock_value'."""
        for person, data in data_dict.iteritems():
            payments = get_without_NaN(data_dict, person, "total_payments")
            stock_value = get_without_NaN(data_dict, person,
                                          "total_stock_value")
            data_dict[person][self.FEATURE_PAYMENT_STOCK_RATIO] = (
                payments + 1) / (stock_value + 1)
        return data_dict

    def new_feature_names(self):
        return [self.FEATURE_PAYMENT_STOCK_RATIO]