"""
Author: Drew Lustig
Please feel free to contact me for questions about this module (drew.lustig@)
"""


from collections import defaultdict

import numpy as np
import pandas as pd


class Attribution(object):
    """Class used to calculate conversions per channel

    The methodologies currently implemented are first touch,
    last touch, any touch, linear, and markov.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the conversion paths, number of conversions,
        and reach per path
    path_col : str, optional
        Name of column that contains conversion paths. Default is 'path'.
    conv_col : str, optional
        Name of column that contains number of conversions. Default is 'conv'.
    uniques_col : str, optional
        Name of column that contains number of uniques reached.
        Default is 'uniques'.
    path_sep : str, optional
        Delimitor on which to separate path. Default is ' > '.

    Attributes
    ----------
    df
    path_col
    conv_col
    uniques_col
    path_sep
    uniques : int
        Total number of uniques reached
    touchpoints : pandas.DataFrame
        DataFrame containing the unique touchpoints in a given dataset
        and the number of uniques reached by that touchpoint.
        Columns are 'touchpoint' and 'uniques'
    """

    def __init__(
        self,
        df: pd.DataFrame,
        path_col='path',
        conv_col='conv',
        uniques_col='uniques',
        path_sep=' > '
    ):
        columns = df.columns
        for col in [path_col, conv_col, uniques_col]:
            if col not in columns:
                raise ValueError(f"{col} not found in DataFrame.")
        shape = df.shape
        if shape[0] <= 0:
            raise ValueError("DataFrame has no rows.")

        self.df = df
        self.path_col = path_col
        self.conv_col = conv_col
        self.uniques_col = uniques_col
        self.path_sep = path_sep
        self.touchpoints = None

    @property
    def uniques(self):
        """Calculates the total number of uniques reached in the dataset"""

        return self.df[self.uniques_col].sum()

    def _touchpoints(self):
        """Returns the unique touchpoints and their reach as a DataFrame"""

        paths = self.df[self.path_col].str.split(self.path_sep)
        touchpoint_list = [item for path in paths for item in path]
        touchpoints = pd.DataFrame(
            data=pd.Series(touchpoint_list).unique(),
            columns=['touchpoint'])

        def touchpoint_in_path(touchpoint):
            uniques = self.df.loc[
                paths.apply(lambda path: touchpoint in path)
            ][self.uniques_col].sum()
            return uniques

        touchpoints['uniques'] = touchpoints['touchpoint'].apply(
            touchpoint_in_path)
        self.touchpoints = touchpoints

    def _markov_format(self):
        """Returns data in format needed for Markov chain attribution"""

        rows = []
        for row in self.df.itertuples():
            path = getattr(row, self.path_col).split(self.path_sep)
            conv = getattr(row, self.conv_col)
            uniques = getattr(row, self.uniques_col)
            null = uniques - conv
            rows.append(['start', path[0], conv+null])
            for i in range(len(path)):
                if i == len(path) - 1:
                    data = [[path[i], 'conv', conv], [path[i], 'null', null]]
                elif path[i] == path[i+1]:
                    continue
                else:
                    data = [[path[i], path[i+1], uniques]]
                rows.extend(data)

        rows_df = pd.DataFrame(
            data=rows,
            columns=['start_node', 'end_node', 'count']
        ).groupby(['start_node', 'end_node'], as_index=False).sum()
        return rows_df

    def linear(self):
        """Returns conversions by touchpoint with linear methodology"""

        if self.touchpoints is None:
            self._touchpoints()

        results = self.touchpoints.copy()
        df = self.df.copy()
        paths = df[self.path_col].str.split(self.path_sep)
        df['credit'] = 1 / paths.apply(lambda x: len(x))

        def num_conversions(touchpoint):
            path = df[self.path_col].str.split(self.path_sep)
            touches = path.apply(lambda path: len(list(filter(
                lambda item: item == touchpoint, path
            ))))
            conversions = (touches * df['credit'] * df[self.conv_col]).sum()
            return conversions
        results['conversions'] = results['touchpoint'].apply(num_conversions)
        return results

    def last_touch(self):
        """Returns conversions by touchpoint with last touch methodology"""

        if self.touchpoints is None:
            self._touchpoints()

        results = self.touchpoints.copy()
        path = self.df[self.path_col].str.split(self.path_sep)
        results['conversions'] = results['touchpoint'].apply(
            lambda touchpoint: self.df.loc[path.apply(
                lambda path: path[-1] == touchpoint
            )][self.conv_col].sum()
        )
        return results

    def first_touch(self):
        """Returns conversions by touchpoint with first touch methodology"""

        if self.touchpoints is None:
            self._touchpoints()

        results = self.touchpoints.copy()
        path = self.df[self.path_col].str.split(self.path_sep)
        results['conversions'] = results['touchpoint'].apply(
            lambda touchpoint: self.df.loc[path.apply(
                    lambda path: path[0]) == touchpoint
            ][self.conv_col].sum()
        )
        return results

    def any_touch(self):
        """Returns conversions by touchpoint with any touch methodology"""

        if self.touchpoints is None:
            self._touchpoints()

        results = self.touchpoints.copy()
        results['conversions'] = results['touchpoint'].apply(
            lambda touchpoint: (self.df.loc[
                self.df[self.path_col].str.split(self.path_sep).apply(
                    lambda path: touchpoint in path
                    )
                ][self.conv_col].sum()
            )
        )
        return results
    
    
    def assisted_conversion(self):
        results = defaultdict(dict)
        totalConversions = dict()
        df_0_converters_removed = self.df[self.df.conv > 0]
        for index, row in df_0_converters_removed.iterrows():
            path, converters = row['path'], row['conv']
            placements = path.split('>')
            if len(placements) <= 1:
                continue
            placements = [_.strip() for _ in placements]
            final_placement = placements[-1]
            
            if final_placement in totalConversions:
                totalConversions[final_placement]+=converters
            else:
                totalConversions[final_placement] = converters
            
            assisted_placements = set(placements[0:-1])
            for placement in assisted_placements:
                if final_placement in results and placement in results[final_placement]:
                    results[final_placement][placement] += converters
                else:
                    results[final_placement][placement] = converters
        pathReps = []
        for i in list(totalConversions):
            pathReps.append(results[i])
        output = pd.DataFrame(list(totalConversions.items()), columns=['FinalPlacement','TotalConversions'])
        output.insert(2,'AssistedConversions', pathReps)
        return output

    def markov(self):
        """Returns conversions by touchpoint with markov chain methodology"""

        markov_df = self._markov_format()
        markov = Markov(markov_df)
        results = markov.attribute_conversions()
        return results


class Markov(object):
    """Class used to perform Markov chain attribution

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing start node, end node, and number of
        times a transition between those two nodes occurred.
    start_col : str, optional
        Name of column with start nodes. Default is "start_node"
    end_col : str, optional
        Name of column with end nodes. Default is "end_node"
    count_col : str, optional
        Name of column with number of times the transition occurred.
        Default is "count"
    start_value : str, optional
        Value used to represent the initial state. Default is "start"
    dropoff_value : str, optional
        Value used to represent the null (no conversion) state.
        Default is "null"
    conv_value : str, optional
        Value used to represent the conversion state. Default is "conv"

    Attributes
    ----------
    df
    start_col
    end_col
    count_col
    start_value
    dropoff_value
    conv_value
    transition_matrix : pd.DataFrame
    results : pd.DataFrame
    """

    def __init__(
        self,
        df: pd.DataFrame,
        start_col='start_node',
        end_col='end_node',
        count_col='count',
        start_value='start',
        dropoff_value='null',
        conv_value='conv'
    ):
        columns = df.columns
        for col in [start_col, end_col, count_col]:
            if col not in columns:
                raise ValueError(f"{col} not found in DataFrame.")
        if df.loc[df[start_col] == start_value].shape[0] == 0:
            raise ValueError(
                f"Start value {start_value} not found in DataFrame.")
        if df.loc[df[end_col] == dropoff_value].shape[0] == 0:
            raise ValueError(
                f"Dropoff value {dropoff_value} not found in DataFrame.")
        if df.loc[df[end_col] == conv_value].shape[0] == 0:
            raise ValueError(
                f"Conversion value {conv_value} not found in DataFrame.")

        self.start_col = start_col
        self.end_col = end_col
        self.count_col = count_col
        self.start_value = start_value
        self.dropoff_value = dropoff_value
        self.conv_value = conv_value
        self.df = df
        self.transition_matrix = None
        self.results = None

    @property
    def conversions(self):
        conversions = self.df.loc[
            self.df[self.end_col] == self.conv_value][self.count_col].sum()
        return conversions

    def calculate_probabilities(self):
        """Calculates the probability of going from one state to another"""

        df = self.df[[self.start_col, self.end_col, self.count_col]].groupby(
            [self.start_col, self.end_col], as_index=False).sum()
        totals = df.groupby(
            self.start_col, as_index=False
        ).sum()[[self.start_col, self.count_col]].rename(
            columns={self.start_col: 'state', self.count_col: 'total'})
        df = df.merge(
            totals, left_on=self.start_col, right_on='state', how='left')
        df['probability'] = df[self.count_col] / df['total']
        return df[['start_node', 'end_node', 'count', 'probability']]

    def _append_end_rows(self, df=None):
        """Appends final state transition rows."""

        if df is None:
            df = self.df

        data = [
            [self.conv_value, self.conv_value, 0, 1.0],
            [self.dropoff_value, self.dropoff_value, 0, 1.0]]
        conv_drop = (df[self.start_col] == self.conv_value) & (df[self.end_col] == self.conv_value)
        dropoff_drop = (df[self.start_col] == self.dropoff_value) & (df[self.end_col] == self.dropoff_value)
        df = df.loc[~(conv_drop | dropoff_drop)]

        row_appends = pd.DataFrame(
            data=data,
            columns=[self.start_col, self.end_col, self.count_col, 'probability']
            )
        new_df = df.append(row_appends, ignore_index=True)
        return new_df

    def create_matrix(self, df=None):
        """Creates the transition matrix."""

        if df is None:
            df = self.df

        transition_matrix = df[[self.start_col, self.end_col, 'probability']].pivot(
            index=self.start_col,
            columns=self.end_col,
            values='probability'
        ).fillna(0)
        transition_matrix[self.start_value] = 0.0
        sum_not_one = transition_matrix.sum(axis=1) != 1.0
        transition_matrix.loc[sum_not_one, self.dropoff_value] = transition_matrix[self.dropoff_value] + 1.0 - transition_matrix.sum(axis=1)
        self.transition_matrix = transition_matrix
        return transition_matrix

    def calculate_removals(self):
        """Calculates the removal effect of each state in the transition matrix."""

        if self.transition_matrix is None:
            raise TypeError("No transition matrix found.")

        transition_matrix = self.transition_matrix
        columns = transition_matrix.columns

        # Calculate conversion rate with all channels included
        base_cvr = transition_matrix.copy()
        i = 0
        while i < len(columns):
            base_cvr = base_cvr.dot(base_cvr)
            i += 1
        base_cvr = base_cvr.loc[self.start_value][self.conv_value]

        columns = transition_matrix.drop(
            [self.start_value, self.dropoff_value, self.conv_value],
            axis=1).columns  # Channels to calculate removal effects for

        removal_effects = pd.DataFrame(columns=['channel', 'effect'])
        for col in columns:
            S = pd.DataFrame(
                data=transition_matrix.loc[self.start_value]
            ).transpose().drop(col, axis=1)

            Q = transition_matrix.drop([col], axis=0).drop(col, axis=1)
            Q.loc[
                Q.sum(axis=1) != 1.0,
                self.dropoff_value
            ] += (1.0 - Q.sum(axis=1))

            for i in range(len(Q.columns)):
                Q = Q.dot(Q)
            removal_cvr = S.dot(Q).loc[self.start_value][self.conv_value]
            removal_effect = 1 - (removal_cvr / base_cvr)
            removal_effects = removal_effects.append(
                pd.DataFrame(
                    data=[[col, removal_effect]],
                    columns=['channel', 'effect']
                ),
                ignore_index=True
            )

        return removal_effects

    def attribute_conversions(self):
        """Returns attributed markov conversions by channel."""

        df = self.calculate_probabilities()
        df = self._append_end_rows(df=df)

        self.create_matrix(df=df)
        removal_effects = self.calculate_removals()
        total_effect = removal_effects.effect.sum()
        removal_effects['percent_effect'] = removal_effects['effect'] / total_effect
        removal_effects['attrib_model_convs'] = removal_effects['percent_effect'] * self.conversions
        self.results = removal_effects[['channel', 'attrib_model_convs']]
        return self.results
