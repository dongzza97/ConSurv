import numpy as np
import warnings
import pdb
import matplotlib.pyplot as plt
import scipy as sp
import bisect
import scipy.integrate as integrate

from scipy.stats import chi2, chisquare
from dataclasses import InitVar, dataclass, field
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.integrate import trapezoid
from typing import Union, Optional, Callable
from collections import Counter


class BS:
    def __init__(
            self,
            pred,
            time_coordinates, 
            train_event_times,
            train_event_indicators,
            event_times,
            event_indicators,
            interpolation="Linear"):


        self.pred = pred
        self.time_coordinates = time_coordinates
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators
        self.event_times = event_times
        self.event_indicators = event_indicators
        self.interpolation = interpolation


    def interpolated_survival_curve(self, times_coordinate, survival_curve, interpolation):
        if interpolation == "Linear":
            spline = interp1d(times_coordinate, survival_curve, kind='linear', fill_value='extrapolate')
        elif interpolation == "Pchip":
            spline = PchipInterpolator(times_coordinate, survival_curve)
        else:
            raise ValueError("interpolation must be one of ['Linear', 'Pchip']")
        return spline

    def predict_multi_probs_from_curve(self, survival_curve,times_coordinate, target_times, interpolation="Linear"):

        target_times = target_times.astype(float).tolist()

        spline = self.interpolated_survival_curve(times_coordinate, survival_curve, interpolation)
        # predicting boundary
        max_time = float(max(times_coordinate))
        slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)
        predict_probabilities = np.array(spline(target_times))
        for i, target_time in enumerate(target_times):
            if target_time > max_time:
                predict_probabilities[i] = max(slope * target_time + 1, 0)

        return predict_probabilities


    def predict_multi_probabilities_from_curve(
            self,
            target_times: np.ndarray
    ) -> np.ndarray:

        predict_probs_mat = []
        for i in range(self.pred.shape[0]):
            predict_probs = self.predict_multi_probs_from_curve(self.pred[i, :], self.time_coordinates,
                                                            target_times, self.interpolation).tolist()
            predict_probs_mat.append(predict_probs)
        predict_probs_mat = np.array(predict_probs_mat)
        return predict_probs_mat


    def brier_multiple_points(
            self, 
            predict_probs_mat: np.ndarray,
            event_times: np.ndarray,
            event_indicators: np.ndarray,
            train_event_times: np.ndarray,
            train_event_indicators: np.ndarray,
            target_times: np.ndarray,
            ipcw:bool=True, 
    ) -> np.ndarray:

        if target_times.ndim != 1:
            error = "'time_grids' is not a one-dimensional array."
            raise TypeError(error)

        target_times_mat = np.repeat(target_times.reshape(1, -1), repeats=len(event_times), axis=0)
        event_times_mat = np.repeat(event_times.reshape(-1, 1), repeats=len(target_times), axis=1)
        event_indicators_mat = np.repeat(event_indicators.reshape(-1, 1), repeats=len(target_times), axis=1)
        event_indicators_mat = event_indicators_mat.astype(bool)
        
        if ipcw:
            inverse_train_event_indicators = 1 - train_event_indicators
            train_event_times, inverse_train_event_indicators = train_event_times.squeeze(1), inverse_train_event_indicators.squeeze(1)
            ipc_model = KaplanMeier(train_event_times, inverse_train_event_indicators)
            ipc_pred = ipc_model.predict(event_times_mat)
            ipc_pred[ipc_pred == 0] = np.inf
            weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat) / ipc_pred
            weight_cat1[np.isnan(weight_cat1)] = 0
            ipc_target_pred = ipc_model.predict(target_times_mat)
            ipc_target_pred[ipc_target_pred == 0] = np.inf
            weight_cat2 = (event_times_mat > target_times_mat) / ipc_target_pred
            weight_cat2[np.isnan(weight_cat2)] = 0
        
        else: 
            weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat)
            weight_cat2 = (event_times_mat > target_times_mat)

        ipcw_square_error_mat = np.square(predict_probs_mat) * weight_cat1 + np.square(1 - predict_probs_mat) * weight_cat2
        brier_scores = np.mean(ipcw_square_error_mat, axis=0)
        return brier_scores


    def brier_score_multiple_points(
            self, 
            target_times:np.ndarray, 
            IPCW_weighted:bool=True):

        predict_probs_mat = self.predict_multi_probabilities_from_curve(target_times)
        return self.brier_multiple_points(predict_probs_mat, self.event_times, self.event_indicators, self.train_event_times, self.train_event_indicators, target_times, IPCW_weighted)

    def integrated_brier_score(
            self,
            num_points: int=None,
            IPCW_weighted: bool=True,
            draw_figure: bool=False):
        
        max_target_time = np.amax(np.concatenate((self.event_times, self.train_event_times)))
        
        if num_points is None:

            censored_times = self.event_times[self.event_indicators == 0]
            time_points = np.unique(censored_times)

            if time_points.size == 0:
                raise ValueError("You don't have censor data in the testset, "
                                 "please provide \"num_points\" for calculating IBS")
            else:
                time_range = np.max(time_points) - np.min(time_points)
        else:
            time_points = np.linspace(0, max_target_time, num_points)
            time_range = max_target_time

        b_scores = self.brier_score_multiple_points(time_points, IPCW_weighted)
        if np.isnan(b_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")
            bs_dict = {}
            for time_point, b_score in zip(time_points, b_scores):
                bs_dict[time_point] = b_score
            print("Brier scores for multiple time points are".format(bs_dict))
        integral_value = trapezoid(b_scores, time_points)
        ibs_score = integral_value / time_range
        
        if draw_figure:
            plt.plot(time_points, b_scores, 'bo-')
            score_text = r'IBS$= {:.3f}$'.format(ibs_score)
            plt.plot([], [], ' ', label=score_text)
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Integrated Brier Score')
            plt.show()

        return ibs_score
        
@dataclass
class KaplanMeier:
    
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    survival_times: np.array = field(init=False)
    population_count: np.array = field(init=False)
    events: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)
    cumulative_dens: np.array = field(init=False)
    probability_dens: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        self.population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        self.events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        event_ratios = 1 - self.events / self.population_count
        self.survival_probabilities = np.cumprod(event_ratios)
        self.cumulative_dens = 1 - self.survival_probabilities
        self.probability_dens = np.diff(np.append(self.cumulative_dens, 1))

    def predict(self, prediction_times: np.array):
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities

@dataclass
class KaplanMeierArea(KaplanMeier):
    area_times: np.array = field(init=False)
    area_probabilities: np.array = field(init=False)
    area: np.array = field(init=False)
    km_linear_zero: float = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        super().__post_init__(event_times, event_indicators)
        area_probabilities = np.append(1, self.survival_probabilities)
        area_times = np.append(0, self.survival_times)
        self.km_linear_zero = -1 / ((area_probabilities[-1] - 1) / area_times[-1])
        if self.survival_probabilities[-1] != 0:
            area_times = np.append(area_times, self.km_linear_zero)
            area_probabilities = np.append(area_probabilities, 0)

        # we are facing the choice of using the trapzoidal rule or directly using the area under the step function
        # we choose to use trapz because it is more accurate
        area_diff = np.diff(area_times, 1)
        average_probabilities = (area_probabilities[0:-1] + area_probabilities[1:]) / 2
        area = np.flip(np.flip(area_diff * average_probabilities).cumsum())
        # area = np.flip(np.flip(area_diff * area_probabilities[0:-1]).cumsum())

        self.area_times = np.append(area_times, np.inf)
        self.area_probabilities = area_probabilities
        self.area = np.append(area, 0)

    @property
    def mean(self):
        return self.best_guess(np.array([0])).item()

    def best_guess(self, censor_times: np.array):
        # calculate the slope using the [0, 1] - [max_time, S(t|x)]
        slope = (1 - min(self.survival_probabilities)) / (0 - max(self.survival_times))
        # if after the last time point, then the best guess is the linear function
        before_last_idx = censor_times <= max(self.survival_times)
        after_last_idx = censor_times > max(self.survival_times)
        surv_prob = np.empty_like(censor_times).astype(float)
        surv_prob[after_last_idx] = 1 + censor_times[after_last_idx] * slope
        surv_prob[before_last_idx] = self.predict(censor_times[before_last_idx])
        # do not use np.clip(a_min=0) here because we will use surv_prob as the denominator,
        # if surv_prob is below 0 (or 1e-10 after clip), the nominator will be 0 anyway.
        surv_prob = np.clip(surv_prob, a_min=1e-10, a_max=None)

        censor_indexes = np.digitize(censor_times, self.area_times)
        censor_indexes = np.where(
            censor_indexes == self.area_times.size + 1,
            censor_indexes - 1,
            censor_indexes,
        )

        # for those beyond the end point, censor_area = 0
        beyond_idx = censor_indexes > len(self.area_times) - 2
        censor_area = np.zeros_like(censor_times).astype(float)
        # trapzoidal rule:  (x1 - x0) * (f(x0) + f(x1)) * 0.5
        censor_area[~beyond_idx] = ((self.area_times[censor_indexes[~beyond_idx]] - censor_times[~beyond_idx]) *
                                    (self.area_probabilities[censor_indexes[~beyond_idx]] + surv_prob[~beyond_idx])
                                    * 0.5)
        censor_area[~beyond_idx] += self.area[censor_indexes[~beyond_idx]]
        return censor_times + censor_area / surv_prob
    
class DDC:
    def __init__(self):
        pass
    
    def binned_dist(self, survival_curves, times, censored, num_bins=10):
        '''Bin estimated survival probabilities at true time points into "num_bins"
        equal length intervals between 0 and 1.'''
        times = times.astype(int)
        bin_edges = np.append(np.arange(0, 1.00, 1/num_bins), [1])
        survival_probs = [survival_curves[i][times[i]] for i in range(len(censored)) if not censored[i]]
        intervals = np.array([bisect.bisect_right(bin_edges, p) for p in survival_probs])
        counts = Counter(intervals)

        # # Handle the case where the last interval has overflow
        # max_interval = len(bin_edges) - 1  # The last index in bins array
        # if max_interval in counts:
        #     counts[max_interval - 1] += counts[max_interval]
        #     del counts[max_interval]

        # Lump together 0 survival probability bin with first bin
        if 0 in counts:
            counts[1] += counts[0]
            del counts[0]

        # Ensure all bins have a count, including potentially empty ones
        for num in range(1, num_bins + 1):
            if num not in counts:
                counts[num] = 0
        
        return [counts[val] for val in sorted(counts)]

    
    def ddc_cal(self,p, q):
        '''Calculate DDC between two binned arrays'''
        p, q = np.asarray(p), np.asarray(q)
        ## normalize p, q to probabilities
        p, q = p/p.sum(), q/q.sum()
        ### Calculate entropy between two. DDC can be measured with any divergence measure, 
        #such as KL or others
        return sp.stats.entropy(p,q, base=len(q))
    
class ONE_CALIBRATION:
    def __init__(self,):
        pass
    def one_calibration(
            self,
            predictions: np.ndarray,
            event_time: np.ndarray,
            event_indicator: np.ndarray,
            target_time,
            num_bins: int = 10,
            method: str = "DN"
    ):
        """
        Compute the one calibration score for a given set of predictions and true event times.
        Parameters
        ----------
        predictions: np.ndarray
            The predicted probabilities at the time of interest.
        event_time: np.ndarray
            The true event times.
        event_indicator: np.ndarray
            The indicator of whether the event is observed or not.
        target_time: Numeric
            The time of interest.
        num_bins: int
            The number of bins to divide the predictions into.
        method: str
            The method to handle censored patients. The options are: "DN" (default), and "Uncensored".

        Returns
        -------
        score: float
            The one calibration score.
        observed_probabilities: list
            The observed probabilities in each bin.
        expected_probabilities: list
            The expected probabilities in each bin.
        """
        predictions = 1 - predictions
        sorted_idx = np.argsort(-predictions)
        sorted_predictions = predictions[sorted_idx]
        sorted_event_time = event_time[sorted_idx]
        sorted_event_indicator = event_indicator[sorted_idx]

        binned_event_time = np.array_split(sorted_event_time, num_bins)
        binned_event_indicator = np.array_split(sorted_event_indicator, num_bins)
        binned_predictions = np.array_split(sorted_predictions, num_bins)

        hl_statistics = 0
        observed_probabilities = []
        expected_probabilities = []
        for b in range(num_bins):
            # mean_prob = np.mean(binned_predictions[b])
            bin_size = len(binned_event_time[b])

            # For Uncensored method, we simply remove the censored patients,
            # for D'Agostina-Nam method, we will use 1-KM(t) as the observed probability.
            if method == "Uncensored":
                filter_idx = ~((binned_event_time[b] < target_time) & (binned_event_indicator[b] == 0))
                mean_prob = np.mean(binned_predictions[b][filter_idx])
                event_count = sum(binned_event_time[b][filter_idx] < target_time)
                event_probability = event_count / bin_size
                hl_statistics += (event_count - bin_size * mean_prob) ** 2 / (
                        bin_size * mean_prob * (1 - mean_prob))
            elif method == "DN":
                mean_prob = np.mean(binned_predictions[b])
                km_model = KaplanMeier(binned_event_time[b], binned_event_indicator[b])
                event_probability = 1 - km_model.predict(target_time)
                hl_statistics += (bin_size * event_probability - bin_size * mean_prob) ** 2 / (bin_size * mean_prob * (1 - mean_prob))
            else:
                error = "Please enter one of 'Uncensored','DN' for method."
                raise TypeError(error)
            observed_probabilities.append(event_probability)
            expected_probabilities.append(mean_prob)

        degree_of_freedom = num_bins - 1 if (num_bins <= 15 and method == "DN") else num_bins - 2
        p_value = 1 - chi2.cdf(hl_statistics, degree_of_freedom)

        return p_value, observed_probabilities, expected_probabilities

class CI:
    def __init__(self):
        pass
    
    def concordance_cal(
        self,
        predicted_survival_curves: np.ndarray,
        event_time: np.ndarray,
        event_indicator: np.ndarray,
        ties: str = "None",
        predicted_time_method: str = "Median"
    ):
        event_time = event_time.squeeze(1)
        event_indicator = event_indicator.squeeze(1)

        if predicted_time_method == "Median":
            predict_method = self.predict_median_survival_time
        elif predicted_time_method == "Mean":
            predict_method = self.predict_mean_survival_time
        else:
            error = "Please enter one of 'Median' or 'Mean' for calculating predicted survival time."
            raise TypeError(error)
        
        # get median/mean survival time from the predicted curve
        predicted_times = []
        for i in range(predicted_survival_curves.shape[0]):
            predicted_time = predict_method(predicted_survival_curves[i], np.arange(0, predicted_survival_curves.shape[1]))
            predicted_times.append(predicted_time)
        predicted_times = np.array(predicted_times)

        return self.concordance(predicted_times, event_time, event_indicator, ties=ties)

    def predict_median_survival_time(
            self,
            survival_curve: np.ndarray,
            times_coordinate: np.ndarray,
            interpolation: str = "Linear"
        ) -> float:
            """
            Get the median survival time from the survival curve. The median survival time is defined as the time point where
            the survival curve crosses 0.5. The curve is first interpolated by the given monotonic cubic interpolation method
            (Linear or Pchip). Then the curve gets extroplated by the linear function of (0, 1) and the last time point. The
            median survival time is calculated by finding the time point where the survival curve crosses 0.5.
            Parameters
            ----------
            survival_curve: np.ndarray
                The survival curve of the sample. 1-D array.
            times_coordinate: np.ndarray
                The time coordinate of the survival curve. 1-D array.
            interpolation: str
                The monotonic cubic interpolation method. One of ['Linear', 'Pchip']. Default: 'Linear'.
                If 'Linear', use the interp1d method from scipy.interpolate.
                If 'Pchip', use the PchipInterpolator from scipy.interpolate.
            Returns
            -------
            median_survival_time: float
                The median survival time.
            """
            # If all the predicted probabilities are 1 the integral will be infinite.
            if np.all(survival_curve == 1):
                warnings.warn("All the predicted probabilities are 1, the median survival time will be infinite.")
                return np.inf

            min_prob = float(min(survival_curve))

            if 0.5 in survival_curve:
                median_probability_time = times_coordinate[np.where(survival_curve == 0.5)[0][0]]
            elif min_prob < 0.5:
                if len(np.where(survival_curve > 0.5)[0])!=0:
                    idx_before_median = np.where(survival_curve > 0.5)[0][-1]
                    idx_after_median = np.where(survival_curve < 0.5)[0][0]
                    min_time_before_median = times_coordinate[idx_before_median]
                    max_time_after_median = times_coordinate[idx_after_median]

                    if interpolation == "Linear":
                        # given last time before median and first time after median, solve the linear equation
                        slope = ((survival_curve[idx_after_median] - survival_curve[idx_before_median]) /
                                (max_time_after_median - min_time_before_median))
                        intercept = survival_curve[idx_before_median] - slope * min_time_before_median
                        median_probability_time = (0.5 - intercept) / slope
                    elif interpolation == "Pchip":
                        # reverse the array because the PchipInterpolator requires the x to be strictly increasing
                        spline = self.interpolated_survival_curve(times_coordinate, survival_curve, interpolation)
                        time_range = np.linspace(min_time_before_median, max_time_after_median, num=1000)
                        prob_range = spline(time_range)
                        inverse_spline = PchipInterpolator(prob_range[::-1], time_range[::-1])
                        median_probability_time = np.array(inverse_spline(0.5)).item()
                    else:
                        raise ValueError("interpolation should be one of ['Linear', 'Pchip']")
                else:
                    max_time = float(max(times_coordinate))
                    min_prob = float(min(survival_curve))
                    slope = (1 - min_prob) / (0 - max_time)
                    median_probability_time = - 0.5 / slope
            else:
                max_time = float(max(times_coordinate))
                min_prob = float(min(survival_curve))
                slope = (1 - min_prob) / (0 - max_time)
                median_probability_time = - 0.5 / slope

            return median_probability_time
    
    def predict_mean_survival_time(
        self,
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray,
        interpolation: str = "Linear"
    ) -> float:
        """
        Get the mean survival time from the survival curve. The mean survival time is defined as the area under the survival
        curve. The curve is first interpolated by the given monotonic cubic interpolation method (Linear or Pchip). Then the
        curve gets extroplated by the linear function of (0, 1) and the last time point. The area is calculated by the
        trapezoidal rule.
        Parameters
        ----------
        survival_curve: np.ndarray
            The survival curve of the sample. 1-D array.
        times_coordinate: np.ndarray
            The time coordinate of the survival curve. 1-D array.
        interpolation: str
            The monotonic cubic interpolation method. One of ['Linear', 'Pchip']. Default: 'Linear'.
            If 'Linear', use the interp1d method from scipy.interpolate.
            If 'Pchip', use the PchipInterpolator from scipy.interpolate.
        Returns
        -------
        mean_survival_time: float
            The mean survival time.
        """
        # If all the predicted probabilities are 1 the integral will be infinite.
        if np.all(survival_curve == 1):
            warnings.warn("All the predicted probabilities are 1, the integral will be infinite.")
            return np.inf

        spline = self.interpolated_survival_curve(times_coordinate, survival_curve, interpolation)

        # predicting boundary
        max_time = float(max(times_coordinate))

        # simply calculate the slope by using the [0, 1] - [max_time, S(t|x)]
        slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

        # zero_probability_time = min(times_coordinate[np.where(survival_curve == 0)],
        #                             max_time + (0 - np.array(spline(max_time)).item()) / slope)
        if 0 in survival_curve:
            zero_probability_time = min(times_coordinate[np.where(survival_curve == 0)])
        else:
            zero_probability_time = max_time + (0 - np.array(spline(max_time)).item()) / slope

        def _func_to_integral(time, maximum_time, slope_rate):
            return np.array(spline(time)).item() if time < maximum_time else (1 + time * slope_rate)
        # _func_to_integral = lambda time: spline(time) if time < max_time else (1 + time * slope)
        # limit controls the subdivision intervals used in the adaptive algorithm.
        # Set it to 1000 is consistent with Haider's R code
        mean_survival_time, *rest = integrate.quad(_func_to_integral, 0, zero_probability_time,
                                                args=(max_time, slope), limit=1000)
        return mean_survival_time
    
    def interpolated_survival_curve(self, times_coordinate, survival_curve, interpolation):
        if interpolation == "Linear":
            spline = interp1d(times_coordinate, survival_curve, kind='linear', fill_value='extrapolate')
        elif interpolation == "Pchip":
            spline = PchipInterpolator(times_coordinate, survival_curve)
        else:
            raise ValueError("interpolation must be one of ['Linear', 'Pchip']")
        return spline
    
    def concordance(
            self,
            predicted_times: np.ndarray,
            event_times: np.ndarray,
            event_indicators: np.ndarray,
            train_event_times: Optional[np.ndarray] = None,
            train_event_indicators: Optional[np.ndarray] = None,
            pair_method: str = "Comparable",
            ties: str = "Risk"
    ):
        """
        Calculate the concordance index between the predicted survival times and the true survival times.
        param predicted_times: array-like, shape = (n_samples,)
            The predicted survival times.
        param event_times: array-like, shape = (n_samples,)
            The true survival times.
        param event_indicators: array-like, shape = (n_samples,)
            The event indicators of the true survival times.
        param train_event_times: array-like, shape = (n_train_samples,)
            The true survival times of the training set.
        param train_event_indicators: array-like, shape = (n_train_samples,)
            The event indicators of the true survival times of the training set.
        param pair_method: str, optional (default="Comparable")
            A string indicating the method for constructing the pairs of samples.
            "Comparable": the pairs are constructed by comparing the predicted survival time of each sample with the
            event time of all other samples. The pairs are only constructed between samples with comparable
            event times. For example, if sample i has a censor time of 10, then the pairs are constructed by
            comparing the predicted survival time of sample i with the event time of all samples with event
            time of 10 or less.
            "Margin": the pairs are constructed between all samples. A best-guess time for the censored samples
            will be calculated and used to construct the pairs.
        param ties: str, optional (default="Risk")
            A string indicating the way ties should be handled.
            Options: "None" (default), "Time", "Risk", or "All"
            "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
            "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
            "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
            "All" includes all ties.
            Note the concordance calculation is given by
            (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
        :return: (float, float, int)
            The concordance index, the number of concordant pairs, and the number of total pairs.
        """
        # the scikit-survival concordance function only takes risk scores to calculate.
        # So at first we should transfer the predicted time -> risk score.
        # The risk score should be higher for subjects that live shorter (i.e. lower average survival time).

        event_indicators = event_indicators.astype(bool)

        if pair_method == "Comparable":
            risks = -1 * predicted_times
            partial_weights = None
            bg_event_times = None
        elif pair_method == "Margin":
            if train_event_times is None or train_event_indicators is None:
                error = "If 'Margin' is chosen, training set information must be provided."
                raise ValueError(error)

            train_event_indicators = train_event_indicators.astype(bool)

            km_model = KaplanMeierArea(train_event_times, train_event_indicators)
            km_linear_zero = -1 / ((1 - min(km_model.survival_probabilities))/(0 - max(km_model.survival_times)))
            if np.isinf(km_linear_zero):
                km_linear_zero = max(km_model.survival_times)
            predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)
            risks = -1 * predicted_times

            censor_times = event_times[~event_indicators]
            partial_weights = np.ones_like(event_indicators, dtype=float)
            partial_weights[~event_indicators] = 1 - km_model.predict(censor_times)

            best_guesses = km_model.best_guess(censor_times)
            best_guesses[censor_times > km_linear_zero] = censor_times[censor_times > km_linear_zero]

            bg_event_times = np.copy(event_times)
            bg_event_times[~event_indicators] = best_guesses
        else:
            raise TypeError("Method for calculating concordance is unrecognized.")
        # risk_ties means predicted times are the same while true times are different.
        # time_ties means true times are the same while predicted times are different.
        # cindex, concordant_pairs, discordant_pairs, risk_ties, time_ties = metrics.concordance_index_censored(
        #     event_indicators, event_times, estimate=risk)
        cindex, concordant_pairs, discordant_pairs, risk_ties, time_ties = self._estimate_concordance_index(
            event_indicators, event_times, estimate=risks, bg_event_time=bg_event_times, partial_weights=partial_weights)
        if ties == "None":
            total_pairs = concordant_pairs + discordant_pairs
            cindex = concordant_pairs / total_pairs
        elif ties == "Time":
            total_pairs = concordant_pairs + discordant_pairs + time_ties
            concordant_pairs = concordant_pairs + 0.5 * time_ties
            cindex = concordant_pairs / total_pairs
        elif ties == "Risk":
            # This should be the same as original outputted cindex from above
            total_pairs = concordant_pairs + discordant_pairs + risk_ties
            concordant_pairs = concordant_pairs + 0.5 * risk_ties
            cindex = concordant_pairs / total_pairs
        elif ties == "All":
            total_pairs = concordant_pairs + discordant_pairs + risk_ties + time_ties
            concordant_pairs = concordant_pairs + 0.5 * (risk_ties + time_ties)
            cindex = concordant_pairs / total_pairs
        else:
            error = "Please enter one of 'None', 'Time', 'Risk', or 'All' for handling ties for concordance."
            raise TypeError(error)

        return cindex, concordant_pairs, total_pairs


    def _estimate_concordance_index(
            self,
            event_indicator: np.ndarray,
            event_time: np.ndarray,
            estimate: np.ndarray,
            bg_event_time: np.ndarray = None,
            partial_weights: np.ndarray = None,
            tied_tol: float = 1e-8
    ):
        order = np.argsort(event_time, kind="stable")

        comparable, tied_time, weight = self._get_comparable(event_indicator, event_time, order)

        if partial_weights is not None:
            event_indicator = np.ones_like(event_indicator)
            comparable_2, tied_time, weight = self._get_comparable(event_indicator, bg_event_time, order, partial_weights)
            for ind, mask in comparable.items():
                weight[ind][mask] = 1
            comparable = comparable_2

        if len(comparable) == 0:
            raise ValueError("Data has no comparable pairs, cannot estimate concordance index.")

        concordant = 0
        discordant = 0
        tied_risk = 0
        numerator = 0.0
        denominator = 0.0
        for ind, mask in comparable.items():
            est_i = estimate[order[ind]]
            event_i = event_indicator[order[ind]]
            # w_i = partial_weights[order[ind]] # change this
            w_i = weight[ind]
            weight_i = w_i[order[mask]]

            est = estimate[order[mask]]

            assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

            ties = np.absolute(est - est_i) <= tied_tol
            # n_ties = ties.sum()
            n_ties = np.dot(weight_i, ties.T)
            # an event should have a higher score
            con = est < est_i
            # n_con = con[~ties].sum()
            con[ties] = False
            n_con = np.dot(weight_i, con.T)

            # numerator += w_i * n_con + 0.5 * w_i * n_ties
            # denominator += w_i * mask.sum()
            numerator += n_con + 0.5 * n_ties
            denominator += np.dot(w_i, mask.T)

            tied_risk += n_ties
            concordant += n_con
            # discordant += est.size - n_con - n_ties
            discordant += np.dot(w_i, mask.T) - n_con - n_ties

        cindex = numerator / denominator
        return cindex, concordant, discordant, tied_risk, tied_time


    def _get_comparable(self, event_indicator: np.ndarray, event_time: np.ndarray, order: np.ndarray,
                        partial_weights: np.ndarray = None):
        if partial_weights is None:
            partial_weights = np.ones_like(event_indicator, dtype=float)
        n_samples = len(event_time)
        tied_time = 0
        comparable = {}
        weight = {}

        i = 0
        while i < n_samples - 1:
            time_i = event_time[order[i]]
            end = i + 1
            while end < n_samples and event_time[order[end]] == time_i:
                end += 1

            # check for tied event times
            event_at_same_time = event_indicator[order[i:end]]
            censored_at_same_time = ~event_at_same_time

            for j in range(i, end):
                if event_indicator[order[j]]:
                    mask = np.zeros(n_samples, dtype=bool)
                    mask[end:] = True
                    # an event is comparable to censored samples at same time point
                    mask[i:end] = censored_at_same_time
                    comparable[j] = mask
                    tied_time += censored_at_same_time.sum()
                    weight[j] = partial_weights[order] * partial_weights[order[j]]
            i = end

        return comparable, tied_time, weight
    
class DCAL:
    def __init__(self,):
        pass
    def d_calibration(
            self, 
            predict_probs,
            event_indicators,
            num_bins: int = 10
    ):
        
        quantile = np.linspace(1, 0, num_bins + 1)
        censor_indicators = 1 - event_indicators
        event_probabilities = predict_probs[event_indicators.astype(bool)]
        event_position = np.digitize(event_probabilities, quantile)
        event_position[event_position == 0] = 1     # class probability==1 to the first bin

        event_binning = np.zeros([num_bins])
        for i in range(len(event_position)):
            event_binning[event_position[i] - 1] += 1
        
        censored_probabilities = predict_probs[censor_indicators.astype(bool)]

        censor_binning = np.zeros([num_bins])
        if len(censored_probabilities) > 0:
            for i in range(len(censored_probabilities)):
                partial_binning = self.create_censor_binning(censored_probabilities[i], num_bins)
                censor_binning += partial_binning

        combine_binning = event_binning + censor_binning
        _, pvalue = chisquare(combine_binning)
        return pvalue, combine_binning


    def create_censor_binning(
            self,
            probability: float,
            num_bins: int
    ) -> np.ndarray:
        """
        For censoring instance,
        b1 will be the infimum probability of the bin that contains S(c),
        for the bin of [b1, b2) which contains S(c), probability = (S(c) - b1) / S(c)
        for the rest of the bins, [b2, b3), [b3, b4), etc., probability = 1 / (B * S(c)), where B is the number of bins.
        :param probability: float
            The predicted probability at the censored time of a censoring instance.
        :param num_bins: int
            The number of bins to use for the D-Calibration score.
        :return:
        final_binning: np.ndarray
            The "split" histogram of this censored subject.
        """
        quantile = np.linspace(1, 0, num_bins + 1)
        censor_binning = np.zeros(num_bins)
        for i in range(num_bins):
            if probability == 1:
                censor_binning += 0.1
                break
            elif quantile[i] > probability >= quantile[i + 1]:
                first_bin = (probability - quantile[i + 1]) / probability if probability != 0 else 1
                rest_bins = 1 / (num_bins * probability) if probability != 0 else 0
                censor_binning[i] += first_bin
                censor_binning[i + 1:] += rest_bins
                break
        # assert len(censor_binning) == num_bins, f"censor binning should have size of {num_bins}"
        return censor_binning


